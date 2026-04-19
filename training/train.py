"""Two-phase training script for GaussianFormer.

Phase 1: Freeze backbone (pretrained from RenderFormer), train only the Gaussian input module.
Phase 2: Unfreeze all parameters and fine-tune the entire model.

Usage:
    uv run python -m training.train
    uv run python -m training.train --phase1_epochs 20 --phase2_epochs 50 --resolution 256
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from gaussianformer.utils.ray_generator import RayGenerator
from gaussianformer.utils.transform import transform_gaussians_to_cam_coord

from training.config import TrainingConfig
from training.dataset import GaussianRenderDataset, collate_fn
from training.weight_transfer import transfer_weights, freeze_backbone, unfreeze_all


def training_forward(
    model: torch.nn.Module,
    ray_generator: RayGenerator,
    gaussians: torch.Tensor,
    mask: torch.Tensor,
    c2w: torch.Tensor,
    fov: torch.Tensor,
    resolution: int,
    config,
) -> torch.Tensor:
    """
    Training-compatible forward pass (no torch.no_grad).

    Replicates GaussianFormerRenderingPipeline.render() but allows gradients
    to flow through the model.

    Returns:
        rendered_imgs: [bs, 1, H, W, 3] in log-HDR space
    """
    bs = gaussians.shape[0]
    nv = 1  # single view per sample in training

    # Add view dimension: [bs, 4, 4] -> [bs, 1, 4, 4]
    c2w = c2w.unsqueeze(1)
    fov = fov.unsqueeze(-1).unsqueeze(-1)  # [bs] -> [bs, 1, 1]

    # Camera coordinate transform (detached -- no grad through the rigid transform)
    if config.turn_to_cam_coord:
        with torch.no_grad():
            c2w_flat = c2w.reshape(-1, 4, 4)
            gaussians_repeated = gaussians  # nv=1, no repeat needed
            gaussians_for_view_tf, c2w_for_view_tf = transform_gaussians_to_cam_coord(
                c2w_flat, gaussians_repeated
            )
            c2w_for_view_tf = c2w_for_view_tf.reshape(bs, nv, 4, 4)
            gaussians_for_view_tf = gaussians_for_view_tf.reshape(bs, nv, -1, config.gaussian_dim)
    else:
        gaussians_for_view_tf = gaussians.unsqueeze(1)
        c2w_for_view_tf = c2w

    # Generate rays (detached -- rays don't need gradients)
    with torch.no_grad():
        rays_o, rays_d = ray_generator(c2w_for_view_tf, fov / 180.0 * torch.pi, resolution)

    # Model forward (gradients flow here)
    rendered_imgs = model(
        gaussians=gaussians,
        valid_mask=mask,
        rays_o=rays_o,
        rays_d=rays_d,
        gaussians_view_tf=gaussians_for_view_tf[..., :config.pos_dim],
        tf32_view_tf=True,
    )

    # [bs, nv, C, H, W] -> [bs, nv, H, W, C]
    rendered_imgs = rendered_imgs.permute(0, 1, 3, 4, 2)
    return rendered_imgs


def compute_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 in log-HDR space. pred is log10(hdr+1); target is LDR in [0,1]."""
    return F.l1_loss(pred.squeeze(1), torch.log10(target + 1.0))


def run_phase(
    phase_name: str,
    model: torch.nn.Module,
    ray_generator: RayGenerator,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config,
    num_epochs: int,
    device: torch.device,
    save_dir: Path,
    log_interval: int,
    save_interval: int,
    global_step: int = 0,
    val_dataloader: DataLoader | None = None,
) -> int:
    """Run a training phase (shared logic for phase 1 and 2)."""
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}", flush=True)
    print(f"{phase_name}: {trainable:,} / {total:,} trainable parameters", flush=True)
    print(f"{'='*60}\n", flush=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t0 = time.time()

        for batch in dataloader:
            gaussians = batch["gaussians"].to(device)
            mask = batch["mask"].to(device)
            c2w = batch["c2w"].to(device)
            fov = batch["fov"].to(device)
            target = batch["target"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = training_forward(
                    model, ray_generator, gaussians, mask, c2w, fov,
                    config.resolution, model.config,
                )
                loss = compute_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            if global_step % log_interval == 0:
                print(
                    f"  [{phase_name}] step {global_step}, loss: {loss.item():.6f}",
                    flush=True,
                )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        avg_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - t0
        print(
            f"[{phase_name}] Epoch {epoch + 1}/{num_epochs}, "
            f"avg loss: {avg_loss:.6f}, lr: {current_lr:.2e}, time: {elapsed:.1f}s",
            flush=True,
        )

        # Validation
        if val_dataloader is not None and (epoch + 1) % save_interval == 0:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for batch in val_dataloader:
                    gaussians = batch["gaussians"].to(device)
                    mask = batch["mask"].to(device)
                    c2w = batch["c2w"].to(device)
                    fov = batch["fov"].to(device)
                    target = batch["target"].to(device)

                    pred = training_forward(
                        model, ray_generator, gaussians, mask, c2w, fov,
                        config.resolution, model.config,
                    )
                    val_loss += compute_loss(pred, target).item()
                    val_steps += 1

            avg_val_loss = val_loss / max(val_steps, 1)
            print(
                f"[{phase_name}] Epoch {epoch + 1}/{num_epochs}, val loss: {avg_val_loss:.6f}",
                flush=True,
            )

        if (epoch + 1) % save_interval == 0:
            ckpt_path = save_dir / f"{phase_name}_epoch_{epoch + 1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}", flush=True)

    return global_step


def main():
    parser = argparse.ArgumentParser(description="Train GaussianFormer")
    parser.add_argument("--renderformer_model_id", type=str, default=TrainingConfig.renderformer_model_id)
    parser.add_argument("--gaussian_h5_dir", type=Path, default=TrainingConfig.gaussian_h5_dir)
    parser.add_argument("--renders_dir", type=Path, default=TrainingConfig.renders_dir)
    parser.add_argument("--save_dir", type=Path, default=TrainingConfig.save_dir)
    parser.add_argument("--resolution", type=int, default=TrainingConfig.resolution)
    parser.add_argument("--phase1_epochs", type=int, default=TrainingConfig.phase1_epochs)
    parser.add_argument("--phase1_lr", type=float, default=TrainingConfig.phase1_lr)
    parser.add_argument("--phase2_epochs", type=int, default=TrainingConfig.phase2_epochs)
    parser.add_argument("--phase2_lr", type=float, default=TrainingConfig.phase2_lr)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--skip_phase1", action="store_true", help="Skip phase 1 and go directly to phase 2")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset to N samples (for quick experiments)")
    parser.add_argument("--resume", type=Path, help="Path to checkpoint to resume from")
    parser.add_argument("--min_lr_ratio", type=float, default=TrainingConfig.min_lr_ratio)
    parser.add_argument("--save_interval", type=int, default=TrainingConfig.save_interval, help="Save checkpoint every N epochs")
    parser.add_argument("--val_h5_dir", type=Path, default=None, help="Validation H5 directory")
    parser.add_argument("--val_renders_dir", type=Path, default=None, help="Validation renders directory")
    args = parser.parse_args()

    config = TrainingConfig(
        renderformer_model_id=args.renderformer_model_id,
        gaussian_h5_dir=args.gaussian_h5_dir,
        renders_dir=args.renders_dir,
        save_dir=args.save_dir,
        resolution=args.resolution,
        phase1_epochs=args.phase1_epochs,
        phase1_lr=args.phase1_lr,
        phase2_epochs=args.phase2_epochs,
        phase2_lr=args.phase2_lr,
        min_lr_ratio=args.min_lr_ratio,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # --- Dataset ---
    dataset = GaussianRenderDataset(
        config.gaussian_h5_dir, config.renders_dir, config.resolution,
        max_samples=args.max_samples,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    # --- Validation dataset ---
    val_dataloader = None
    if args.val_h5_dir and args.val_renders_dir:
        val_dataset = GaussianRenderDataset(
            args.val_h5_dir, args.val_renders_dir, config.resolution,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_fn,
        )
        print(f"Validation set: {len(val_dataset)} samples", flush=True)

    # --- Model ---
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}", flush=True)
        from gaussianformer.models.gaussianformer import GaussianFormer
        from gaussianformer.models.config import GaussianFormerConfig
        model = GaussianFormer(GaussianFormerConfig())
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model = transfer_weights(config.renderformer_model_id)

    model.to(device)
    ray_generator = RayGenerator().to(device)

    config.save_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0

    # --- Phase 1: Train input module only ---
    if not args.skip_phase1:
        trainable_names = freeze_backbone(model)
        print(f"Phase 1 trainable params: {trainable_names}", flush=True)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.phase1_lr,
            weight_decay=0.01,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.phase1_epochs,
            eta_min=config.phase1_lr * config.min_lr_ratio,
        )

        global_step = run_phase(
            "phase1", model, ray_generator, dataloader, optimizer, scheduler,
            config, config.phase1_epochs, device, config.save_dir,
            config.log_interval, args.save_interval, global_step,
            val_dataloader=val_dataloader,
        )

    # --- Phase 2: Full fine-tune ---
    unfreeze_all(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.phase2_lr,
        weight_decay=0.01,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.phase2_epochs,
        eta_min=config.phase2_lr * config.min_lr_ratio,
    )

    global_step = run_phase(
        "phase2", model, ray_generator, dataloader, optimizer, scheduler,
        config, config.phase2_epochs, device, config.save_dir,
        config.log_interval, args.save_interval, global_step,
        val_dataloader=val_dataloader,
    )

    # --- Save final model ---
    final_path = config.save_dir / "gaussianformer_final"
    final_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_path)
    print(f"\nFinal model saved to: {final_path}", flush=True)
    print(f"Use with: python infer_gaussian.py --model_id {final_path}", flush=True)


if __name__ == "__main__":
    main()
