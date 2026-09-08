"""Two-phase training script for GaussianFormer.

Phase 1: Freeze backbone (pretrained from RenderFormer), train only the Gaussian input module.
Phase 2: Unfreeze all parameters and fine-tune the entire model.

Usage:
    uv run python -m training.train
    uv run python -m training.train --phase1_epochs 20 --phase2_epochs 50 --resolution 256
"""

import argparse
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from gaussianformer.utils.ray_generator import RayGenerator
from gaussianformer.utils.transform import transform_gaussians_to_cam_coord

from training.config import TrainingConfig
from training.dataset import GaussianRenderDataset, collate_fn
from training.weight_transfer import transfer_weights, freeze_backbone, unfreeze_all


def setup_ddp() -> tuple[int, int, int, torch.device]:
    """Init NCCL when launched via torchrun. Returns (rank, local_rank, world_size, device).

    Single-GPU path: when LOCAL_RANK is unset, returns (0, 0, 1, cuda) and skips
    process-group init -- the rest of the script then behaves exactly like before.
    """
    if "LOCAL_RANK" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 0, 1, device
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size, torch.device(f"cuda:{local_rank}")


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def unwrap_model(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if isinstance(m, DDP) else m


def wrap_ddp(module: torch.nn.Module, world_size: int, local_rank: int) -> torch.nn.Module:
    """Wrap `module` in DDP for the current trainable set, or return it bare on 1 GPU.

    Constructed once per phase, AFTER requires_grad is set: DDP's reducer registers
    only the params that require grad at construction time, so every registered param
    produces a gradient every step and find_unused_parameters=False is correct.
    """
    if world_size > 1:
        return DDP(module, device_ids=[local_rank], output_device=local_rank,
                   find_unused_parameters=False)
    return module


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


_lpips_fn = None


def get_lpips(device: torch.device) -> torch.nn.Module:
    """Lazy-init LPIPS-VGG. Frozen, eval mode."""
    global _lpips_fn
    if _lpips_fn is None:
        import lpips
        _lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
        for p in _lpips_fn.parameters():
            p.requires_grad = False
    return _lpips_fn


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    log_w: float,
    lpips_w: float,
    device: torch.device,
    fg_bg_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (total, log_term, lpips_term).

    pred: [bs, 1, H, W, 3] in log10(hdr+1) space.
    target: [bs, H, W, 3] LDR PNG values in [0, 1].

    log_term = L1(pred, log10(target + 1))  -- v6 baseline.
    lpips_term = LPIPS-VGG(pred_ldr, target) on display-space values in [-1, 1].
                 pred_ldr = clamp(10^pred - 1, 0, 1) brings the log-HDR prediction
                 back to the same display-referred space the PNG target lives in.

    fg_bg_weight < 1 down-weights BACKGROUND pixels (target luminance <= 0.02, the
    same threshold as the eval-side _fg_crop) in the log term. Objects cover 2-7% of
    pixels, so the plain mean dilutes the object's gradient 14-50x; at 0.05 the
    foreground carries ~60% of the log term instead of ~7%. LPIPS is left whole-image
    (its conv stats are not meaningfully maskable). 1.0 = exact baseline behavior.
    """
    log_target = torch.log10(target + 1.0)
    if fg_bg_weight != 1.0:
        fg = (target.amax(dim=-1, keepdim=True) > 0.02).float()
        w = (fg + (1.0 - fg) * fg_bg_weight).expand_as(log_target)
        log_term = (w * (pred.squeeze(1) - log_target).abs()).sum() / w.sum()
    else:
        log_term = F.l1_loss(pred.squeeze(1), log_target)
    total = log_w * log_term

    lpips_term = torch.tensor(0.0, device=device)
    if lpips_w > 0:
        pred_ldr = torch.clamp(10.0 ** pred.squeeze(1) - 1.0, 0.0, 1.0)
        p = pred_ldr.permute(0, 3, 1, 2) * 2.0 - 1.0
        g = target.permute(0, 3, 1, 2) * 2.0 - 1.0
        lpips_term = get_lpips(device)(p, g).mean()
        total = total + lpips_w * lpips_term

    return total, log_term, lpips_term


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
    log_loss_weight: float,
    lpips_loss_weight: float,
    fg_bg_weight: float,
    model_config,
    global_step: int = 0,
    val_dataloader: DataLoader | None = None,
    train_sampler: DistributedSampler | None = None,
    start_epoch: int = 0,
    keep_last_n: int | None = None,
    grad_accum: int = 1,
) -> int:
    """Run a training phase (shared logic for phase 1 and 2)."""
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if is_main_process():
        print(f"\n{'='*60}", flush=True)
        print(f"{phase_name}: {trainable:,} / {total:,} trainable parameters", flush=True)
        print(f"{'='*60}\n", flush=True)

    for epoch in range(start_epoch, num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        # Redraw this epoch's view subsample (no-op unless views_per_epoch is set). Must run
        # before the dataloader spawns workers so the fresh forks inherit this selection.
        dataloader.dataset.set_epoch(epoch)
        model.train()
        # On-device accumulators: the old `epoch_x += term.item()` forced a GPU sync on every
        # step (three of them). Sums stay on-device; the only per-step sync left is the
        # log_interval print.
        epoch_sums = torch.zeros(3, device=device)
        epoch_steps = 0
        t0 = time.time()

        optimizer.zero_grad()
        for micro, batch in enumerate(dataloader):
            b = {k: batch[k].to(device, non_blocking=True)
                 for k in ("gaussians", "mask", "c2w", "fov", "target")}

            # Gradient accumulation: `grad_accum` micro-batches per optimizer step, so
            # world_size x grad_accum x batch_size is the effective batch (e.g. 4 GPUs x 2 == 8 GPUs x 1).
            # DDP all-reduce is skipped on non-final micro-steps via no_sync().
            is_last_micro = (micro + 1) % grad_accum == 0
            sync_ctx = (model.no_sync() if (grad_accum > 1 and not is_last_micro
                                            and hasattr(model, "no_sync")) else nullcontext())
            with sync_ctx:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = training_forward(
                        model, ray_generator, b["gaussians"], b["mask"], b["c2w"], b["fov"],
                        config.resolution, model_config,
                    )
                    loss, log_term, lpips_term = compute_loss(
                        pred, b["target"], log_loss_weight, lpips_loss_weight, device,
                        fg_bg_weight=fg_bg_weight,
                    )
                (loss / grad_accum).backward()

            with torch.no_grad():
                epoch_sums += torch.stack([loss.detach(), log_term.detach(), lpips_term.detach()])
            epoch_steps += 1
            if not is_last_micro:
                continue

            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % log_interval == 0 and is_main_process():
                print(
                    f"  [{phase_name}] step {global_step}, total: {loss.item():.6f}, "
                    f"log: {log_term.item():.6f}, lpips: {lpips_term.item():.6f}",
                    flush=True,
                )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        # One cross-rank reduce + one sync per epoch. AVG of per-rank sums equals the global
        # average because DistributedSampler pads ranks to identical step counts.
        if dist.is_initialized():
            dist.all_reduce(epoch_sums, op=dist.ReduceOp.AVG)
        avg_loss, avg_log, avg_lpips = (epoch_sums / max(epoch_steps, 1)).tolist()
        elapsed = time.time() - t0
        if is_main_process():
            print(
                f"[{phase_name}] Epoch {epoch + 1}/{num_epochs}, "
                f"avg total: {avg_loss:.6f}, log: {avg_log:.6f}, lpips: {avg_lpips:.6f}, "
                f"lr: {current_lr:.2e}, time: {elapsed:.1f}s",
                flush=True,
            )

        # Validation
        if val_dataloader is not None and (epoch + 1) % save_interval == 0:
            model.eval()
            val_sums = torch.zeros(3, device=device)
            val_steps = 0
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for batch in val_dataloader:
                    b = {k: batch[k].to(device, non_blocking=True)
                         for k in ("gaussians", "mask", "c2w", "fov", "target")}
                    pred = training_forward(
                        model, ray_generator, b["gaussians"], b["mask"], b["c2w"], b["fov"],
                        config.resolution, model_config,
                    )
                    total, log_term, lpips_term = compute_loss(
                        pred, b["target"], log_loss_weight, lpips_loss_weight, device,
                        fg_bg_weight=fg_bg_weight,
                    )
                    val_sums += torch.stack([total, log_term, lpips_term])
                    val_steps += 1

            if dist.is_initialized():
                dist.all_reduce(val_sums, op=dist.ReduceOp.AVG)
            avg_val_loss, avg_val_log, avg_val_lpips = (val_sums / max(val_steps, 1)).tolist()
            if is_main_process():
                print(
                    f"[{phase_name}] Epoch {epoch + 1}/{num_epochs}, "
                    f"val total: {avg_val_loss:.6f}, log: {avg_val_log:.6f}, "
                    f"lpips: {avg_val_lpips:.6f}",
                    flush=True,
                )

        is_last_epoch = (epoch + 1) == num_epochs
        if ((epoch + 1) % save_interval == 0 or is_last_epoch) and is_main_process():
            ckpt_path = save_dir / f"{phase_name}_epoch_{epoch + 1}.pt"
            # Atomic write: save to .tmp then os.replace, so a preemption mid-write can
            # never leave a half-written checkpoint that --resume would choke on.
            tmp_path = ckpt_path.with_suffix(".pt.tmp")
            torch.save({
                "phase": phase_name,
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": unwrap_model(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss,
            }, tmp_path)
            os.replace(tmp_path, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}", flush=True)

            # Rolling prune: keep only the N most recent same-phase checkpoints.
            if keep_last_n is not None and keep_last_n > 0:
                saved = sorted(
                    save_dir.glob(f"{phase_name}_epoch_*.pt"),
                    key=lambda p: int(p.stem.rsplit("_", 1)[1]),
                )
                for stale in saved[:-keep_last_n]:
                    stale.unlink()
                    print(f"  Pruned old checkpoint: {stale}", flush=True)

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
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Micro-batches per optimizer step (effective batch = world x batch_size x grad_accum)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset to N samples (for quick experiments)")
    parser.add_argument("--resume", type=Path,
                        help="Escape hatch for crash recovery: resume from a phase1/phase2 "
                        "checkpoint (restores phase, epoch, optimizer, scheduler). Not part "
                        "of the normal recipe -- the default path trains end to end.")
    parser.add_argument("--init_from", type=Path,
                        help="Warm-start fine-tune: load ONLY model weights from a checkpoint, "
                        "then run a FRESH Phase 2 (new optimizer + new cosine schedule, "
                        "global_step=0); Phase 1 is skipped. Unlike --resume, nothing about the "
                        "prior schedule/epoch/optimizer is carried over -- use this to fine-tune "
                        "a converged model under a new loss/LR (e.g. the V9->V10b or V13->V13b "
                        "LPIPS fine-tune). Mutually exclusive with --resume.")
    parser.add_argument("--min_lr_ratio", type=float, default=TrainingConfig.min_lr_ratio)
    parser.add_argument("--save_interval", type=int, default=TrainingConfig.save_interval, help="Save checkpoint every N epochs")
    parser.add_argument("--val_h5_dir", type=Path, default=None, help="Validation H5 directory")
    parser.add_argument("--val_renders_dir", type=Path, default=None, help="Validation renders directory")
    parser.add_argument("--log_loss_weight", type=float, default=1.0,
                        help="Weight on the log-HDR L1 term (v6 baseline loss).")
    parser.add_argument("--lpips_loss_weight", type=float, default=0.0,
                        help="Weight on LPIPS-VGG (display-space). 0 = disabled (v6 behavior).")
    parser.add_argument("--encoder_layers", type=int, default=12,
                        help="View-independent encoder depth. Non-default depths need an init "
                        "made by data_v10/make_pruned_ckpt.py (or --from_scratch).")
    parser.add_argument("--view_layers", type=int, default=6,
                        help="View-transformer depth. See --encoder_layers.")
    parser.add_argument("--latent_dim", type=int, default=768,
                        help="Transformer width (encoder + view transformer; FFNs scale 4x). "
                        "Non-default widths cannot load RenderFormer weights -> use --from_scratch.")
    parser.add_argument("--from_scratch", action="store_true",
                        help="Random init (no RenderFormer transfer, no phase 1). Required for "
                        "non-default --latent_dim; use with a matched from-scratch baseline.")
    parser.add_argument("--input_mlp_hidden", type=int, default=0)
    parser.add_argument("--ffn_mult", type=int, default=4)
    parser.add_argument("--log_scale_input", action="store_true",
                        help="Feed log10(scale)+3 instead of raw scales (train AND eval must match).")
    parser.add_argument("--geom_bias", action="store_true",
                        help="Zero-init-gated ray/Gaussian alignment bias on the view transformer's "
                        "cross-attention logits. Loads unbiased checkpoints via --init_from (the "
                        "missing gates init to zero = exact baseline); eval must pass --geom_bias too.")
    parser.add_argument("--fg_bg_weight", type=float, default=1.0,
                        help="Down-weight background pixels (GT luminance <= 0.02) in the log-L1 "
                        "term. 1.0 = whole-image baseline; 0.05 gives the foreground ~60%% of the "
                        "term instead of ~7%%. LPIPS stays whole-image.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="AdamW weight decay for phase 2 (0 disables; used by memorization controls)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader num_workers per rank. None = use TrainingConfig default. "
                        "DDP runs with num_workers=0 saturate one main thread per rank on disk IO; "
                        "set to 4 for prefetching parallelism.")
    parser.add_argument("--pe_type", type=str, default="nerf",
                        choices=["rope", "nerf"],
                        help="Gaussian input encoder. 'nerf' = concat encoder "
                        "(NeRF-encoded position + log-scale); 'rope' = V9 baseline.")
    parser.add_argument("--augment_rotation", action="store_true",
                        help="Apply on-the-fly Haar-uniform scene+camera rotation to the "
                        "TRAIN set (image-preserving; val stays fixed). Teaches rotation "
                        "robustness for free -- the model is otherwise rotation-variant.")
    parser.add_argument("--keep_last_n", type=int, default=None,
                        help="Retain only the N most recent same-phase checkpoints "
                        "(prune older ones after each save). None = keep all (default).")
    parser.add_argument("--model_cfg", nargs="*", default=None, metavar="KEY=VAL",
                        help="GaussianFormerConfig overrides for architecture probes, e.g. "
                             "--model_cfg rope_pos_scale=4 ray_rope_2d=true. With --init_from, "
                             "parameters absent from the seed must be zero-init (warm-safe).")
    parser.add_argument("--views_per_epoch", type=int, default=None,
                        help="Per-epoch view subsampling for the TRAIN set: draw this many "
                        "random views per scene each epoch (redrawn per epoch; all views seen "
                        "across epochs). Shrinks the epoch ~14/K. None = all views (default). "
                        "Val is never subsampled.")
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

    rank, local_rank, world_size, device = setup_ddp()
    num_workers = args.num_workers if args.num_workers is not None else config.num_workers
    if is_main_process():
        from gaussianformer.layers.attention import ATTN as _GF_ATTN
        print(f"Using device: {device}, world_size: {world_size}, "
              f"num_workers (per rank): {num_workers}", flush=True)
        print(f"attention backend: {_GF_ATTN}", flush=True)

    # --- Dataset ---
    dataset = GaussianRenderDataset(
        config.gaussian_h5_dir, config.renders_dir, config.resolution,
        max_samples=args.max_samples,
        augment_rotation=args.augment_rotation,
        views_per_epoch=args.views_per_epoch,
        log_scale_input=args.log_scale_input,
    )
    train_sampler: DistributedSampler | None = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False,
        )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        # With view subsampling the per-epoch selection is set on the dataset just before
        # each epoch; persistent workers would keep a stale fork, so disable in that mode.
        persistent_workers=(num_workers > 0 and not args.views_per_epoch),
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    # --- Validation dataset ---
    val_dataloader = None
    if args.val_h5_dir and args.val_renders_dir:
        val_dataset = GaussianRenderDataset(
            args.val_h5_dir, args.val_renders_dir, config.resolution,
            log_scale_input=args.log_scale_input,
        )
        val_sampler: DistributedSampler | None = None
        if world_size > 1:
            val_sampler = DistributedSampler(
                val_dataset, num_replicas=world_size, rank=rank,
                shuffle=False, drop_last=False,
            )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            pin_memory=device.type == "cuda",
            collate_fn=collate_fn,
        )
        if is_main_process():
            print(f"Validation set: {len(val_dataset)} samples", flush=True)

    # --- Model ---
    from gaussianformer.models.config import GaussianFormerConfig
    from gaussianformer.models.gaussianformer import GaussianFormer
    gf_config = GaussianFormerConfig(
        pe_type=args.pe_type,
        latent_dim=args.latent_dim,
        num_layers=args.encoder_layers,
        view_transformer_n_layers=args.view_layers,
        dim_feedforward=args.latent_dim * args.ffn_mult,
        view_transformer_latent_dim=args.latent_dim,
        view_transformer_ffn_hidden_dim=args.latent_dim * args.ffn_mult,
        input_mlp_hidden=args.input_mlp_hidden,
        geom_bias=args.geom_bias,
    ).with_overrides(args.model_cfg)
    if args.model_cfg and is_main_process():
        print(f"model_cfg overrides: {args.model_cfg}", flush=True)

    # --resume is a phase-aware escape hatch (crash recovery), not the normal recipe:
    # it carries the phase + epoch so training continues from where it stopped. The
    # default path always builds the model from RenderFormer weights.
    resume_ckpt = None
    resume_phase = None
    resume_epoch = 0
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        resume_phase = resume_ckpt["phase"]
        resume_epoch = resume_ckpt["epoch"]
        module = GaussianFormer(gf_config)
        module.load_state_dict(resume_ckpt["model_state_dict"])
        if is_main_process():
            print(f"Resuming from {args.resume} ({resume_phase} epoch {resume_epoch})", flush=True)
    elif args.init_from:
        # Warm-start fine-tune: load weights only. resume_ckpt/resume_phase stay None,
        # so nothing about the prior optimizer/scheduler/epoch is carried over -- the
        # Phase-2 restore block below is not taken (fresh optimizer + fresh cosine from
        # phase2_lr, global_step/start_epoch = 0). Phase 1 is skipped via the gate below.
        init_ckpt = torch.load(args.init_from, map_location="cpu", weights_only=True)
        module = GaussianFormer(gf_config)
        if args.geom_bias or args.model_cfg:
            # Architecture probes add modules the seed lacks. Warm-safe rule: every missing
            # tensor must be zero at init (gates, zero-init linears), so the wrapped model is
            # exactly the seed until training moves it. RoPE-only changes add no tensors.
            sd = module.state_dict()
            # RoPE frequency tables (*.freqs) are config-determined constants stored as
            # non-trainable Parameters: drop the seed's copies so a changed rotary dim loads.
            seed_sd = {k: v for k, v in init_ckpt["model_state_dict"].items()
                       if not (k.endswith(".freqs") and (k not in sd or sd[k].shape != v.shape))}
            missing, unexpected = module.load_state_dict(seed_sd, strict=False)
            assert not unexpected, f"seed has keys the model lacks: {list(unexpected)[:5]}"
            nonzero = [k for k in missing if not k.endswith(".freqs") and sd[k].abs().sum() > 0]
            assert not nonzero, f"missing seed keys are NOT zero-init (would damage the warm init): {nonzero[:5]}"
            if is_main_process() and missing:
                print(f"init_from: {len(missing)} zero-init tensors absent from seed "
                      f"(e.g. {missing[0]})", flush=True)
        else:
            module.load_state_dict(init_ckpt["model_state_dict"])
        if is_main_process():
            print(f"Warm-start (weights only) from {args.init_from} "
                  f"-> fresh Phase 2 fine-tune", flush=True)
    elif args.from_scratch:
        module = GaussianFormer(gf_config)
        if is_main_process():
            print(f"FROM SCRATCH: random init, latent_dim={args.latent_dim}", flush=True)
    else:
        module = transfer_weights(config.renderformer_model_id, gf_config)

    module.to(device)
    model_config = module.config
    ray_generator = RayGenerator().to(device)

    if is_main_process():
        config.save_dir.mkdir(parents=True, exist_ok=True)
    global_step = resume_ckpt["global_step"] if resume_ckpt else 0

    # --- Phase 1: encoder warmup, frozen backbone ---
    # DDP is constructed per phase, AFTER requires_grad is set. Freezing the backbone
    # before the wrap means DDP's reducer registers only the trainable encoder params,
    # so find_unused_parameters=False is correct (no Phase-1 crash, no corruption).
    if resume_phase != "phase2" and args.init_from is None and not args.from_scratch:
        trainable_names = freeze_backbone(module)
        if is_main_process():
            print(f"Phase 1 trainable params: {trainable_names}", flush=True)
        p1 = wrap_ddp(module, world_size, local_rank)
        optimizer = torch.optim.AdamW(
            [p for p in p1.parameters() if p.requires_grad],
            lr=config.phase1_lr,
            weight_decay=0.01,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.phase1_epochs,
            eta_min=config.phase1_lr * config.min_lr_ratio,
        )
        phase1_start = 0
        if resume_phase == "phase1":
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
            phase1_start = resume_epoch
        global_step = run_phase(
            "phase1", p1, ray_generator, dataloader, optimizer, scheduler,
            config, config.phase1_epochs, device, config.save_dir,
            config.log_interval, args.save_interval,
            args.log_loss_weight, args.lpips_loss_weight, args.fg_bg_weight,
            model_config, global_step, val_dataloader=val_dataloader,
            train_sampler=train_sampler, start_epoch=phase1_start, grad_accum=args.grad_accum,
            keep_last_n=args.keep_last_n,
        )
        del p1  # drop the Phase-1 DDP wrapper; its reducer hooks go inert once unused

    # --- Phase 2: full fine-tune ---
    unfreeze_all(module)
    p2 = wrap_ddp(module, world_size, local_rank)
    optimizer = torch.optim.AdamW(
        p2.parameters(),
        lr=config.phase2_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.phase2_epochs,
        eta_min=config.phase2_lr * config.min_lr_ratio,
    )
    phase2_start = 0
    if resume_phase == "phase2":
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        phase2_start = resume_epoch
    global_step = run_phase(
        "phase2", p2, ray_generator, dataloader, optimizer, scheduler,
        config, config.phase2_epochs, device, config.save_dir,
        config.log_interval, args.save_interval,
        args.log_loss_weight, args.lpips_loss_weight, args.fg_bg_weight,
        model_config, global_step, val_dataloader=val_dataloader,
        train_sampler=train_sampler, start_epoch=phase2_start, grad_accum=args.grad_accum,
        keep_last_n=args.keep_last_n,
    )

    # --- Save final model (rank 0 only) ---
    if is_main_process():
        final_path = config.save_dir / "gaussianformer_final"
        final_path.mkdir(parents=True, exist_ok=True)
        module.save_pretrained(final_path)
        print(f"\nFinal model saved to: {final_path}", flush=True)
        print(f"Use with: python infer_gaussian.py --model_id {final_path}", flush=True)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
