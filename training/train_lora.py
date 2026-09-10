"""Per-object LoRA adapter training on a frozen GaussianFormer base.

Deliberately separate from training/train.py: one phase, no weight transfer, no generic
val set (its loss measures forgetting of other objects, not quality on this one -- the
object's own held-out verdict is the metric, see data_external/codec_scaleout_eval.py).
Reuses train.py's DDP setup, forward and loss so the recipe matches the full fine-tune.

Checkpoints hold only the adapter (A/B) + the base checkpoint path, so a run dir is MBs.
Resume is automatic from the newest lora_epoch_*.pt in --save_dir (killable requeues).

  torchrun --standalone --nproc_per_node=4 -m training.train_lora \
      --gaussian_h5_dir ... --renders_dir ... --init_from checkpoints_v18_256/phase2_epoch_30.pt \
      --save_dir checkpoints_lora_gopro_r4 --rank 4 --lr 2e-4 --grad_accum 2
"""
from __future__ import annotations

import argparse
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler

from gaussianformer.layers.lora import DEFAULT_TARGETS, apply_lora, load_lora, lora_state_dict
from gaussianformer.models.config import GaussianFormerConfig
from gaussianformer.models.gaussianformer import GaussianFormer
from gaussianformer.utils.ray_generator import RayGenerator
from training.dataset import GaussianRenderDataset, collate_fn
from training.train import (
    compute_loss, is_main_process, setup_ddp, training_forward, unwrap_model, wrap_ddp,
)

GRAD_CLIP = 1.0  # same as TrainingConfig.grad_clip


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA adapter training for one object")
    p.add_argument("--gaussian_h5_dir", type=Path, required=True)
    p.add_argument("--renders_dir", type=Path, required=True)
    p.add_argument("--save_dir", type=Path, required=True)
    p.add_argument("--init_from", type=Path, required=True, help="frozen base checkpoint (.pt)")
    p.add_argument("--pe_type", type=str, default="rope")
    p.add_argument("--model_cfg", nargs="*", default=None, metavar="KEY=VAL",
                   help="GaussianFormerConfig overrides of the BASE (e.g. proj_rope_2d=true); "
                        "recorded in the adapter checkpoint so evals rebuild the same base")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--epochs", type=int, default=27)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min_lr_ratio", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--alpha", type=float, default=None, help="default 2*rank")
    p.add_argument("--targets", type=str, default=DEFAULT_TARGETS,
                   help="regex over qualified nn.Linear names to adapt")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--log_loss_weight", type=float, default=0.5)
    p.add_argument("--lpips_loss_weight", type=float, default=0.5)
    p.add_argument("--fg_bg_weight", type=float, default=1.0)
    p.add_argument("--augment_rotation", action="store_true")
    p.add_argument("--num_workers", type=int, default=3)
    p.add_argument("--save_interval", type=int, default=3)
    p.add_argument("--keep_last_n", type=int, default=2)
    p.add_argument("--log_interval", type=int, default=50)
    return p.parse_args()


def newest_checkpoint(save_dir: Path) -> Path | None:
    ckpts = sorted(save_dir.glob("lora_epoch_*.pt"), key=lambda q: int(q.stem.rsplit("_", 1)[1]))
    return ckpts[-1] if ckpts else None


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, device = setup_ddp()
    alpha = args.alpha if args.alpha is not None else 2.0 * args.rank
    meta = {"rank": args.rank, "alpha": alpha, "targets": args.targets, "model_cfg": args.model_cfg or [],
            "dropout": args.dropout, "base_ckpt": str(args.init_from)}

    # --- Data (object views only) ---
    dataset = GaussianRenderDataset(args.gaussian_h5_dir, args.renders_dir, args.resolution,
                                    augment_rotation=args.augment_rotation)
    sampler = (DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
               if world_size > 1 else None)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                            shuffle=sampler is None, num_workers=args.num_workers,
                            persistent_workers=args.num_workers > 0,
                            pin_memory=device.type == "cuda", collate_fn=collate_fn)

    # --- Frozen base + adapter ---
    module = GaussianFormer(GaussianFormerConfig(pe_type=args.pe_type).with_overrides(args.model_cfg))
    base = torch.load(args.init_from, map_location="cpu", weights_only=True)
    own = module.state_dict()
    # RoPE frequency tables are config constants: drop the seed's copies where shapes differ.
    base_sd = {k: v for k, v in base["model_state_dict"].items()
               if not (k.endswith(".freqs") and (k not in own or own[k].shape != v.shape))}
    missing, unexpected = module.load_state_dict(base_sd, strict=False)
    assert not unexpected, unexpected
    assert all(k.endswith(".freqs") for k in missing), f"seed lacks non-RoPE tensors: {missing[:5]}"
    del base
    resume = newest_checkpoint(args.save_dir)
    start_epoch, global_step = 0, 0
    if resume is not None:
        # Early adapter checkpoints stored Path objects in "args"; allowlist them (our own file).
        with torch.serialization.safe_globals([Path, type(Path())]):
            ckpt = torch.load(resume, map_location="cpu", weights_only=True)
        assert ckpt["lora"] == meta, f"resume meta mismatch: {ckpt['lora']} vs {meta}"
        load_lora(module, ckpt, merge=False)
        start_epoch, global_step = ckpt["epoch"], ckpt["global_step"]
    else:
        ckpt = None
        apply_lora(module, args.rank, alpha, args.targets, args.dropout)
    module.to(device)
    model_config = module.config
    ray_generator = RayGenerator().to(device)

    n_adapter = sum(v.numel() for v in lora_state_dict(module).values())
    n_total = sum(p.numel() for p in module.parameters())
    if is_main_process():
        args.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"LoRA r={args.rank} alpha={alpha} dropout={args.dropout} targets={args.targets!r}: "
              f"{n_adapter:,} / {n_total:,} params ({n_adapter*4/1e3:.0f} KB fp32)", flush=True)
        print(f"{len(dataset)} samples, world {world_size} x bs {args.batch_size} x accum "
              f"{args.grad_accum} = effective batch {world_size*args.batch_size*args.grad_accum}, "
              f"lr {args.lr:.1e}, {args.epochs} epochs", flush=True)
        if resume is not None:
            print(f"RESUME from {resume} (epoch {start_epoch})", flush=True)

    model = wrap_ddp(module, world_size, local_rank)  # registers only A/B (base is frozen)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * args.min_lr_ratio)
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        del ckpt

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        dataset.set_epoch(epoch)
        model.train()
        sums = torch.zeros(3, device=device)
        steps = 0
        t0 = time.time()
        optimizer.zero_grad()
        for micro, batch in enumerate(dataloader):
            b = {k: batch[k].to(device, non_blocking=True)
                 for k in ("gaussians", "mask", "c2w", "fov", "target")}
            last_micro = (micro + 1) % args.grad_accum == 0
            sync = (model.no_sync() if (args.grad_accum > 1 and not last_micro
                                        and hasattr(model, "no_sync")) else nullcontext())
            with sync:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = training_forward(model, ray_generator, b["gaussians"], b["mask"],
                                            b["c2w"], b["fov"], args.resolution, model_config)
                    loss, log_term, lpips_term = compute_loss(
                        pred, b["target"], args.log_loss_weight, args.lpips_loss_weight, device,
                        fg_bg_weight=args.fg_bg_weight)
                (loss / args.grad_accum).backward()
            with torch.no_grad():
                sums += torch.stack([loss.detach(), log_term.detach(), lpips_term.detach()])
            steps += 1
            if not last_micro:
                continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step % args.log_interval == 0 and is_main_process():
                print(f"  step {global_step}, total: {loss.item():.6f}, log: {log_term.item():.6f}, "
                      f"lpips: {lpips_term.item():.6f}", flush=True)

        scheduler.step()
        if dist.is_initialized():
            dist.all_reduce(sums, op=dist.ReduceOp.AVG)
        avg_loss, avg_log, avg_lpips = (sums / max(steps, 1)).tolist()
        if is_main_process():
            print(f"[lora] Epoch {epoch + 1}/{args.epochs}, avg total: {avg_loss:.6f}, "
                  f"log: {avg_log:.6f}, lpips: {avg_lpips:.6f}, "
                  f"lr: {scheduler.get_last_lr()[0]:.2e}, time: {time.time() - t0:.1f}s", flush=True)

        last = (epoch + 1) == args.epochs
        if ((epoch + 1) % args.save_interval == 0 or last) and is_main_process():
            path = args.save_dir / f"lora_epoch_{epoch + 1}.pt"
            tmp = path.with_suffix(".pt.tmp")
            torch.save({"lora": meta, "lora_state_dict": lora_state_dict(unwrap_model(model)),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch + 1, "global_step": global_step, "loss": avg_loss,
                        "args": {k: str(v) if isinstance(v, Path) else v
                                 for k, v in vars(args).items()}}, tmp)
            os.replace(tmp, path)
            print(f"  Saved checkpoint: {path}", flush=True)
            if args.keep_last_n > 0:
                old = sorted(args.save_dir.glob("lora_epoch_*.pt"),
                             key=lambda q: int(q.stem.rsplit("_", 1)[1]))[:-args.keep_last_n]
                for q in old:
                    q.unlink()

    if is_main_process():
        final = args.save_dir / "lora_final.pt"
        torch.save({"lora": meta, "lora_state_dict": lora_state_dict(unwrap_model(model))}, final)
        print(f"\nFinal adapter saved to: {final} ({final.stat().st_size/1e3:.0f} KB)", flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
