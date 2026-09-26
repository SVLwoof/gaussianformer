"""Full-data fine-tune with a warmup-stable-decay learning rate (2026-09-23).

Separate from training/train.py (whose phase-2 cosine is tied to the epoch budget): here the LR is a
function of the optimizer step only --

    warmup   0 -> lr over --warmup_steps
    stable   lr, until --steps - --decay_steps
    decay    cosine lr -> lr * --min_lr_ratio over the last --decay_steps

so a run can be stopped or extended at any stable checkpoint (the LR there is the peak, a valid
continuation point), and a finished model for evaluation is a short DECAY BRANCH off any stable
checkpoint: a new --save_dir with --resume_from <stable ckpt> and --steps = its step + --decay_steps.

One epoch = --views_per_object random views of every object (redrawn per epoch), all views of an object in one step. Checkpoints are
step-numbered; the newest in --save_dir is resumed automatically (preemption / time limit), else
--resume_from (optimizer + step carried over), else --init_from (weights only, step 0).
Freeze on demand: `touch <save_dir>/FREEZE` -> the run checkpoints within 20 steps and exits cleanly; resubmitting
the same command continues from that step. Otherwise a kill loses at most --save_every steps.

  torchrun --standalone --nproc_per_node=8 -m training.train_full \
      --gaussian_h5_dir data_v10/h5s_20k_rec_r3 --renders_dir data_v10/renders_r3 \
      --val_h5_dir data_v10/nsweep/val100_h5 --val_renders_dir data_v10/nsweep/val100_renders \
      --init_from <stage-R ckpt> --save_dir checkpoints_full_wsd --steps 375000 --decay_steps 0 \
      --model_cfg proj_rope_2d=true patch_size=4 ...
"""
from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from gaussianformer.utils.checkpoint import load_checkpoint
from gaussianformer.utils.ray_generator import RayGenerator
from training.dataset import GaussianRenderDataset, collate_fn
from training.train import compute_loss, is_main_process, setup_ddp, training_forward, unwrap_model, wrap_ddp

GRAD_CLIP = 1.0  # same as TrainingConfig.grad_clip


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-data fine-tune, warmup-stable-decay LR")
    p.add_argument("--gaussian_h5_dir", type=Path, required=True)
    p.add_argument("--renders_dir", type=Path, required=True)
    p.add_argument("--val_h5_dir", type=Path, required=True)
    p.add_argument("--val_renders_dir", type=Path, required=True)
    p.add_argument("--save_dir", type=Path, required=True)
    p.add_argument("--init_from", type=Path, default=None, help="weights-only warm start (step 0)")
    p.add_argument("--resume_from", type=Path, default=None,
                   help="continue from this checkpoint's weights, optimizer and step (decay branches)")
    p.add_argument("--model_cfg", nargs="*", default=None, metavar="KEY=VAL")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--views_per_object", type=int, default=1,
                   help="views of one object per step, sharing one scene encoding (the scene encoder is ~half a step)")
    p.add_argument("--steps", type=int, required=True, help="total optimizer steps (the end of the decay)")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--decay_steps", type=int, default=0, help="0 = stable to the end (extendable run)")
    p.add_argument("--min_lr_ratio", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--log_loss_weight", type=float, default=0.5)
    p.add_argument("--lpips_loss_weight", type=float, default=0.5)
    p.add_argument("--fg_bg_weight", type=float, default=0.05)
    p.add_argument("--save_every", type=int, default=2000, help="steps between rolling checkpoints (+ val loss)")
    p.add_argument("--keep_last_n", type=int, default=2)
    p.add_argument("--milestone_every", type=int, default=0, help="steps between checkpoints never pruned (0 = none)")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--log_interval", type=int, default=100)
    return p.parse_args()


def lr_at(step: int, a: argparse.Namespace) -> float:
    if step < a.warmup_steps:
        return a.lr * (step + 1) / a.warmup_steps
    start = a.steps - a.decay_steps
    if step < start:
        return a.lr
    frac = min((step - start) / max(a.decay_steps, 1), 1.0)
    floor = a.lr * a.min_lr_ratio
    return floor + (a.lr - floor) * 0.5 * (1 + math.cos(math.pi * frac))


def step_of(q: Path) -> int:
    return int(q.stem.rsplit("_", 1)[1])


def main() -> None:
    a = parse_args()
    assert (a.init_from is None) != (a.resume_from is None), "exactly one of --init_from / --resume_from"
    rank, local_rank, world_size, device = setup_ddp()

    dataset = GaussianRenderDataset(a.gaussian_h5_dir, a.renders_dir, a.resolution,
                                    augment_rotation=True, views_per_epoch=a.views_per_object,
                                    views_per_item=a.views_per_object)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, shuffle=sampler is None,
                        num_workers=a.num_workers, persistent_workers=False,  # fresh view draw per epoch
                        pin_memory=True, collate_fn=collate_fn)
    val = GaussianRenderDataset(a.val_h5_dir, a.val_renders_dir, a.resolution)
    val_sampler = DistributedSampler(val, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    val_loader = DataLoader(val, batch_size=1, sampler=val_sampler, num_workers=a.num_workers,
                            pin_memory=True, collate_fn=collate_fn)

    own = sorted(a.save_dir.glob("full_step_*.pt"), key=step_of)
    src = own[-1] if own else (a.resume_from or a.init_from)
    carry = bool(own) or a.resume_from is not None  # optimizer + step come along
    module, ckpt = load_checkpoint(src, overrides=a.model_cfg, allow_new=not carry)
    module.to(device)
    model_config = module.config
    model = wrap_ddp(module, world_size, local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    step, epoch = 0, 0
    if carry:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step, epoch = ckpt["global_step"], ckpt["epoch"]
    del ckpt
    ray_generator = RayGenerator().to(device)
    if is_main_process():
        a.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"{'RESUME' if carry else 'INIT'} from {src} at step {step}; epoch = {len(dataset)} objects x {a.views_per_object} views "
              f"/ {world_size} GPUs = {len(loader)} steps; lr {a.lr:.1e}, warmup {a.warmup_steps}, "
              f"decay {a.decay_steps} of {a.steps} steps", flush=True)

    def save(tag_step: int, avg: float) -> None:
        path = a.save_dir / f"full_step_{tag_step}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save({"phase": "full", "epoch": epoch + 1, "global_step": tag_step,
                    "config": asdict(unwrap_model(model).config),
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "loss": avg}, tmp)
        os.replace(tmp, path)
        print(f"  Saved checkpoint: {path} (lr {lr_at(tag_step, a):.2e})", flush=True)
        if a.milestone_every and tag_step % a.milestone_every == 0 and not (a.save_dir / f"milestone_step_{tag_step}.pt").exists():
            os.link(path, a.save_dir / f"milestone_step_{tag_step}.pt")
        for q in sorted(a.save_dir.glob("full_step_*.pt"), key=step_of)[:-a.keep_last_n]:
            q.unlink()

    def validate() -> list[float]:
        model.eval()
        sums, n = torch.zeros(3, device=device), 0
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for batch in val_loader:
                b = {k: batch[k].to(device, non_blocking=True) for k in ("gaussians", "mask", "c2w", "fov", "target")}
                pred = training_forward(model, ray_generator, b["gaussians"], b["mask"], b["c2w"], b["fov"],
                                        a.resolution, model_config)
                sums += torch.stack(compute_loss(pred, b["target"], a.log_loss_weight, a.lpips_loss_weight,
                                                 device, fg_bg_weight=a.fg_bg_weight))
                n += 1
        if dist.is_initialized():
            dist.all_reduce(sums, op=dist.ReduceOp.AVG)
        model.train()
        return (sums / max(n, 1)).tolist()

    frozen = False
    while step < a.steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        dataset.set_epoch(epoch)
        model.train()
        sums, n, t0 = torch.zeros(3, device=device), 0, time.time()
        for batch in loader:
            lr = lr_at(step, a)
            for g in optimizer.param_groups:
                g["lr"] = lr
            b = {k: batch[k].to(device, non_blocking=True) for k in ("gaussians", "mask", "c2w", "fov", "target")}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = training_forward(model, ray_generator, b["gaussians"], b["mask"], b["c2w"], b["fov"],
                                        a.resolution, model_config)
                pred = pred.reshape(-1, 1, *pred.shape[2:])  # [bs*V, 1, H, W, 3]
                loss, log_term, lpips_term = compute_loss(pred, b["target"].reshape(-1, *pred.shape[2:]), a.log_loss_weight,
                                                          a.lpips_loss_weight, device, fg_bg_weight=a.fg_bg_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            with torch.no_grad():
                sums += torch.stack([loss.detach(), log_term.detach(), lpips_term.detach()])
            n += 1
            if step % a.log_interval == 0 and is_main_process():
                print(f"  step {step}, total: {loss.item():.6f}, log: {log_term.item():.6f}, "
                      f"lpips: {lpips_term.item():.6f}, lr: {lr:.2e}", flush=True)
            if step % a.save_every == 0 or step == a.steps:
                v = validate()
                if dist.is_initialized():
                    dist.all_reduce(sums, op=dist.ReduceOp.AVG)
                avg = (sums / max(n, 1)).tolist()
                if is_main_process():
                    print(f"[full] step {step}/{a.steps} epoch {epoch}, train total: {avg[0]:.6f}, "
                          f"log: {avg[1]:.6f}, lpips: {avg[2]:.6f} | val total: {v[0]:.6f}, log: {v[1]:.6f}, "
                          f"lpips: {v[2]:.6f} | {(time.time() - t0) / n:.2f} s/step", flush=True)
                    save(step, avg[0])
                sums, n, t0 = torch.zeros(3, device=device), 0, time.time()
            if step % 20 == 0:  # freeze: `touch <save_dir>/FREEZE` -> checkpoint within ~20 steps and exit
                flag = torch.tensor(float((a.save_dir / "FREEZE").exists()), device=device)
                if dist.is_initialized():
                    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                if flag.item():
                    frozen = True
                    if is_main_process():
                        save(step, float("nan"))
                        (a.save_dir / "FREEZE").unlink()
                        print(f"FROZEN at step {step}: resubmit the same command to continue", flush=True)
                    break
            if step >= a.steps:
                break
        if frozen:
            break
        epoch += 1

    if is_main_process() and not frozen:
        print(f"DONE_FULL step {step}", flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
