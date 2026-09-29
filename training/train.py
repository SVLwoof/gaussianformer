"""GaussianFormer training with a warmup-stable-decay learning rate.

The learning rate warms up over --warmup_steps, stays at --lr, then decays with a cosine to 1% of it over
the last --decay_steps. One epoch is one random view of every object. Checkpoints are step-numbered; a
run resumes from the newest one in --save_dir. `touch <save_dir>/STOP` checkpoints and exits.

  torchrun --nproc_per_node=8 -m training.train --data data/train --val_data data/val \\
      --init renderformer --save_dir checkpoints/head --resolution 256 --steps 33500 ...

scripts/train.sh runs the four stages of the released model.
"""
import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import lpips
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from gaussianformer.models.config import GaussianFormerConfig
from gaussianformer.pipelines.rendering_pipeline import model_forward
from gaussianformer.models.gaussianformer import GaussianFormer
from gaussianformer.utils.checkpoint import load_checkpoint, load_seed
from training.dataset import GaussianRenderDataset, collate_fn
from training.weight_transfer import from_renderformer

WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
MIN_LR_RATIO = 0.01


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="directory with h5s/ and renders/")
    p.add_argument("--val_data", type=Path, required=True)
    p.add_argument("--val_objects", type=Path, default=None, help="JSON list of validation objects to use")
    p.add_argument("--save_dir", type=Path, required=True)
    p.add_argument("--init", required=True,
                   help="'renderformer' (pretrained RenderFormer backbone), a checkpoint or a Hugging Face model (weights only)")
    p.add_argument("--resume_from", type=Path, default=None,
                   help="continue this checkpoint's run (weights, optimizer and step) in a new --save_dir")
    p.add_argument("--model_cfg", nargs="*", default=None, metavar="KEY=VALUE",
                   help="GaussianFormerConfig overrides, e.g. proj_rope_2d=true patch_size=4")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--decay_steps", type=int, default=0, help="cosine decay over the last N steps (0 = none)")
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lpips_weight", type=float, default=0.5, help="LPIPS weight; the log-L1 term gets 1 - this")
    p.add_argument("--bg_weight", type=float, default=0.05, help="weight of background pixels in the L1 term")
    p.add_argument("--head_only", action="store_true", help="train only the Gaussian input head")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--keep_last", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=8)
    return p.parse_args()


def lr_at(step: int, a: argparse.Namespace) -> float:
    if step < a.warmup_steps:
        return a.lr * (step + 1) / a.warmup_steps
    start = a.steps - a.decay_steps
    if step < start:
        return a.lr
    t = min((step - start) / max(a.decay_steps, 1), 1.0)
    floor = a.lr * MIN_LR_RATIO
    return floor + (a.lr - floor) * 0.5 * (1 + math.cos(math.pi * t))


class Loss(torch.nn.Module):
    """(1 - w) * log-space L1 + w * LPIPS-VGG, predictions in log10(HDR + 1), targets in [0, 1]."""

    def __init__(self, lpips_weight: float, bg_weight: float):
        super().__init__()
        self.w, self.bg_weight = lpips_weight, bg_weight
        self.lpips = lpips.LPIPS(net="vgg").eval().requires_grad_(False) if lpips_weight > 0 else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = pred.squeeze(1)
        log_target = torch.log10(target + 1.0)
        fg = (target.amax(dim=-1, keepdim=True) > 0.02).float()  # objects cover a few % of the image
        weight = (fg + (1.0 - fg) * self.bg_weight).expand_as(log_target)
        l1 = (weight * (pred - log_target).abs()).sum() / weight.sum()
        perceptual = torch.zeros((), device=pred.device)
        if self.lpips is not None:
            with torch.autocast(device_type="cuda", enabled=False):  # LPIPS gradients are too noisy in bf16
                p = torch.clamp(10.0 ** pred.float() - 1.0, 0.0, 1.0).permute(0, 3, 1, 2) * 2 - 1
                g = target.float().permute(0, 3, 1, 2) * 2 - 1
                perceptual = self.lpips(p, g).mean()
        return (1 - self.w) * l1 + self.w * perceptual, l1, perceptual


def setup_ddp() -> tuple[int, int, torch.device]:
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, torch.device("cuda")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank, dist.get_world_size(), torch.device(f"cuda:{local_rank}")


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def step_of(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def load_model(a: argparse.Namespace):
    """Model, the checkpoint to continue (or None), and the names of the trainable parameters."""
    own = sorted(a.save_dir.glob("step_*.pt"), key=step_of)
    if own or a.resume_from:
        model, ckpt = load_checkpoint(own[-1] if own else a.resume_from)
        return model, ckpt, ckpt.get("trainable")
    if a.init == "renderformer":
        model, head = from_renderformer(GaussianFormerConfig().with_overrides(a.model_cfg))
        return model, None, head if a.head_only else None
    if Path(a.init).is_file():
        model, _ = load_checkpoint(Path(a.init), overrides=a.model_cfg, allow_new=True)
    else:  # Hugging Face id or directory
        base = GaussianFormer.from_pretrained(a.init)
        model = GaussianFormer(base.config.with_overrides(a.model_cfg))
        load_seed(model, base.state_dict())
    head = ["gaussian_encoder.weight", "gaussian_encoder.bias", "gaussian_encoder_norm.weight", "gaussian_token"]
    return model, None, head if a.head_only else None


def main() -> None:
    a = parse_args()
    local_rank, world_size, device = setup_ddp()
    data = GaussianRenderDataset(a.data / "h5s", a.data / "renders", a.resolution, augment_rotation=True,
                                 views_per_epoch=1)
    sampler = DistributedSampler(data) if world_size > 1 else None
    loader = DataLoader(data, batch_size=1, sampler=sampler, shuffle=sampler is None, num_workers=a.num_workers,
                        pin_memory=True, collate_fn=collate_fn)
    val = GaussianRenderDataset(a.val_data / "h5s", a.val_data / "renders", a.resolution,
                                objects=json.loads(a.val_objects.read_text()) if a.val_objects else None)
    val_loader = DataLoader(val, batch_size=1, sampler=DistributedSampler(val, shuffle=False) if world_size > 1 else None,
                            num_workers=a.num_workers, pin_memory=True, collate_fn=collate_fn)

    module, ckpt, trainable = load_model(a)
    if trainable is not None:
        for name, param in module.named_parameters():
            param.requires_grad = name in trainable
    module.to(device)
    model = DDP(module, device_ids=[local_rank]) if world_size > 1 else module
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr,
                                  weight_decay=WEIGHT_DECAY)
    step, epoch = 0, 0
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step, epoch = ckpt["step"], ckpt["epoch"]
    del ckpt
    loss_fn = Loss(a.lpips_weight, a.bg_weight).to(device)
    if is_main():
        a.save_dir.mkdir(parents=True, exist_ok=True)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"step {step} of {a.steps}; {n / 1e6:.1f}M trainable parameters; {len(loader)} steps per epoch "
              f"on {world_size} GPUs", flush=True)

    def batch_loss(batch):
        b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        pred = model_forward(model, b["gaussians"], b["mask"], b["c2w"][:, None], b["fov"][:, None], a.resolution,
                             tf32_view=True)
        return loss_fn(pred, b["target"])

    def save(at: int) -> None:
        path = a.save_dir / f"step_{at}.pt"
        torch.save({"step": at, "epoch": epoch, "config": asdict(module.config), "trainable": trainable,
                    "model_state_dict": module.state_dict(), "optimizer_state_dict": optimizer.state_dict()},
                   path.with_suffix(".tmp"))
        os.replace(path.with_suffix(".tmp"), path)
        for old in sorted(a.save_dir.glob("step_*.pt"), key=step_of)[:-a.keep_last]:
            old.unlink()

    def validate() -> list[float]:
        model.eval()
        sums, n = torch.zeros(3, device=device), 0
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for batch in val_loader:
                sums += torch.stack(batch_loss(batch))
                n += 1
        if dist.is_initialized():
            dist.all_reduce(sums, op=dist.ReduceOp.AVG)
        model.train()
        return (sums / max(n, 1)).tolist()

    model.train()
    stop = False
    while step < a.steps and not stop:
        if sampler is not None:
            sampler.set_epoch(epoch)
        data.set_epoch(epoch)
        t0 = time.time()
        for batch in loader:
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, a)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, l1, perceptual = batch_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if step % 100 == 0 and is_main():
                print(f"step {step}: loss {loss.item():.5f} (L1 {l1.item():.5f}, LPIPS {perceptual.item():.5f}), "
                      f"lr {lr_at(step - 1, a):.2e}, {(time.time() - t0) / 100:.2f} s/step", flush=True)
                t0 = time.time()
            if step % a.save_every == 0 or step == a.steps:
                v = validate()
                if is_main():
                    print(f"step {step}: validation loss {v[0]:.5f} (L1 {v[1]:.5f}, LPIPS {v[2]:.5f})", flush=True)
                    save(step)
            if step % 20 == 0:
                flag = torch.tensor(float((a.save_dir / "STOP").exists()), device=device)
                if dist.is_initialized():
                    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                if flag.item():
                    stop = True
                    if is_main():
                        save(step)
                        (a.save_dir / "STOP").unlink()
                        print(f"stopped at step {step}; rerun the same command to continue", flush=True)
                    break
            if step >= a.steps:
                break
        epoch += 1
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
