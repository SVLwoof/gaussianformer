"""Two objects in one scene: compose two pruned splats side by side and render model vs gsplat rasterizer.

Every training scene holds ONE object centred in [-0.45, 0.45]^3. Here each object is scaled by --scale
(means and Gaussian scales) and shifted to x = -/+ --offset, so the pair fits the same bounds; the model
sees 2 x 20k Gaussians (twice its training length). Outputs a comparison sheet (rasterizer | model | error
per view) and an orbit clip, both side by side.

  SCENES="387 1875" CKPT=checkpoints_probe_p2r_fg_r512p4_win_aug/phase2_epoch_3000.pt \\
  PYTHONPATH=. uv run --no-sync python data_v10/render_pair.py --out docs/report/pair
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from data_external.orbit import c2w_to_viewmat
from data_v10.prune_recovery import rasterize
from data_v10.render_video import FONT, FOV, band, cam_path
from gaussianformer.pipelines.rendering_pipeline import GaussianFormerRenderingPipeline
from gaussianformer.utils.checkpoint import load_checkpoint
from infer_gaussian import load_single_gaussian_h5_data


def compose(paths: list[Path], scale: float, offset: float, dev: str) -> torch.Tensor:
    """[N,14] = pos(3) scale(3) quat(4) color(3) opacity(1); object i scaled and moved to x = (2i-1) * offset."""
    parts = []
    for i, p in enumerate(paths):
        g = load_single_gaussian_h5_data(p)["gaussians"].to(dev).clone()
        g[:, :6] *= scale
        g[:, 0] += (2 * i - 1) * offset
        parts.append(g)
    return torch.cat(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--offset", type=float, default=0.24)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--radius", type=float, default=1.7)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--fps", type=int, default=24)
    a = ap.parse_args()
    scenes = [int(s) for s in os.environ["SCENES"].split()]
    dev = "cuda"
    a.out.mkdir(parents=True, exist_ok=True)
    tag = "_".join(f"{s:04d}" for s in scenes)

    g = compose([Path(f"data_v10/nsweep/n10_h5/scene_{s:04d}.h5") for s in scenes], a.scale, a.offset, dev)
    mask = torch.ones(1, g.size(0), dtype=torch.bool, device=dev)
    model, _ = load_checkpoint(Path(os.environ["CKPT"]))
    pipe = GaussianFormerRenderingPipeline(model.to(dev).eval())
    font = ImageFont.truetype(FONT, 22)
    focal = 0.5 * a.res / np.tan(0.5 * np.radians(FOV))
    K = torch.as_tensor(np.array([[focal, 0, a.res / 2], [0, focal, a.res / 2], [0, 0, 1]], np.float32), device=dev)
    rast_p = dict(means=g[:, :3].float(), quats=g[:, 6:10].float(), scales=g[:, 3:6].float(),
                  colors=g[:, 10:13].float(), opacities=g[:, 13].float())
    print(f"{tag}: {g.size(0)} Gaussians, scale {a.scale}, offset +-{a.offset}", flush=True)

    c2ws = cam_path("orbit", a.frames, a.radius)
    frames, rows = [], []
    for i in range(a.frames):
        c2w = torch.as_tensor(c2ws[i], device=dev)
        with torch.no_grad():
            out = pipe(gaussians=g[None], mask=mask, c2w=c2w[None][None], fov=torch.tensor([[FOV]], device=dev),
                       resolution=a.res, torch_dtype=torch.bfloat16)
        mdl = np.clip(out[0, 0].cpu().float().numpy(), 0, 1)
        vm = torch.as_tensor(c2w_to_viewmat(c2ws[i]), device=dev)[None]
        rast = rasterize(rast_p, vm, K[None], [0], a.res)[0].clamp(0, 1).cpu().numpy()
        psnr = 10 * np.log10(1 / max(np.mean((mdl - rast) ** 2), 1e-12))
        frames.append(np.concatenate([band(rast, f"rasterizer {i + 1}/{a.frames}", font),
                                      band(mdl, f"GaussianFormer  {psnr:.1f} dB", font)], axis=1))
        if i % (a.frames // 4) == 0:
            err = np.clip(np.abs(mdl - rast) * 4, 0, 1)
            rows.append(np.concatenate([frames[-1], band(err, "|model - rasterizer| x4", font)], axis=1))
            print(f"view {i}: model vs rasterizer {psnr:.2f} dB", flush=True)
    iio.imwrite(a.out / f"pair_{tag}_sheet.png", np.concatenate(rows, axis=0))
    iio.imwrite(a.out / f"pair_{tag}_orbit.mp4", np.stack(frames), fps=a.fps, quality=9)
    print(f"WROTE {a.out}/pair_{tag}_sheet.png + _orbit.mp4\nPAIR_DONE", flush=True)


if __name__ == "__main__":
    main()
