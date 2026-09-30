"""Side-by-side clips of the rasterized input (left) and GaussianFormer (right), one forward pass per frame.

  orbit   camera orbit at --radius
  dolly   camera moves from distance 2.45 to 1.15 and back
  move    the object translates in front of a fixed camera
  tumble  the object rotates about a tilted axis in front of a fixed camera

  python -m tools.render_video --h5 data/val/h5s/scene_0031.h5 --out videos/ --clips orbit tumble
"""
import argparse
from pathlib import Path

import gsplat
import imageio.v3 as iio
import numpy as np
import roma
import torch
from PIL import Image, ImageDraw

from gaussianformer.pipelines.rendering_pipeline import GaussianFormerRenderingPipeline
from gaussianformer.utils.cameras import FOV, look_at, to_gsplat
from gaussianformer.utils.transform import quaternion_multiply
from training.dataset import load_gaussians


def camera(clip: str, t: float, radius: float) -> np.ndarray:
    if clip == "dolly":
        r = 1.8 + 0.65 * np.cos(2 * np.pi * t)
        theta = 2 * np.pi * t / 6
    else:
        r, theta = radius, 2 * np.pi * t if clip == "orbit" else 0.0
    eye = np.array([r * np.cos(theta), 0.25 * r, r * np.sin(theta)], np.float32)
    return look_at(eye, np.zeros(3, np.float32), np.array([0, 1, 0], np.float32))


def move_object(g: torch.Tensor, clip: str, t: float) -> torch.Tensor:
    g = g.clone()
    if clip == "move":
        g[:, :3] += torch.tensor([0.35 * np.sin(2 * np.pi * t), 0.12 * np.sin(4 * np.pi * t), 0.0], device=g.device)
    elif clip == "tumble":
        axis = torch.tensor([0.3, 1.0, 0.15], device=g.device)
        R = roma.rotvec_to_rotmat(axis / axis.norm() * 2 * np.pi * t)
        g[:, :3] = g[:, :3] @ R.T
        g[:, 6:10] = quaternion_multiply(roma.rotmat_to_unitquat(R)[[3, 0, 1, 2]].expand(len(g), 4), g[:, 6:10])
    return g


def label(img: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray((img * 255).astype(np.uint8))
    ImageDraw.Draw(im).text((8, 6), text, fill=(255, 230, 0), font_size=22)
    return np.asarray(im)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="shahafvl/gaussianformer")
    ap.add_argument("--clips", nargs="+", default=["orbit", "dolly", "move", "tumble"])
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--radius", type=float, default=1.7)
    a = ap.parse_args()
    device = torch.device("cuda")
    pipe = GaussianFormerRenderingPipeline.from_pretrained(a.model).to(device)
    g0 = torch.from_numpy(load_gaussians(a.h5)[0]).to(device)
    mask = torch.ones(1, len(g0), dtype=torch.bool, device=device)
    a.out.mkdir(parents=True, exist_ok=True)
    for clip in a.clips:
        frames = []
        for i in range(a.frames):
            t = i / a.frames
            g, c2w = move_object(g0, clip, t), camera(clip, t, a.radius)
            model = pipe(g[None], mask, torch.from_numpy(c2w)[None, None].to(device), torch.tensor([[FOV]], device=device))
            viewmat, K = (torch.from_numpy(x).to(device) for x in to_gsplat(c2w[None]))
            raster, _, _ = gsplat.rasterization(
                means=g[:, :3], quats=g[:, 6:10], scales=g[:, 3:6], opacities=g[:, 13], colors=g[:, 10:13],
                viewmats=viewmat, Ks=K, width=512, height=512, sh_degree=None, eps2d=0.3, render_mode="RGB",
                near_plane=0.01, packed=True)
            frames.append(np.concatenate([label(raster[0].clamp(0, 1).cpu().numpy(), "rasterizer"),
                                          label(model[0, 0].float().clamp(0, 1).cpu().numpy(), "GaussianFormer")], 1))
        path = a.out / f"{a.h5.stem}_{clip}.mp4"
        iio.imwrite(path, np.stack(frames), fps=24, quality=9)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
