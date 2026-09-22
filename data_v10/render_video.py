"""Animation clips from a trained GaussianFormer: model vs gsplat rasterizer, side by side.

RenderFormer renders animations as a sequence of independent frames (batch_infer.py --save_video,
imageio fps=24) -- there is no temporal model. Same here: every frame is one forward pass, so the
clips also show how temporally stable the renderer is on its own.

Clips (--clips, default all):
  orbit   smooth 360 deg camera orbit at --radius, constant elevation (the training regime)
  dolly   camera pulls from radius 2.45 in to 1.15 and back while drifting 60 deg
  move    the OBJECT translates (camera fixed) -- never seen in training, the splat leaves the origin
  tumble  the OBJECT rotates about a tilted axis (camera fixed): means rotated, quaternions composed

  SCENE=scene_0007 CKPT=checkpoints_codec_so_..._l2/phase2_epoch_27.pt \\
  PYTHONPATH=. uv run --no-sync python data_v10/render_video.py --out docs/report/video
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
import roma
import torch
from PIL import Image, ImageDraw, ImageFont

from data_external.orbit import c2w_to_viewmat, look_at_blender
from data_v10.prune_recovery import rasterize
from gaussianformer.pipelines.rendering_pipeline import GaussianFormerRenderingPipeline
from gaussianformer.utils.checkpoint import load_checkpoint
from gaussianformer.utils.quaternion import quaternion_multiply
from infer_gaussian import load_single_gaussian_h5_data

FOV, FONT = 45.0, "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def cam_path(name: str, n: int, radius: float) -> np.ndarray:
    """c2w [n,4,4] for the camera clips; object clips keep the camera at theta=0."""
    out = []
    for i in range(n):
        t = i / n
        if name == "dolly":
            r = 1.8 + 0.65 * np.cos(2 * np.pi * t)  # 2.45 -> 1.15 -> 2.45
            theta, elev = 2 * np.pi * t / 6, 0.25 * r
        elif name == "orbit":
            r, theta, elev = radius, 2 * np.pi * t, 0.25 * radius
        else:  # object clips: the camera never moves, so all motion on screen is the object's
            r, theta, elev = radius, 0.0, 0.25 * radius
        eye = np.array([r * np.cos(theta), elev, r * np.sin(theta)], np.float32)
        out.append(look_at_blender(eye, np.zeros(3, np.float32), up=np.array([0, 1, 0], np.float32)))
    return np.stack(out)


def transform(g: torch.Tensor, name: str, t: float) -> torch.Tensor:
    """Object motion in world space: g is [N,14] = pos(3) scale(3) quat(wxyz)(4) color(3) opacity(1)."""
    if name == "move":
        off = torch.tensor([0.35 * np.sin(2 * np.pi * t), 0.12 * np.sin(4 * np.pi * t), 0.0],
                           device=g.device, dtype=g.dtype)
        out = g.clone()
        out[:, :3] = g[:, :3] + off
        return out
    if name == "tumble":
        axis = torch.tensor([0.3, 1.0, 0.15], device=g.device, dtype=torch.float32)
        axis = axis / axis.norm()
        R = roma.rotvec_to_rotmat(axis * (2 * np.pi * t))
        q = roma.rotmat_to_unitquat(R)[[3, 0, 1, 2]].to(g.dtype)  # roma is xyzw, ours is wxyz
        out = g.clone()
        out[:, :3] = g[:, :3] @ R.to(g.dtype).T
        out[:, 6:10] = quaternion_multiply(q[None].expand(g.size(0), 4), g[:, 6:10])
        return out
    return g


def band(img: np.ndarray, text: str, font) -> np.ndarray:
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 30], fill=(0, 0, 0))
    d.text((6, 4), text, fill=(255, 230, 0), font=font)
    return np.asarray(im)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--clips", nargs="+", default=["orbit", "dolly", "move", "tumble"])
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--radius", type=float, default=1.7)
    ap.add_argument("--fps", type=int, default=24)
    a = ap.parse_args()
    scene, ckpt = os.environ["SCENE"], Path(os.environ["CKPT"])
    dev = "cuda"
    a.out.mkdir(parents=True, exist_ok=True)

    h5 = Path(f"experiments/overfit/data/codec_scaleout/{scene}/h5s/{scene}.h5")
    data = load_single_gaussian_h5_data(h5)
    g0, mask = data["gaussians"].to(dev), data["mask"][None].to(dev)
    with h5py.File(h5) as f:
        rec = {k: torch.as_tensor(np.array(f[k], np.float32), device=dev)
               for k in ("means", "scales", "rotations", "colors", "opacities")}
    model, _ = load_checkpoint(ckpt)
    pipe = GaussianFormerRenderingPipeline(model.to(dev).eval())
    font = ImageFont.truetype(FONT, 22)
    focal = 0.5 * a.res / np.tan(0.5 * np.radians(FOV))
    K = torch.as_tensor(np.array([[focal, 0, a.res / 2], [0, focal, a.res / 2], [0, 0, 1]], np.float32), device=dev)

    for clip in a.clips:
        c2ws = cam_path(clip, a.frames, a.radius)
        frames = []
        for i in range(a.frames):
            t = i / a.frames
            g = transform(g0, clip, t)
            c2w = torch.as_tensor(c2ws[i], device=dev)
            with torch.no_grad():
                out = pipe(gaussians=g[None], mask=mask, c2w=c2w[None][None],
                           fov=torch.tensor([[FOV]], device=dev), resolution=a.res, torch_dtype=torch.bfloat16)
            model_img = np.clip(out[0, 0].cpu().float().numpy(), 0, 1)
            # rasterizer reference on the same transformed splat
            p = dict(means=g[:, :3].float(), quats=g[:, 6:10].float(), scales=rec["scales"],
                     colors=g[:, 10:13].float(), opacities=rec["opacities"].reshape(-1))
            vm = torch.as_tensor(c2w_to_viewmat(c2ws[i]), device=dev)[None]
            rast = rasterize(p, vm, K[None], [0], a.res)[0].clamp(0, 1).cpu().numpy()
            frames.append(np.concatenate([band(rast, f"rasterizer  {clip} {i + 1}/{a.frames}", font),
                                          band(model_img, "GaussianFormer", font)], axis=1))
            if i % 20 == 0:
                print(f"{clip} {i}/{a.frames}", flush=True)
        path = a.out / f"{scene}_{clip}.mp4"
        iio.imwrite(path, np.stack(frames), fps=a.fps, quality=9)
        print(f"WROTE {path} ({len(frames)} frames)", flush=True)
    print("VIDEO_DONE", flush=True)


if __name__ == "__main__":
    main()
