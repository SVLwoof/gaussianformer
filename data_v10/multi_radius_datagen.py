"""Camera-distance augmentation data: GT renders of the FULL splat on extra orbits.

Two uses:
  train set   --src_h5_dir nsweep/n10_h5 --src_renders_dir nsweep/n10_renders --radii 1.15 2.45
              -> out H5 = the same pruned Gaussians with 14 + 14*len(radii) cameras, out renders =
                 the base views copied (0..13) + the new orbits (14..) rendered from the full splat.
  eval set    --eval_radius 1.15 -> only renders (view 0..13) of that orbit, for
              `ceiling_eval --radius 1.15 --renders_dir <out_renders_dir>`.

  PYTHONPATH=. uv run --no-sync python data_v10/multi_radius_datagen.py \\
      --full_h5_dir data_v10/rebuild_n10/full_h5s --scenes_file data_v10/nsweep/n10_scenes.json \\
      --src_h5_dir data_v10/nsweep/n10_h5 --src_renders_dir data_v10/nsweep/n10_renders \\
      --out_h5_dir data_v10/nsweep/n10_r3_h5 --out_renders_dir data_v10/nsweep/n10_r3_renders --radii 1.15 2.45
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import gsplat
import h5py
import imageio.v3 as iio
import numpy as np
import torch

from data_external.orbit import make_orbit_views, orbit_c2w

FOV, RES, N_VIEWS = 45.0, 512, 14


def render_full(h5: Path, radius: float, device) -> list[np.ndarray]:
    with h5py.File(h5, "r") as f:
        t = {k: torch.from_numpy(np.array(f[k], np.float32)).to(device) for k in ("means", "scales", "rotations", "colors", "opacities")}
    vm_np, Ks_np = make_orbit_views(N_VIEWS, radius, FOV, RES, up_axis="y")
    vm, Ks = torch.from_numpy(vm_np).to(device), torch.from_numpy(Ks_np).to(device)
    imgs = []
    for i in range(N_VIEWS):
        with torch.no_grad():
            img, _, _ = gsplat.rasterization(
                means=t["means"], quats=t["rotations"], scales=t["scales"], opacities=t["opacities"].reshape(-1),
                colors=t["colors"], viewmats=vm[i:i + 1], Ks=Ks[i:i + 1], width=RES, height=RES,
                sh_degree=None, eps2d=0.3, render_mode="RGB", near_plane=0.01, packed=True)
        imgs.append((np.clip(img[0].cpu().numpy(), 0, 1) * 255).astype(np.uint8))
    return imgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_h5_dir", type=Path, required=True)
    ap.add_argument("--scenes_file", type=Path, required=True)
    ap.add_argument("--out_renders_dir", type=Path, required=True)
    ap.add_argument("--radii", type=float, nargs="*", default=[])
    ap.add_argument("--src_h5_dir", type=Path, default=None)
    ap.add_argument("--src_renders_dir", type=Path, default=None)
    ap.add_argument("--out_h5_dir", type=Path, default=None)
    ap.add_argument("--eval_radius", type=float, default=None, help="renders-only mode at this radius, views 0..13")
    a = ap.parse_args()
    device = torch.device("cuda")
    scenes = [int(s) for s in json.loads(a.scenes_file.read_text())]
    a.out_renders_dir.mkdir(parents=True, exist_ok=True)
    if a.eval_radius is not None:
        for n, s in enumerate(scenes):
            if (a.out_renders_dir / f"scene_{s:04d}_view_{N_VIEWS - 1}.png").exists():
                continue
            for v, img in enumerate(render_full(a.full_h5_dir / f"scene_{s:04d}.h5", a.eval_radius, device)):
                iio.imwrite(a.out_renders_dir / f"scene_{s:04d}_view_{v}.png", img)
            if n % 25 == 0:
                print(f"{n}/{len(scenes)} scene_{s:04d}", flush=True)
        print("DATAGEN_DONE eval", flush=True)
        return
    assert a.src_h5_dir and a.src_renders_dir and a.out_h5_dir and a.radii
    a.out_h5_dir.mkdir(parents=True, exist_ok=True)
    for s in scenes:
        name = f"scene_{s:04d}"
        src = a.src_h5_dir / f"{name}.h5"
        with h5py.File(src, "r") as f:
            base = {k: np.array(f[k]) for k in f}
        assert base["c2w"].shape[0] == N_VIEWS
        c2w = [base["c2w"]]
        fov = [base["fov"]]
        for v in range(N_VIEWS):
            shutil.copyfile(a.src_renders_dir / f"{name}_view_{v}.png", a.out_renders_dir / f"{name}_view_{v}.png")
        for r in a.radii:
            off = len(c2w) * N_VIEWS
            for v, img in enumerate(render_full(a.full_h5_dir / f"{name}.h5", r, device)):
                iio.imwrite(a.out_renders_dir / f"{name}_view_{off + v}.png", img)
            c2w.append(orbit_c2w(N_VIEWS, r))
            fov.append(np.full(N_VIEWS, FOV, np.float32))
        with h5py.File(a.out_h5_dir / f"{name}.h5", "w") as f:
            for k in ("means", "scales", "rotations", "colors", "opacities"):
                f.create_dataset(k, data=base[k])
            f.create_dataset("c2w", data=np.concatenate(c2w).astype(np.float32))
            f.create_dataset("fov", data=np.concatenate(fov).astype(np.float32))
        print(f"{name}: {len(c2w) * N_VIEWS} views", flush=True)
    print("DATAGEN_DONE train", flush=True)


if __name__ == "__main__":
    main()
