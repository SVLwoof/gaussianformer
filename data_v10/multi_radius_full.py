"""Full-data camera-distance augmentation: extra orbits at r 1.15 / 2.45 for every h5s_20k_rec object.

Same idea as multi_radius_datagen.py (GT from the FULL splat, inputs = the pruned 20k splat), at 27k-object
scale, streamed so nothing large lands on the lab share:
  - the full splat is decoded straight from the Objaverse_Splats chunk zip (node-local HF cache, deleted per
    chunk) with process_full's normalisation -- never written as a full H5;
  - the base views are hardlinks to data_v10/renders, only the new views are real PNGs;
  - the out H5 holds external links to the h5s_20k_rec Gaussians plus the concatenated cameras (~KB).
Views 0..13 = base orbit r 1.7, 14.. = N_NEW views per extra radius. Resume-safe per object (H5 written last).
The first object of every chunk re-renders base view 0 and must match data_v10/renders (frame check).

  PYTHONPATH=. uv run --no-sync python data_v10/multi_radius_full.py --task 0 --n_tasks 11
"""
from __future__ import annotations

import argparse
import json
import os
import zipfile
from collections import defaultdict
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from data_external.orbit import make_orbit_views, orbit_c2w
from data_v10.process_full import REPO_ID, load_normalized_rot, render_full

ROOT = Path(__file__).resolve().parent
SRC_H5, SRC_RENDERS = ROOT / "h5s_20k_rec", ROOT / "renders"
OUT_H5, OUT_RENDERS = ROOT / "h5s_20k_rec_r3", ROOT / "renders_r3"
RADII, N_NEW, N_BASE, FOV, RES = (1.15, 2.45), 7, 14, 45.0, 512


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(10 * np.log10(1.0 / max(np.mean((a.astype(np.float64) - b) ** 2), 1e-12)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--n_tasks", type=int, required=True)
    a = ap.parse_args()
    device = torch.device("cuda")
    OUT_H5.mkdir(exist_ok=True)
    OUT_RENDERS.mkdir(exist_ok=True)

    by_chunk = defaultdict(list)
    for o in json.loads((ROOT / "object_list_train.json").read_text()):
        if (SRC_H5 / f"scene_{o['scene_idx']:04d}.h5").exists():  # low-opacity objects were never kept
            by_chunk[o["chunk"]].append(o)
    chunks = sorted(by_chunk)[a.task::a.n_tasks]
    views = [make_orbit_views(N_NEW, r, FOV, RES, up_axis="y") for r in RADII]
    views = [(torch.from_numpy(vm).to(device), torch.from_numpy(K).to(device)) for vm, K in views]
    base_vm, base_K = (torch.from_numpy(x).to(device) for x in make_orbit_views(N_BASE, 1.7, FOV, RES, up_axis="y"))
    new_c2w = np.concatenate([orbit_c2w(N_NEW, r) for r in RADII]).astype(np.float32)

    for chunk in chunks:
        todo = [o for o in by_chunk[chunk] if not (OUT_H5 / f"scene_{o['scene_idx']:04d}.h5").exists()]
        print(f"=== chunk {chunk}: {len(todo)}/{len(by_chunk[chunk])} to do", flush=True)
        if not todo:
            continue
        zip_path = hf_hub_download(REPO_ID, f"{chunk}.zip", repo_type="dataset")
        with zipfile.ZipFile(zip_path) as zf:
            for n, o in enumerate(todo):
                name = f"scene_{o['scene_idx']:04d}"
                arr = load_normalized_rot(zf.read(f"{chunk}/{o['uid']}/ckpts/point_cloud_15000.ply"))
                if n == 0:  # same frame as the existing GT?
                    ref = iio.imread(SRC_RENDERS / f"{name}_view_0.png")
                    img = render_full(arr, base_vm[:1], base_K[:1], RES, device)[0]
                    p = psnr((img * 255).round() / 255, ref / 255)
                    assert p > 40, f"{name}: base view 0 re-render {p:.1f} dB vs data_v10/renders -- frame mismatch"
                    print(f"  frame check {name}: {p:.1f} dB", flush=True)
                v = N_BASE
                for vm, K in views:
                    for img in render_full(arr, vm, K, RES, device):
                        iio.imwrite(OUT_RENDERS / f"{name}_view_{v}.png", (img * 255).astype(np.uint8), compress_level=9)
                        v += 1
                for b in range(N_BASE):  # hardlinks: a symlink costs a 32 KB block on this share, a hardlink nothing
                    link = OUT_RENDERS / f"{name}_view_{b}.png"
                    if not link.exists():
                        os.link(SRC_RENDERS / f"{name}_view_{b}.png", link)
                src = SRC_H5 / f"{name}.h5"
                with h5py.File(src, "r") as f:
                    c2w, fov = np.array(f["c2w"], np.float32), np.array(f["fov"], np.float32)
                    keys = [k for k in ("means", "scales", "rotations", "colors", "opacities") if k in f]
                assert c2w.shape[0] == N_BASE
                tmp = OUT_H5 / f"{name}.h5.tmp"
                with h5py.File(tmp, "w") as f:
                    for k in keys:
                        f[k] = h5py.ExternalLink(str(src), k)
                    f.create_dataset("c2w", data=np.concatenate([c2w, new_c2w]))
                    f.create_dataset("fov", data=np.concatenate([fov, np.full(len(new_c2w), FOV, np.float32)]))
                os.replace(tmp, OUT_H5 / f"{name}.h5")
                if n % 100 == 0:
                    print(f"  {name} ({n + 1}/{len(todo)})", flush=True)
        blob = Path(zip_path).resolve()  # hf_hub_download returns a symlink into the blob store
        Path(zip_path).unlink(missing_ok=True)
        blob.unlink(missing_ok=True)
    print(f"DONE_MULTI_RADIUS task {a.task}", flush=True)


if __name__ == "__main__":
    main()
