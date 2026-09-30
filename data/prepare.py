"""Build the dataset from Objaverse_Splats: for every object, 28 ground-truth views of the full splat and a
20k-Gaussian input.

Per object: decode the 3DGS PLY, centre it, turn it Y-up and scale it into [-0.45, 0.45]^3; render the
28 views of cameras.all_views() from the full splat; keep the 20k Gaussians with the highest LightGaussian
importance and fine-tune them (L1 + SSIM, no densification) to match the full splat from 64 views
(gaussianformer/splat.py). Objects whose maximum opacity is below 0.4 are skipped. Resumable; shard with
--task / --n_tasks.

  python -m data.prepare --objects data/splits/val.json --out data/val
"""
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

from gaussianformer.splat import normalize, prune, rasterize, read_ply
from gaussianformer.utils.cameras import FOV, all_views, to_gsplat

REPO = "ShapeSplats/Objaverse_Splats"


def process(ply: bytes, name: str, out: Path, views, c2w: np.ndarray, device) -> bool:
    arr = normalize(read_ply(ply), up="z")
    if arr is None or arr["opacities"].max() < 0.4:
        return False
    full = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in arr.items()}
    with torch.no_grad():
        for i in range(len(views[0])):
            img = rasterize(full, views[0][i:i + 1], views[1][i:i + 1])[0][0].clamp(0, 1)
            iio.imwrite(out / "renders" / f"{name}_view_{i}.png", (img.cpu().numpy() * 255).astype(np.uint8))
    g = {k: v.cpu().numpy() for k, v in prune(full).items()}
    tmp = out / "h5s" / f"{name}.h5.tmp"
    with h5py.File(tmp, "w") as f:
        for k in ("means", "scales", "rotations", "colors"):
            f.create_dataset(k, data=g[k])
        f.create_dataset("opacities", data=g["opacities"][:, None])
        f.create_dataset("c2w", data=c2w)
        f.create_dataset("fov", data=np.full(len(c2w), FOV, np.float32))
    os.replace(tmp, out / "h5s" / f"{name}.h5")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objects", type=Path, required=True, help="split file: [{scene, chunk, uid}, ...]")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--n_tasks", type=int, default=1)
    a = ap.parse_args()
    device = torch.device("cuda")
    (a.out / "h5s").mkdir(parents=True, exist_ok=True)
    (a.out / "renders").mkdir(parents=True, exist_ok=True)

    c2w = all_views()
    views = tuple(torch.from_numpy(x).to(device) for x in to_gsplat(c2w))
    by_chunk = defaultdict(list)
    for o in json.loads(a.objects.read_text()):
        by_chunk[o["chunk"]].append(o)
    for chunk in sorted(by_chunk)[a.task::a.n_tasks]:
        todo = [o for o in by_chunk[chunk] if not (a.out / "h5s" / f"scene_{o['scene']:04d}.h5").exists()]
        if not todo:
            continue
        zip_path = hf_hub_download(REPO, f"{chunk}.zip", repo_type="dataset")
        kept = 0
        with zipfile.ZipFile(zip_path) as zf:
            for o in todo:
                ply = zf.read(f"{chunk}/{o['uid']}/ckpts/point_cloud_15000.ply")
                kept += process(ply, f"scene_{o['scene']:04d}", a.out, views, c2w, device)
                torch.cuda.empty_cache()
        print(f"chunk {chunk}: {kept} of {len(todo)} objects kept", flush=True)


if __name__ == "__main__":
    main()
