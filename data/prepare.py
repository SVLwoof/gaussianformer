"""Build the dataset from Objaverse_Splats: for every object, 28 ground-truth views of the full splat and a
20k-Gaussian input.

Per object: decode the 3DGS PLY, centre it, turn it Y-up and scale it into [-0.45, 0.45]^3; render the
28 views of cameras.all_views() from the full splat; keep the 20k Gaussians with the highest LightGaussian
importance and fine-tune them (L1 + SSIM, no densification) to match the full splat from 64 views.
Objects whose maximum opacity is below 0.4 are skipped. Resumable; shard with --task / --n_tasks.

  python -m data.prepare --objects data/splits/val.json --out data/val
"""
import argparse
import io
import json
import os
import zipfile
from collections import defaultdict
from pathlib import Path

import gsplat
import h5py
import imageio.v3 as iio
import numpy as np
import roma
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from plyfile import PlyData

from data.cameras import FOV, RESOLUTION, all_views, orbit, to_gsplat

REPO = "ShapeSplats/Objaverse_Splats"
SH_C0 = 0.28209479177387814
Z_UP_TO_Y_UP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
KEEP = 20_000
RECOVERY_VIEWS = 64
RECOVERY_ITERS = 1500


def load_splat(ply: bytes) -> dict[str, np.ndarray] | None:
    """PLY bytes -> normalised Gaussians (means, scales, rotations wxyz, colors, opacities), None if degenerate."""
    v = PlyData.read(io.BytesIO(ply))["vertex"]
    means = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float32)
    scales = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], -1)).astype(np.float32)
    quats = np.stack([v[f"rot_{i}"] for i in range(4)], -1).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-9
    colors = np.clip(0.5 + SH_C0 * np.stack([v[f"f_dc_{i}"] for i in range(3)], -1), 0, 1).astype(np.float32)
    opacities = (1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], np.float32)))).astype(np.float32)

    means = ((means - np.median(means, axis=0)) @ Z_UP_TO_Y_UP.T).astype(np.float32)
    rot = roma.unitquat_to_rotmat(torch.from_numpy(quats)[:, [1, 2, 3, 0]])
    quats = roma.rotmat_to_unitquat(torch.from_numpy(Z_UP_TO_Y_UP) @ rot)[:, [3, 0, 1, 2]].numpy()
    extent = float(np.abs(means).max())
    if extent < 1e-4:
        return None
    s = 0.45 / extent
    return dict(means=means * s, scales=scales * s, rotations=quats, colors=colors, opacities=opacities)


def rasterize(g: dict[str, torch.Tensor], viewmats: torch.Tensor, Ks: torch.Tensor) -> torch.Tensor:
    img, _, info = gsplat.rasterization(
        means=g["means"], quats=g["rotations"] / g["rotations"].norm(dim=-1, keepdim=True), scales=g["scales"],
        opacities=g["opacities"], colors=g["colors"], viewmats=viewmats, Ks=Ks, width=RESOLUTION,
        height=RESOLUTION, sh_degree=None, eps2d=0.3, render_mode="RGB", near_plane=0.01, packed=True)
    return img, info


def importance_topk(g: dict[str, torch.Tensor], viewmats, Ks, keep: int) -> torch.Tensor:
    """LightGaussian importance: opacity x projected area, summed over views, x max scale^0.1."""
    score = torch.zeros(len(g["means"]), device=g["means"].device, dtype=torch.float64)
    for v in range(len(viewmats)):
        with torch.no_grad():
            _, info = rasterize(g, viewmats[v:v + 1], Ks[v:v + 1])
        area = info["radii"][:, 0].double() * info["radii"][:, 1].double()
        score.scatter_add_(0, info["gaussian_ids"], info["opacities"].double() * area)
    score = score * g["scales"].max(1).values.double() ** 0.1
    return torch.topk(score, keep).indices


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    g = torch.exp(-(torch.arange(11, dtype=torch.float32, device=x.device) - 5) ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :]).expand(3, 1, 11, 11).contiguous()
    conv = lambda t: F.conv2d(t, w, padding=5, groups=3)
    mx, my = conv(x), conv(y)
    sx, sy, sxy = conv(x * x) - mx * mx, conv(y * y) - my * my, conv(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sx + sy + c2))).mean()


def recover(g: dict[str, torch.Tensor], target: torch.Tensor, viewmats, Ks) -> dict[str, np.ndarray]:
    """Fine-tune the kept Gaussians against the full splat's renders `target` [V, H, W, 3]."""
    p = {"means": g["means"], "log_scales": g["scales"].clamp_min(1e-8).log(), "rotations": g["rotations"],
         "opacity_logits": torch.logit(g["opacities"].clamp(1e-4, 1 - 1e-4)),
         "color_logits": torch.logit(g["colors"].clamp(1e-4, 1 - 1e-4))}
    p = {k: torch.nn.Parameter(v.clone()) for k, v in p.items()}
    lrs = {"means": 1.6e-4, "log_scales": 5e-3, "rotations": 1e-3, "opacity_logits": 5e-2, "color_logits": 2.5e-3}
    opt = torch.optim.Adam([{"params": [p[k]], "lr": lr} for k, lr in lrs.items()])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95 ** (1 / 400))
    current = lambda: dict(means=p["means"], scales=p["log_scales"].exp(), rotations=p["rotations"],
                           opacities=p["opacity_logits"].sigmoid(), colors=p["color_logits"].sigmoid())
    for _ in range(RECOVERY_ITERS):
        v = torch.randint(0, len(viewmats), (4,), device=target.device)
        img = rasterize(current(), viewmats[v], Ks[v])[0].clamp(0, 1)
        loss = 0.8 * (img - target[v]).abs().mean() + 0.2 * (1 - ssim(img.permute(0, 3, 1, 2), target[v].permute(0, 3, 1, 2)))
        loss.backward()
        opt.step()
        opt.zero_grad()
        sched.step()
    with torch.no_grad():
        out = current()
        out["rotations"] = out["rotations"] / out["rotations"].norm(dim=-1, keepdim=True)
        return {k: v.cpu().numpy() for k, v in out.items()}


def process(ply: bytes, name: str, out: Path, views, rec_views, c2w: np.ndarray, device) -> bool:
    arr = load_splat(ply)
    if arr is None or arr["opacities"].max() < 0.4:
        return False
    full = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in arr.items()}
    with torch.no_grad():
        for i in range(len(views[0])):
            img = rasterize(full, views[0][i:i + 1], views[1][i:i + 1])[0][0].clamp(0, 1)
            iio.imwrite(out / "renders" / f"{name}_view_{i}.png", (img.cpu().numpy() * 255).astype(np.uint8))
        target = torch.cat([rasterize(full, rec_views[0][i:i + 1], rec_views[1][i:i + 1])[0].clamp(0, 1)
                            for i in range(RECOVERY_VIEWS)])
    keep = importance_topk(full, *rec_views, KEEP)
    g = recover({k: v[keep] for k, v in full.items()}, target, *rec_views)
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
    rec_views = tuple(torch.from_numpy(x).to(device) for x in to_gsplat(orbit(RECOVERY_VIEWS, 1.7)))
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
                kept += process(ply, f"scene_{o['scene']:04d}", a.out, views, rec_views, c2w, device)
                torch.cuda.empty_cache()
        print(f"chunk {chunk}: {kept} of {len(todo)} objects kept", flush=True)


if __name__ == "__main__":
    main()
