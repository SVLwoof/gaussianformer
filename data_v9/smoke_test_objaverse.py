"""V9 smoke test: download Objaverse_Splats chunk 000-000, extract 10
high-quality objects, render orientation views to verify the dataset is
suitable for V9 training (clean isolated objects, consistent orientation,
recognizable, colorful).

For each object:
  - Load PLY from extracted zip
  - Normalize (center on median, scale to fit +-0.45)
  - Print stats (N, AABB, scales, opacity, color distribution)
  - Render 4 views: top-down, front-side, isometric-high, isometric-low
  - Save a per-object overview (4 views in a row)

Then build a master 10-object grid for at-a-glance inspection.

Outputs:
  data_v9/smoke_test/objects/{uid}/point_cloud.ply  (extracted)
  data_v9/smoke_test/renders/{uid}_overview.png     (4 views per object)
  data_v9/smoke_test/master_grid.png                (all 10 in one image)
  data_v9/smoke_test/stats.txt                       (printed stats per object)
"""
from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import gsplat
import imageio
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from plyfile import PlyData

from data_external.orbit import look_at_blender, c2w_to_viewmat


SMOKE_DIR = Path("data_v9/smoke_test")
OBJECTS_DIR = SMOKE_DIR / "objects"
RENDERS_DIR = SMOKE_DIR / "renders"
N_OBJECTS = 10
RESOLUTION = 256
FOV_DEG = 45.0
RADIUS = 1.7

# Quality filter for picking top objects from this chunk.
MIN_PSNR = 32.0
MAX_LPIPS = 0.06


def k_matrix(resolution: int, fov_deg: float) -> np.ndarray:
    fov_rad = fov_deg * np.pi / 180.0
    focal = 0.5 * resolution / np.tan(0.5 * fov_rad)
    return np.array(
        [[focal, 0, resolution / 2.0], [0, focal, resolution / 2.0], [0, 0, 1]],
        dtype=np.float32,
    )


def load_normalized_ply(ply_bytes: bytes) -> dict:
    """Load PLY, decode 3DGS fields, normalize (center on median + scale to +-0.45)."""
    plydata = PlyData.read(io.BytesIO(ply_bytes))
    v = plydata["vertex"]
    n = v.count
    means = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    scales_log = np.stack([v[f"scale_{i}"] for i in range(3)], axis=-1).astype(np.float32)
    quats = np.stack([v[f"rot_{i}"] for i in range(4)], axis=-1).astype(np.float32)
    quats = quats / (np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-9)
    f_dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], axis=-1).astype(np.float32)
    C0 = 0.28209479177387814
    colors = np.clip(0.5 + C0 * f_dc, 0, 1).astype(np.float32)
    opacity_logit = np.asarray(v["opacity"], dtype=np.float32)
    opacities = (1.0 / (1.0 + np.exp(-opacity_logit))).astype(np.float32)

    median = np.median(means, axis=0)
    means_centered = means - median
    aabb_max = np.abs(means_centered).max()
    norm_scale = 0.45 / max(aabb_max, 1e-6)
    means_n = means_centered * norm_scale
    scales_n = np.exp(scales_log) * norm_scale

    return {
        "n_raw": n,
        "median": median,
        "aabb_max_raw": aabb_max,
        "norm_scale": norm_scale,
        "means": means_n,
        "scales": scales_n,
        "quats": quats,
        "colors": colors,
        "opacities": opacities,
    }


def render_axis_views(arr: dict, device: torch.device, label: str) -> np.ndarray:
    """Render 4 diagnostic views: top, side-X, isometric-high, isometric-low."""
    means = torch.from_numpy(arr["means"]).to(device)
    scales = torch.from_numpy(arr["scales"]).to(device)
    quats = torch.from_numpy(arr["quats"]).to(device)
    opacities = torch.from_numpy(arr["opacities"]).to(device)
    colors = torch.from_numpy(arr["colors"]).to(device)
    K = k_matrix(RESOLUTION, FOV_DEG)
    Ks_t = torch.from_numpy(K[None, :, :]).to(device)

    # Try +Y up convention (standard for Objaverse).
    views = [
        ("top",       np.array([0, +RADIUS, 0.001], dtype=np.float32),
                      np.array([0, 0, 1], dtype=np.float32)),
        ("front",     np.array([+RADIUS, 0.0, 0.0], dtype=np.float32),
                      np.array([0, 1, 0], dtype=np.float32)),
        ("iso_high",  np.array([RADIUS * 0.7, RADIUS * 0.7, RADIUS * 0.7], dtype=np.float32),
                      np.array([0, 1, 0], dtype=np.float32)),
        ("iso_low",   np.array([RADIUS * 0.7, -RADIUS * 0.3, RADIUS * 0.7], dtype=np.float32),
                      np.array([0, 1, 0], dtype=np.float32)),
    ]
    rendered = []
    for vname, eye, up in views:
        c2w = look_at_blender(eye, np.zeros(3, dtype=np.float32), up=up)
        viewmat = c2w_to_viewmat(c2w)
        viewmat_t = torch.from_numpy(viewmat[None, :, :]).to(device)
        with torch.no_grad():
            img, _, _ = gsplat.rasterization(
                means=means, quats=quats, scales=scales,
                opacities=opacities, colors=colors,
                viewmats=viewmat_t, Ks=Ks_t,
                width=RESOLUTION, height=RESOLUTION,
                sh_degree=None, eps2d=0.3, render_mode="RGB",
                near_plane=0.01, packed=True,
            )
        rendered.append(img[0].clamp(0, 1).cpu().numpy())
    gutter = np.ones((RESOLUTION, 4, 3), dtype=np.float32)
    row = []
    for i, img in enumerate(rendered):
        if i > 0:
            row.append(gutter)
        row.append(img)
    return np.clip(np.concatenate(row, axis=1), 0, 1)


def main():
    OBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    # 1. Load metadata, filter to chunk 000-000 with quality threshold.
    print("Loading metadata CSV from HF cache...")
    csv_path = hf_hub_download("ShapeSplats/Objaverse_Splats",
                                 "completed_3dgs_metadata.csv", repo_type="dataset")
    chunk = "000-000"
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if f"glbs/{chunk}/" in r["path"]:
                try:
                    psnr = float(r["psnr"])
                    lpips = float(r["lpips"])
                except ValueError:
                    continue
                if psnr >= MIN_PSNR and lpips <= MAX_LPIPS and int(r["num_GS"]) == 50000:
                    rows.append((psnr, lpips, r["object_name"], r.get("caption", "")))
    rows.sort(key=lambda x: (-x[0], x[1]))  # high PSNR first, low LPIPS tiebreaker
    chosen = rows[:N_OBJECTS]
    print(f"Found {len(rows)} high-quality objects in chunk {chunk} "
          f"(PSNR>={MIN_PSNR}, LPIPS<={MAX_LPIPS}); selecting top {N_OBJECTS}")
    for psnr, lpips, uid, _ in chosen:
        print(f"  {uid}  psnr={psnr:.2f}  lpips={lpips:.4f}")

    # 2. Download the chunk zip (cached on second run).
    print(f"\nDownloading {chunk}.zip...")
    zip_path = hf_hub_download("ShapeSplats/Objaverse_Splats",
                                 f"{chunk}.zip", repo_type="dataset")
    print(f"  cached at {zip_path}")

    # 3. Extract chosen PLYs from the zip.
    print(f"\nExtracting {N_OBJECTS} PLYs from zip...")
    extracted = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        for psnr, lpips, uid, caption in chosen:
            ply_member = f"{chunk}/{uid}/ckpts/point_cloud_15000.ply"
            if ply_member not in names:
                print(f"  MISSING: {ply_member}")
                continue
            ply_bytes = zf.read(ply_member)
            out_dir = OBJECTS_DIR / uid
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "point_cloud.ply").write_bytes(ply_bytes)
            extracted.append((uid, psnr, lpips, caption, ply_bytes))
            print(f"  {uid}  ({len(ply_bytes)/1e6:.1f} MB)")

    # 4. Per-object: normalize, stats, render axis views.
    stats_lines = []
    overview_rows = []
    print("\nRendering 4 axis-aligned views per object (+Y up)...")
    for uid, psnr, lpips, caption, ply_bytes in extracted:
        arr = load_normalized_ply(ply_bytes)
        stats = (
            f"\n=== {uid}  psnr={psnr:.2f}  lpips={lpips:.4f}\n"
            f"  caption: {caption[:200] if caption else '(none)'}\n"
            f"  N={arr['n_raw']:,}  median={arr['median']}  aabb_max_raw={arr['aabb_max_raw']:.3f}\n"
            f"  norm_scale={arr['norm_scale']:.3f}\n"
            f"  scales linear: min={arr['scales'].min():.5f}  med={np.median(arr['scales']):.5f}  "
            f"p95={np.percentile(arr['scales'], 95):.5f}  max={arr['scales'].max():.5f}\n"
            f"  opacities: min={arr['opacities'].min():.3f}  med={np.median(arr['opacities']):.3f}  "
            f"max={arr['opacities'].max():.3f}\n"
            f"  colors RGB mean: ({arr['colors'][:,0].mean():.3f}, "
            f"{arr['colors'][:,1].mean():.3f}, {arr['colors'][:,2].mean():.3f})"
        )
        stats_lines.append(stats)
        print(stats)
        row = render_axis_views(arr, device, uid)
        out_path = RENDERS_DIR / f"{uid}_overview.png"
        imageio.v3.imwrite(out_path, (row * 255).astype(np.uint8))
        overview_rows.append(row)

    # 5. Master grid: all 10 objects stacked.
    H, W, _ = overview_rows[0].shape
    gutter = np.ones((8, W, 3), dtype=np.float32)
    parts = []
    for i, r in enumerate(overview_rows):
        if i > 0:
            parts.append(gutter)
        parts.append(r)
    master = np.concatenate(parts, axis=0)
    master_path = SMOKE_DIR / "master_grid.png"
    imageio.v3.imwrite(master_path, (np.clip(master, 0, 1) * 255).astype(np.uint8))
    print(f"\nMaster grid -> {master_path} ({master.shape[1]}x{master.shape[0]})")

    # 6. Stats file.
    (SMOKE_DIR / "stats.txt").write_text("\n".join(stats_lines))
    print(f"Stats -> {SMOKE_DIR / 'stats.txt'}")


if __name__ == "__main__":
    main()
