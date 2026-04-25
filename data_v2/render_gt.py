"""Render ground-truth images for all composed scenes using gsplat.

Reads H5 scene files from data_v2/h5s/, renders each view with gsplat,
saves as PNG to data_v2/renders/.

Output naming: scene_XXXX_view_Y.png (matches training/dataset.py expectations).

Usage:
    uv run python data_v2/render_gt.py
    uv run python data_v2/render_gt.py --resolution 512 --max_scenes 10
"""

import argparse
import sys
from pathlib import Path

import h5py
import imageio
import numpy as np
import torch


def c2w_to_viewmat(c2w: np.ndarray) -> np.ndarray:
    """Convert Blender-convention c2w to gsplat viewmat (pinhole w2c)."""
    w2c = np.linalg.inv(c2w)
    flip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    return (flip @ w2c).astype(np.float32)


def render_with_gsplat(
    means: np.ndarray,
    quats: np.ndarray,
    scales: np.ndarray,
    opacities: np.ndarray,
    colors: np.ndarray,
    viewmat: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    device: torch.device,
) -> np.ndarray:
    """Render Gaussians using gsplat."""
    import gsplat

    means_t = torch.tensor(means, dtype=torch.float32, device=device)
    quats_t = torch.tensor(quats, dtype=torch.float32, device=device)
    scales_t = torch.tensor(scales, dtype=torch.float32, device=device)
    opacities_t = torch.tensor(opacities, dtype=torch.float32, device=device)
    colors_t = torch.tensor(colors, dtype=torch.float32, device=device)

    viewmats = torch.tensor(viewmat, dtype=torch.float32, device=device).unsqueeze(0)
    Ks = torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        renders, alphas, meta = gsplat.rasterization(
            means=means_t,
            quats=quats_t,
            scales=scales_t,
            opacities=opacities_t,
            colors=colors_t,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=None,
            eps2d=0.3,
            render_mode="RGB",
            near_plane=0.01,
        )

    return renders[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Render ground-truth with gsplat")
    parser.add_argument("--h5_dir", type=Path, default=Path("data_v2/h5s"))
    parser.add_argument("--output_dir", type=Path, default=Path("data_v2/renders"))
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max_scenes", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: gsplat requires CUDA. Running on CPU will fail.")
        sys.exit(1)
    print(f"Using device: {device}")

    h5_files = sorted(args.h5_dir.glob("*.h5"))
    if args.max_scenes:
        h5_files = h5_files[:args.max_scenes]

    print(f"Rendering {len(h5_files)} scenes at {args.resolution}px with gsplat...\n")

    total_views = 0
    for idx, h5_path in enumerate(h5_files):
        scene_name = h5_path.stem

        with h5py.File(h5_path, "r") as f:
            means = np.array(f["means"], dtype=np.float32)
            scales = np.array(f["scales"], dtype=np.float32)
            rotations = np.array(f["rotations"], dtype=np.float32)
            colors = np.array(f["colors"], dtype=np.float32)
            opacities = np.array(f["opacities"], dtype=np.float32).ravel()
            c2w_all = np.array(f["c2w"], dtype=np.float32)
            fov_all = np.array(f["fov"], dtype=np.float32)

        n_views = c2w_all.shape[0]

        for v in range(n_views):
            c2w = c2w_all[v]
            fov_deg = float(fov_all[v])
            fov_rad = fov_deg * np.pi / 180.0
            focal = 0.5 * args.resolution / np.tan(0.5 * fov_rad)
            K = np.array([
                [focal, 0.0, args.resolution / 2.0],
                [0.0, focal, args.resolution / 2.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)

            viewmat = c2w_to_viewmat(c2w)
            img = render_with_gsplat(
                means, rotations, scales, opacities, colors,
                viewmat, K, args.resolution, args.resolution, device,
            )
            img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            out_path = args.output_dir / f"{scene_name}_view_{v}.png"
            imageio.v3.imwrite(out_path, img_u8)
            total_views += 1

        if (idx + 1) % 50 == 0 or idx < 3 or idx == len(h5_files) - 1:
            print(f"  [{idx + 1}/{len(h5_files)}] {scene_name} ({n_views} views, "
                  f"{len(means)} Gaussians)")

    print(f"\nDone. Rendered {total_views} views across {len(h5_files)} scenes.")
    print(f"Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
