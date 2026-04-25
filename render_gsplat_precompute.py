"""Pre-render all Gaussian training scenes with gsplat (GPU reference renderer).

Run this once while the GPU is free. The resulting PNGs are used by
render_triplet.py for comparison — no GPU needed at triplet assembly time.

Output naming matches training_renders/: scene_XXXX_view_Y.png

Usage:
    uv run python render_gsplat_precompute.py
    uv run python render_gsplat_precompute.py --h5_dir gaussian_training_h5s --resolution 512
"""

import argparse
from pathlib import Path

import imageio
import numpy as np
import torch

from render_gsplat import load_gaussian_h5
from validate_with_gsplat import c2w_to_viewmat, render_with_gsplat


def main():
    parser = argparse.ArgumentParser(description="Pre-render all scenes with gsplat")
    parser.add_argument("--h5_dir", type=Path, default=Path("gaussian_training_h5s"))
    parser.add_argument("--output_dir", type=Path, default=Path("gsplat_renders"))
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max_scenes", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    h5_files = sorted(args.h5_dir.glob("*.h5"))
    if args.max_scenes:
        h5_files = h5_files[:args.max_scenes]

    print(f"Pre-rendering {len(h5_files)} scenes at {args.resolution}px with gsplat...\n")

    total_views = 0
    for idx, h5_path in enumerate(h5_files):
        scene_name = h5_path.stem
        data = load_gaussian_h5(h5_path)
        n_views = data["c2w"].shape[0]

        for v in range(n_views):
            c2w = data["c2w"][v]
            fov_deg = float(data["fov"][v])

            fov_rad = fov_deg * np.pi / 180.0
            focal = 0.5 * args.resolution / np.tan(0.5 * fov_rad)
            K = np.array([
                [focal, 0.0, args.resolution / 2.0],
                [0.0, focal, args.resolution / 2.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)
            viewmat = c2w_to_viewmat(c2w)

            img = render_with_gsplat(
                data["means"], data["rotations"], data["scales"],
                data["opacities"], data["colors"],
                viewmat, K, args.resolution, args.resolution, device,
            )
            img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            out_path = args.output_dir / f"{scene_name}_view_{v}.png"
            imageio.v3.imwrite(out_path, img_u8)
            total_views += 1

        if (idx + 1) % 50 == 0 or idx == len(h5_files) - 1:
            print(f"  [{idx + 1}/{len(h5_files)}] {scene_name} ({n_views} views)")

    print(f"\nDone. Rendered {total_views} views across {len(h5_files)} scenes.")
    print(f"Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
