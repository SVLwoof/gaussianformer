"""Diagnostic script for inspecting Gaussian H5 training data."""

import sys
from pathlib import Path

import h5py
import numpy as np


def inspect_h5(path: Path) -> dict:
    """Read a Gaussian H5 file and return diagnostic stats."""
    with h5py.File(path, "r") as f:
        means = np.array(f["means"])
        scales = np.array(f["scales"])
        rotations = np.array(f["rotations"])
        colors = np.array(f["colors"])
        opacities = np.array(f["opacities"])
        c2w = np.array(f["c2w"])
        fov = np.array(f["fov"])

    n = means.shape[0]

    # Quaternion norms (should be ~1.0)
    quat_norms = np.linalg.norm(rotations, axis=1)

    # Unique colors (rounded to 2 decimal places)
    unique_colors = len(np.unique(np.round(colors, 2), axis=0))

    # c2w determinant of rotation part (should be ~1.0 for rigid transforms)
    det = np.linalg.det(c2w[:, :3, :3])

    return {
        "name": path.stem,
        "num_gaussians": n,
        "pos_min": means.min(axis=0),
        "pos_max": means.max(axis=0),
        "scale_min": scales.min(),
        "scale_max": scales.max(),
        "scale_mean": scales.mean(),
        "color_min": colors.min(),
        "color_max": colors.max(),
        "color_mean": colors.mean(),
        "unique_colors": unique_colors,
        "opacity_min": opacities.min(),
        "opacity_max": opacities.max(),
        "quat_norm_min": quat_norms.min(),
        "quat_norm_max": quat_norms.max(),
        "num_cameras": c2w.shape[0],
        "fov": fov.tolist(),
        "c2w_det": det.tolist(),
    }


def main():
    h5_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("gaussian_training_h5s")
    if not h5_dir.is_dir():
        print(f"Directory not found: {h5_dir}")
        return

    h5_files = sorted(h5_dir.glob("*.h5"))
    if not h5_files:
        print(f"No H5 files found in {h5_dir}")
        return

    print(f"Inspecting {len(h5_files)} H5 files in {h5_dir}\n")
    print(f"{'Scene':<20} {'#Gauss':>8} {'#Cam':>5} {'Colors':>8} "
          f"{'Color range':>14} {'Scale range':>18} {'Quat norm':>14}")
    print("-" * 100)

    total_gaussians = 0
    all_unique_colors = 0

    for path in h5_files:
        s = inspect_h5(path)
        total_gaussians += s["num_gaussians"]
        all_unique_colors = max(all_unique_colors, s["unique_colors"])

        print(f"{s['name']:<20} {s['num_gaussians']:>8} {s['num_cameras']:>5} "
              f"{s['unique_colors']:>8} "
              f"[{s['color_min']:.3f}, {s['color_max']:.3f}] "
              f"[{s['scale_min']:.5f}, {s['scale_max']:.5f}] "
              f"[{s['quat_norm_min']:.4f}, {s['quat_norm_max']:.4f}]")

    print("-" * 100)
    print(f"Total: {len(h5_files)} scenes, {total_gaussians} Gaussians, "
          f"max unique colors in a scene: {all_unique_colors}")


if __name__ == "__main__":
    main()
