"""Subsample and normalize raw ModelNet_Splats objects for scene composition.

Reads PLY files from raw_objects/, applies:
  1. Opacity filtering (drop < 0.05)
  2. Subsampling by opacity (keep top-k)
  3. Normalization (center at origin, scale to unit sphere)
  4. Format conversion (SH DC → RGB, log-scale → linear, logit → sigmoid)

Saves processed objects as .npz files in data_v2/objects/.

Usage:
    uv run python data_v2/process_objects.py
    uv run python data_v2/process_objects.py --max_gaussians 3000
"""

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData


def load_3dgs_ply(path: Path) -> dict[str, np.ndarray]:
    """Load a standard 3DGS PLY file, converting to linear values."""
    plydata = PlyData.read(str(path))
    vertex = plydata["vertex"]

    means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)

    # Log-scale → linear
    scales = np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=-1).astype(np.float32)
    scales = np.exp(scales)

    # Quaternion (w, x, y, z), normalized
    rotations = np.stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=-1).astype(np.float32)
    rotations = rotations / (np.linalg.norm(rotations, axis=-1, keepdims=True) + 1e-9)

    # SH DC → RGB
    C0 = 0.28209479177387814
    colors = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=-1).astype(np.float32)
    colors = np.clip(0.5 + C0 * colors, 0, 1)

    # Logit → sigmoid
    opacities = vertex["opacity"].astype(np.float32)
    opacities = 1.0 / (1.0 + np.exp(-opacities))

    return {
        "means": means,
        "scales": scales,
        "rotations": rotations,
        "colors": colors,
        "opacities": opacities,
    }


def process_object(
    data: dict[str, np.ndarray],
    max_gaussians: int,
    min_opacity: float,
) -> dict[str, np.ndarray]:
    """Filter, subsample, and normalize a single object."""
    # 1. Filter by opacity
    keep = data["opacities"] >= min_opacity
    data = {k: v[keep] for k, v in data.items()}

    # 2. Subsample by opacity (keep the most opaque)
    n = len(data["means"])
    if n > max_gaussians:
        order = np.argsort(-data["opacities"])[:max_gaussians]
        data = {k: v[order] for k, v in data.items()}

    # 3. Normalize: center at origin, scale to fit unit sphere
    center = data["means"].mean(axis=0)
    data["means"] = data["means"] - center
    max_dist = np.linalg.norm(data["means"], axis=-1).max()
    if max_dist > 0:
        scale_factor = 0.45 / max_dist  # fit in [-0.45, 0.45]
        data["means"] *= scale_factor
        data["scales"] *= scale_factor

    return data


def main():
    parser = argparse.ArgumentParser(description="Process raw ModelNet_Splats objects")
    parser.add_argument("--raw_dir", type=Path, default=Path("data_v2/raw_objects"))
    parser.add_argument("--output_dir", type=Path, default=Path("data_v2/objects"))
    parser.add_argument("--max_gaussians", type=int, default=3000)
    parser.add_argument("--min_opacity", type=float, default=0.05)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    obj_dirs = sorted(d for d in args.raw_dir.iterdir() if d.is_dir())
    print(f"Processing {len(obj_dirs)} objects → {args.output_dir}/")
    print(f"  max_gaussians={args.max_gaussians}, min_opacity={args.min_opacity}\n")

    for obj_dir in obj_dirs:
        ply_path = obj_dir / "point_cloud.ply"
        if not ply_path.exists():
            print(f"  {obj_dir.name}: SKIP (no point_cloud.ply)")
            continue

        data = load_3dgs_ply(ply_path)
        n_raw = len(data["means"])

        processed = process_object(data, args.max_gaussians, args.min_opacity)
        n_out = len(processed["means"])

        out_path = args.output_dir / f"{obj_dir.name}.npz"
        np.savez_compressed(out_path, **processed)

        bbox = processed["means"].max(0) - processed["means"].min(0)
        print(f"  {obj_dir.name}: {n_raw} → {n_out} Gaussians, "
              f"bbox=[{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}]")

    print(f"\nDone. {len(obj_dirs)} objects saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
