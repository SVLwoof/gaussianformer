"""Validate our software splatting renderer against the Voxel51 3DGS dataset.

Downloads a scene from the Voxel51/gaussian_splatting dataset (original 3DGS
paper scenes), loads the PLY, subsamples to a manageable count, and renders
from a synthetic camera pose. Compares against the reference image.

This runs on CPU — no GPU needed. Can run while the GPU trains.

Usage:
    uv run python validate_renderer.py
    uv run python validate_renderer.py --scene truck --max_gaussians 50000
"""

import argparse
from pathlib import Path

import imageio
import numpy as np
from huggingface_hub import hf_hub_download
from plyfile import PlyData
from scipy.spatial.transform import Rotation as R

from render_gsplat import render_gaussians


DATASET_REPO = "Voxel51/gaussian_splatting"
SCENES = ["truck", "train", "playroom", "drjohnson"]


def download_scene(scene: str, cache_dir: Path) -> tuple[Path, Path]:
    """Download PLY and reference image for a scene."""
    ply_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=f"FO_dataset/{scene}/point_cloud/iteration_7000/point_cloud.ply",
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
    ref_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=f"FO_dataset/{scene}/000001.jpg",
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
    return Path(ply_path), Path(ref_path)


def load_3dgs_ply(path: Path) -> dict[str, np.ndarray]:
    """Load a standard 3DGS PLY file (from original Kerbl et al. format).

    The PLY contains: x,y,z, nx,ny,nz, f_dc_0..2, f_rest_0..44,
    opacity, scale_0..2, rot_0..3
    """
    plydata = PlyData.read(str(path))
    vertex = plydata["vertex"]

    means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)

    # Scales are stored as log-scale in original 3DGS
    scales = np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=-1).astype(np.float32)
    scales = np.exp(scales)  # convert from log-space

    # Rotations: quaternion (w, x, y, z) — normalize
    rotations = np.stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=-1).astype(np.float32)
    rotations = rotations / np.linalg.norm(rotations, axis=-1, keepdims=True)

    # Colors: DC component of spherical harmonics → RGB
    # SH DC coefficient to color: color = 0.5 + C0 * sh_dc, where C0 = 0.28209479177
    C0 = 0.28209479177387814
    colors = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=-1).astype(np.float32)
    colors = 0.5 + C0 * colors
    colors = np.clip(colors, 0, 1)

    # Opacity: stored as logit (inverse sigmoid)
    opacities = vertex["opacity"].astype(np.float32)
    opacities = 1.0 / (1.0 + np.exp(-opacities))  # sigmoid

    print(f"Loaded {len(means)} Gaussians from {path.name}")
    print(f"  Means range: [{means.min():.2f}, {means.max():.2f}]")
    print(f"  Scales range: [{scales.min():.4f}, {scales.max():.4f}]")
    print(f"  Colors range: [{colors.min():.3f}, {colors.max():.3f}]")
    print(f"  Opacities range: [{opacities.min():.3f}, {opacities.max():.3f}]")

    return {
        "means": means,
        "scales": scales,
        "rotations": rotations,
        "colors": colors,
        "opacities": opacities,
    }


def subsample_gaussians(data: dict[str, np.ndarray], max_n: int) -> dict[str, np.ndarray]:
    """Subsample Gaussians by opacity (keep the most opaque ones)."""
    n = len(data["means"])
    if n <= max_n:
        return data

    # Sort by opacity descending, keep top max_n
    order = np.argsort(-data["opacities"])[:max_n]
    return {k: v[order] for k, v in data.items()}


def make_orbit_camera(center: np.ndarray, distance: float, elevation: float, azimuth: float, fov: float) -> tuple[np.ndarray, float]:
    """Create a camera-to-world matrix looking at center from orbit position."""
    # Spherical to Cartesian
    el_rad = np.radians(elevation)
    az_rad = np.radians(azimuth)
    x = distance * np.cos(el_rad) * np.sin(az_rad)
    y = distance * np.cos(el_rad) * np.cos(az_rad)
    z = distance * np.sin(el_rad)

    cam_pos = center + np.array([x, y, z], dtype=np.float32)

    # Look-at matrix (Z-up convention matching our pipeline)
    forward = center - cam_pos
    forward = forward / np.linalg.norm(forward)
    up = np.array([0, 0, 1], dtype=np.float32)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    # Build c2w using Blender convention (-Z forward)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward  # -Z = view direction
    c2w[:3, 3] = cam_pos

    return c2w, fov


def main():
    parser = argparse.ArgumentParser(description="Validate splatting renderer against Voxel51 3DGS data")
    parser.add_argument("--scene", type=str, default="truck", choices=SCENES)
    parser.add_argument("--max_gaussians", type=int, default=200000, help="Subsample to this many Gaussians")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--output_dir", type=Path, default=Path("renderer_validation"))
    parser.add_argument("--cache_dir", type=Path, default=Path(".hf_cache"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Download
    print(f"Downloading {args.scene} scene...")
    ply_path, ref_path = download_scene(args.scene, args.cache_dir)

    # Load
    data = load_3dgs_ply(ply_path)

    # Subsample
    data = subsample_gaussians(data, args.max_gaussians)
    print(f"After subsampling: {len(data['means'])} Gaussians")

    # Compute scene center using median (robust to outliers) and extent
    center = np.median(data["means"], axis=0)
    # Use interquartile range for extent (outlier-robust)
    q75 = np.percentile(data["means"], 75, axis=0)
    q25 = np.percentile(data["means"], 25, axis=0)
    iqr_extent = q75 - q25
    distance = float(np.linalg.norm(iqr_extent)) * 4.0

    # Render from multiple viewpoints
    azimuths = [0, 90, 180, 270]
    for az in azimuths:
        c2w, fov = make_orbit_camera(center, distance, elevation=20, azimuth=az, fov=50)
        print(f"Rendering azimuth={az}...")
        img = render_gaussians(
            data["means"], data["scales"], data["rotations"],
            data["colors"], data["opacities"],
            c2w, fov, args.resolution, args.resolution,
            brightness_boost=1.0,  # Voxel51 colors are already proper
            opacity_scale=1.0,
        )
        out_path = args.output_dir / f"{args.scene}_az{az}.png"
        imageio.v3.imwrite(out_path, (img * 255).astype(np.uint8))
        print(f"  Saved {out_path}")

    # Copy reference image
    ref_img = imageio.v3.imread(ref_path)
    ref_out = args.output_dir / f"{args.scene}_reference.jpg"
    imageio.v3.imwrite(ref_out, ref_img)
    print(f"  Saved reference: {ref_out}")

    print(f"\nAll renders saved to {args.output_dir}/")
    print("Compare the rendered views against the reference to validate the renderer.")


if __name__ == "__main__":
    main()
