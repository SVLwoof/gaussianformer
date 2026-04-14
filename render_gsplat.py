"""Render Gaussian H5 scenes using a simple software splatting renderer.

Produces renders from the same camera viewpoints as the training data,
allowing direct comparison between:
  - This splatted render (what the Gaussians look like)
  - RenderFormer ground truth (training_renders/)
  - GaussianFormer output (checkpoint_renders/)

No CUDA toolkit required -- runs on CPU with numpy.

Usage:
    uv run python render_gsplat.py --h5_file gaussian_training_h5s/scene_0000.h5
    uv run python render_gsplat.py --h5_dir gaussian_training_h5s --max_scenes 5
"""

import argparse
from pathlib import Path

import h5py
import imageio
import numpy as np
from scipy.spatial.transform import Rotation as R


def load_gaussian_h5(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        return {
            "means": np.array(f["means"], dtype=np.float32),
            "scales": np.array(f["scales"], dtype=np.float32),
            "rotations": np.array(f["rotations"], dtype=np.float32),
            "colors": np.array(f["colors"], dtype=np.float32),
            "opacities": np.array(f["opacities"], dtype=np.float32).squeeze(-1),
            "c2w": np.array(f["c2w"], dtype=np.float32),
            "fov": np.array(f["fov"], dtype=np.float32),
        }


def render_gaussians(
    means: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    c2w: np.ndarray,
    fov_deg: float,
    width: int,
    height: int,
    brightness_boost: float = 3.0,
    opacity_scale: float = 0.5,
) -> np.ndarray:
    """Render Gaussians via EWA splatting with proper covariance projection.

    Each 3D Gaussian's full covariance (from scales + quaternion rotation) is
    projected to a 2D covariance on the image plane using the perspective
    Jacobian, then rasterized as an oriented ellipse with Gaussian falloff.
    """
    # World-to-camera transform
    w2c = np.linalg.inv(c2w)
    R_w2c = w2c[:3, :3]
    t_w2c = w2c[:3, 3]

    # Intrinsics
    fov_rad = fov_deg * np.pi / 180.0
    focal = 0.5 * width / np.tan(0.5 * fov_rad)
    cx, cy = width / 2.0, height / 2.0

    # Transform means to camera space
    means_cam = (R_w2c @ means.T).T + t_w2c  # [N, 3]

    # Blender convention: -Z is forward, so visible points have z < 0
    # Flip to standard pinhole (z > 0 = in front)
    means_cam[:, 1] *= -1
    means_cam[:, 2] *= -1

    # Build flip matrix for transforming covariances consistently
    flip = np.diag([1.0, -1.0, -1.0])
    R_cam = flip @ R_w2c  # combined rotation: world → flipped camera

    # Filter: keep only points in front of camera
    valid = means_cam[:, 2] > 0.01
    if not valid.any():
        return np.zeros((height, width, 3), dtype=np.float32)

    means_cam = means_cam[valid]
    scales_v = scales[valid]
    rotations_v = rotations[valid]  # (w, x, y, z) quaternions
    colors_v = colors[valid]
    opacities_v = opacities[valid]

    # Project to 2D pixel coordinates
    depths = means_cam[:, 2]
    px = focal * means_cam[:, 0] / depths + cx
    py = focal * means_cam[:, 1] / depths + cy

    # Build 3D covariance and project to 2D for each Gaussian
    # Convert quaternions from (w,x,y,z) to scipy's (x,y,z,w)
    scipy_quats = rotations_v[:, [1, 2, 3, 0]]
    rot_mats = R.from_quat(scipy_quats).as_matrix()  # [N, 3, 3]

    # 3D covariance: Σ = R @ diag(s²) @ R^T
    # Then transform to camera space: Σ_cam = R_cam @ Σ @ R_cam^T
    # Then project to 2D via Jacobian
    n = len(depths)
    cov2d_list = np.zeros((n, 2, 2), dtype=np.float32)

    for i in range(n):
        S = np.diag(scales_v[i] ** 2)
        Sigma = rot_mats[i] @ S @ rot_mats[i].T  # 3D covariance in world
        Sigma_cam = R_cam @ Sigma @ R_cam.T  # 3D covariance in camera

        # Perspective projection Jacobian at this point's depth
        z = depths[i]
        tx = means_cam[i, 0]
        ty = means_cam[i, 1]
        J = np.array([
            [focal / z, 0.0, -focal * tx / (z * z)],
            [0.0, focal / z, -focal * ty / (z * z)],
        ], dtype=np.float32)

        cov2d_list[i] = J @ Sigma_cam @ J.T

    # Boost colors for visualization (raw material colors are very dark
    # because they lack lighting simulation)
    colors_v = np.clip(colors_v * brightness_boost, 0, 1)

    # Scale opacity for visualization (raw opacity is 1.0 for all Gaussians,
    # which causes complete occlusion; reducing it allows see-through blending)
    opacities_v = opacities_v * opacity_scale

    # Sort front-to-back for proper alpha compositing with transmittance
    order = np.argsort(depths)
    px, py = px[order], py[order]
    depths_sorted = depths[order]
    cov2d_list = cov2d_list[order]
    colors_v = colors_v[order]
    opacities_v = opacities_v[order]

    # Rasterize (front-to-back with transmittance tracking)
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    T_acc = np.ones((height, width), dtype=np.float32)  # remaining transmittance

    for i in range(len(px)):
        x, y = px[i], py[i]
        alpha = opacities_v[i]
        color = colors_v[i]
        cov = cov2d_list[i].copy()

        # Add small regularization for numerical stability
        cov[0, 0] += 0.3
        cov[1, 1] += 0.3

        # Invert 2D covariance for Mahalanobis distance
        det = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
        if det <= 1e-6:
            continue
        inv_cov = np.array([
            [cov[1, 1] / det, -cov[0, 1] / det],
            [-cov[1, 0] / det, cov[0, 0] / det],
        ], dtype=np.float32)

        # Bounding box from eigenvalues (3-sigma)
        trace = cov[0, 0] + cov[1, 1]
        disc = max(trace * trace / 4.0 - det, 0.0)
        lambda_max = trace / 2.0 + np.sqrt(disc)
        radius = 3.0 * np.sqrt(lambda_max)

        x0 = max(int(x - radius), 0)
        x1 = min(int(x + radius) + 1, width)
        y0 = max(int(y - radius), 0)
        y1 = min(int(y + radius) + 1, height)

        if x0 >= x1 or y0 >= y1:
            continue

        # Skip if this region is already fully opaque
        T_region = T_acc[y0:y1, x0:x1]
        if T_region.max() < 0.01:
            continue

        # Pixel grid
        gx = np.arange(x0, x1, dtype=np.float32) + 0.5 - x
        gy = np.arange(y0, y1, dtype=np.float32) + 0.5 - y

        # Mahalanobis distance: d² = [dx, dy] @ inv_cov @ [dx, dy]^T
        dx_grid, dy_grid = np.meshgrid(gx, gy)
        maha = (
            inv_cov[0, 0] * dx_grid * dx_grid
            + (inv_cov[0, 1] + inv_cov[1, 0]) * dx_grid * dy_grid
            + inv_cov[1, 1] * dy_grid * dy_grid
        )
        gauss = np.exp(-0.5 * maha)
        pixel_alpha = alpha * gauss

        # Front-to-back compositing: color += T * alpha * c; T *= (1 - alpha)
        contribution = T_region * pixel_alpha
        canvas[y0:y1, x0:x1] += contribution[:, :, np.newaxis] * color
        T_acc[y0:y1, x0:x1] *= (1 - pixel_alpha)

    # Apply gamma correction for display
    canvas = np.clip(canvas, 0, 1)
    canvas = canvas ** (1.0 / 2.2)
    return canvas


def render_scene(data: dict, output_dir: Path, scene_name: str, resolution: int):
    num_views = data["c2w"].shape[0]
    for v in range(num_views):
        img = render_gaussians(
            data["means"], data["scales"], data["rotations"],
            data["colors"], data["opacities"],
            data["c2w"][v], data["fov"][v], resolution, resolution,
        )
        img_uint8 = (img * 255).astype(np.uint8)
        out_path = output_dir / f"{scene_name}_view_{v}.png"
        imageio.v3.imwrite(out_path, img_uint8)
        print(f"  Saved {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Render Gaussian scenes (software splatting)")
    parser.add_argument("--h5_file", type=Path, help="Single H5 file to render")
    parser.add_argument("--h5_dir", type=Path, help="Directory of H5 files to render")
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("gsplat_renders"))
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.h5_file:
        h5_files = [args.h5_file]
    elif args.h5_dir:
        h5_files = sorted(args.h5_dir.glob("*.h5"))
        if args.max_scenes:
            h5_files = h5_files[:args.max_scenes]
    else:
        parser.error("Provide --h5_file or --h5_dir")

    for i, h5_path in enumerate(h5_files, 1):
        print(f"[{i}/{len(h5_files)}] Rendering {h5_path.stem}...")
        data = load_gaussian_h5(h5_path)
        render_scene(data, args.output_dir, h5_path.stem, args.resolution)

    print(f"\nAll renders saved to {args.output_dir}")


if __name__ == "__main__":
    main()
