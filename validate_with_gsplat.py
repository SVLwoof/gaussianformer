"""Validate our software splatting renderer against gsplat (reference GPU renderer).

Renders identical Gaussians from identical cameras with both renderers and
compares pixel-by-pixel. Three test tiers:
  1. Synthetic unit test (5 Gaussians) — eliminates all ambiguity
  2. Training H5 scene (1504 Gaussians) — tests real data pipeline
  3. Voxel51 truck scene (1.69M Gaussians) — stress test at scale

Usage:
    uv run python validate_with_gsplat.py
    uv run python validate_with_gsplat.py --tier 1        # synthetic only
    uv run python validate_with_gsplat.py --tier 2        # training data
    uv run python validate_with_gsplat.py --tier 3        # Voxel51 truck
"""

import argparse
from pathlib import Path

import imageio
import numpy as np
import torch

from render_gsplat import render_gaussians, load_gaussian_h5


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images in [0, 1] range."""
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return float("inf")
    return -10.0 * np.log10(mse)


def save_comparison(
    img_ours: np.ndarray,
    img_ref: np.ndarray,
    output_dir: Path,
    label: str,
) -> float:
    """Save side-by-side comparison and difference image. Returns PSNR."""
    psnr = compute_psnr(img_ours, img_ref)

    # Side-by-side
    combined = np.concatenate([img_ours, img_ref], axis=1)
    combined_u8 = (np.clip(combined, 0, 1) * 255).astype(np.uint8)
    imageio.v3.imwrite(output_dir / f"{label}_sidebyside.png", combined_u8)

    # Difference (amplified 5x for visibility)
    diff = np.abs(img_ours - img_ref)
    diff_amp = np.clip(diff * 5.0, 0, 1)
    diff_u8 = (diff_amp * 255).astype(np.uint8)
    imageio.v3.imwrite(output_dir / f"{label}_diff5x.png", diff_u8)

    # Individual images
    imageio.v3.imwrite(output_dir / f"{label}_ours.png", (np.clip(img_ours, 0, 1) * 255).astype(np.uint8))
    imageio.v3.imwrite(output_dir / f"{label}_gsplat.png", (np.clip(img_ref, 0, 1) * 255).astype(np.uint8))

    return psnr


def make_camera(
    position: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    fov_deg: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Create c2w (Blender convention) and K (intrinsic matrix).

    Returns: (c2w [4,4], K [3,3], fov_deg)
    """
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    cam_up = np.cross(right, forward)

    # Blender convention: -Z = view direction, +Y = up, +X = right
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = cam_up
    c2w[:3, 2] = -forward  # -Z = view direction
    c2w[:3, 3] = position

    fov_rad = fov_deg * np.pi / 180.0
    focal = 0.5 * width / np.tan(0.5 * fov_rad)
    K = np.array([
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    return c2w, K, fov_deg


def c2w_to_viewmat(c2w: np.ndarray) -> np.ndarray:
    """Convert Blender-convention c2w to gsplat viewmat (pinhole w2c).

    Our renderer internally does:
      w2c = inv(c2w)
      means_cam[:, 1] *= -1  (flip Y)
      means_cam[:, 2] *= -1  (flip Z)

    For gsplat, we bake the flip into the viewmat:
      viewmat = flip @ inv(c2w)
    """
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
    """Render Gaussians using gsplat (reference renderer)."""
    import gsplat

    means_t = torch.tensor(means, dtype=torch.float32, device=device)
    quats_t = torch.tensor(quats, dtype=torch.float32, device=device)
    scales_t = torch.tensor(scales, dtype=torch.float32, device=device)
    # gsplat expects opacities as [N] with values in (0, 1)
    opacities_t = torch.tensor(opacities, dtype=torch.float32, device=device)
    colors_t = torch.tensor(colors, dtype=torch.float32, device=device)

    viewmats = torch.tensor(viewmat, dtype=torch.float32, device=device).unsqueeze(0)  # [1, 4, 4]
    Ks = torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0)  # [1, 3, 3]

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
            sh_degree=None,  # raw RGB, not SH
            eps2d=0.3,  # match our renderer's cov2d regularization
            render_mode="RGB",
            near_plane=0.01,
        )

    # renders shape: [1, H, W, 3]
    return renders[0].cpu().numpy()


def render_with_ours(
    means: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    c2w: np.ndarray,
    fov_deg: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Render Gaussians using our software renderer (no visualization hacks)."""
    img = render_gaussians(
        means, scales, rotations, colors, opacities,
        c2w, fov_deg, width, height,
        brightness_boost=1.0,
        opacity_scale=1.0,
    )
    # Remove gamma correction — compare in linear space
    # Our renderer applies gamma: canvas ** (1/2.2)
    # Undo it: canvas ** 2.2
    img = np.clip(img, 0, 1) ** 2.2
    return img


# ── Tier 1: Synthetic unit test ──────────────────────────────────────────


def create_synthetic_scene() -> dict[str, np.ndarray]:
    """Create a simple 5-Gaussian scene with known properties."""
    means = np.array([
        [0.0, 0.0, 0.0],     # center — white
        [0.5, 0.0, 0.0],     # right — red
        [-0.5, 0.0, 0.0],    # left — green
        [0.0, 0.5, 0.0],     # up — blue
        [0.0, -0.5, 0.0],    # down — yellow
    ], dtype=np.float32)

    scales = np.full((5, 3), 0.15, dtype=np.float32)

    # Identity quaternions (w, x, y, z)
    rotations = np.zeros((5, 4), dtype=np.float32)
    rotations[:, 0] = 1.0  # w=1

    colors = np.array([
        [1.0, 1.0, 1.0],  # white
        [1.0, 0.0, 0.0],  # red
        [0.0, 1.0, 0.0],  # green
        [0.0, 0.0, 1.0],  # blue
        [1.0, 1.0, 0.0],  # yellow
    ], dtype=np.float32)

    opacities = np.ones(5, dtype=np.float32)

    return {
        "means": means,
        "scales": scales,
        "rotations": rotations,
        "colors": colors,
        "opacities": opacities,
    }


def test_synthetic(output_dir: Path, device: torch.device) -> float:
    """Tier 1: Render 5 synthetic Gaussians with both renderers."""
    print("\n=== Tier 1: Synthetic Unit Test (5 Gaussians) ===")

    data = create_synthetic_scene()
    W, H = 256, 256

    # Camera at z=3 looking at origin
    c2w, K, fov_deg = make_camera(
        position=np.array([0.0, 0.0, 3.0]),
        target=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
        fov_deg=60.0,
        width=W, height=H,
    )
    viewmat = c2w_to_viewmat(c2w)

    print(f"  Camera position: {c2w[:3, 3]}")
    print(f"  View direction: {-c2w[:3, 2]}")

    # Render with gsplat
    img_gsplat = render_with_gsplat(
        data["means"], data["rotations"], data["scales"],
        data["opacities"], data["colors"],
        viewmat, K, W, H, device,
    )
    print(f"  gsplat render: min={img_gsplat.min():.3f}, max={img_gsplat.max():.3f}")

    # Render with ours
    img_ours = render_with_ours(
        data["means"], data["scales"], data["rotations"],
        data["colors"], data["opacities"],
        c2w, fov_deg, W, H,
    )
    print(f"  Our render:    min={img_ours.min():.3f}, max={img_ours.max():.3f}")

    psnr = save_comparison(img_ours, img_gsplat, output_dir, "tier1_synthetic")
    print(f"  PSNR: {psnr:.1f} dB")
    return psnr


# ── Tier 2: Training H5 scene ────────────────────────────────────────────


def test_training_h5(h5_path: Path, output_dir: Path, device: torch.device) -> float:
    """Tier 2: Render a training H5 scene with both renderers."""
    print(f"\n=== Tier 2: Training H5 Scene ({h5_path.name}) ===")

    data = load_gaussian_h5(h5_path)
    n_gaussians = len(data["means"])
    n_views = data["c2w"].shape[0]
    print(f"  {n_gaussians} Gaussians, {n_views} view(s)")

    W, H = 256, 256
    # Use the first view from the H5
    c2w = data["c2w"][0]
    fov_deg = float(data["fov"][0])

    fov_rad = fov_deg * np.pi / 180.0
    focal = 0.5 * W / np.tan(0.5 * fov_rad)
    K = np.array([
        [focal, 0.0, W / 2.0],
        [0.0, focal, H / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    viewmat = c2w_to_viewmat(c2w)

    # Render with gsplat
    img_gsplat = render_with_gsplat(
        data["means"], data["rotations"], data["scales"],
        data["opacities"], data["colors"],
        viewmat, K, W, H, device,
    )
    print(f"  gsplat render: min={img_gsplat.min():.3f}, max={img_gsplat.max():.3f}")

    # Render with ours
    img_ours = render_with_ours(
        data["means"], data["scales"], data["rotations"],
        data["colors"], data["opacities"],
        c2w, fov_deg, W, H,
    )
    print(f"  Our render:    min={img_ours.min():.3f}, max={img_ours.max():.3f}")

    psnr = save_comparison(img_ours, img_gsplat, output_dir, f"tier2_{h5_path.stem}")
    print(f"  PSNR: {psnr:.1f} dB")
    return psnr


# ── Tier 3: Voxel51 truck scene ──────────────────────────────────────────


def test_voxel51(output_dir: Path, device: torch.device, max_gaussians: int = 200000) -> float | None:
    """Tier 3: Render Voxel51 truck scene with gsplat (full) and ours (subsampled)."""
    print(f"\n=== Tier 3: Voxel51 Truck Scene ===")

    try:
        from validate_renderer import download_scene, load_3dgs_ply, subsample_gaussians, make_orbit_camera
    except ImportError:
        print("  Skipping: validate_renderer.py not found")
        return None

    cache_dir = Path(".hf_cache")
    try:
        ply_path, ref_path = download_scene("truck", cache_dir)
    except Exception as e:
        print(f"  Skipping: download failed ({e})")
        return None

    data = load_3dgs_ply(ply_path)
    n_full = len(data["means"])

    # Compute camera at reasonable distance
    center = np.median(data["means"], axis=0)
    q75 = np.percentile(data["means"], 75, axis=0)
    q25 = np.percentile(data["means"], 25, axis=0)
    iqr_extent = q75 - q25
    distance = float(np.linalg.norm(iqr_extent)) * 2.0

    W, H = 512, 512
    c2w, fov_deg = make_orbit_camera(center, distance, elevation=20, azimuth=45, fov=50)

    fov_rad = fov_deg * np.pi / 180.0
    focal = 0.5 * W / np.tan(0.5 * fov_rad)
    K = np.array([
        [focal, 0.0, W / 2.0],
        [0.0, focal, H / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    viewmat = c2w_to_viewmat(c2w)

    # gsplat: render ALL Gaussians (takes milliseconds on GPU)
    print(f"  Rendering {n_full} Gaussians with gsplat...")
    img_gsplat = render_with_gsplat(
        data["means"], data["rotations"], data["scales"],
        data["opacities"], data["colors"],
        viewmat, K, W, H, device,
    )
    imageio.v3.imwrite(output_dir / "tier3_truck_gsplat.png", (np.clip(img_gsplat, 0, 1) * 255).astype(np.uint8))
    print(f"  gsplat render: min={img_gsplat.min():.3f}, max={img_gsplat.max():.3f}")

    # Ours: subsampled
    data_sub = subsample_gaussians(data, max_gaussians)
    print(f"  Rendering {len(data_sub['means'])} Gaussians with our renderer...")
    img_ours = render_with_ours(
        data_sub["means"], data_sub["scales"], data_sub["rotations"],
        data_sub["colors"], data_sub["opacities"],
        c2w, fov_deg, W, H,
    )
    imageio.v3.imwrite(output_dir / "tier3_truck_ours.png", (np.clip(img_ours, 0, 1) * 255).astype(np.uint8))
    print(f"  Our render:    min={img_ours.min():.3f}, max={img_ours.max():.3f}")

    # Save reference image
    ref_img = imageio.v3.imread(ref_path)
    imageio.v3.imwrite(output_dir / "tier3_truck_reference.jpg", ref_img)

    # Side-by-side (not pixel-comparable due to different Gaussian counts)
    combined = np.concatenate([
        np.clip(img_ours, 0, 1),
        np.clip(img_gsplat, 0, 1),
    ], axis=1)
    imageio.v3.imwrite(output_dir / "tier3_truck_sidebyside.png", (combined * 255).astype(np.uint8))

    print(f"  Note: Tier 3 is a visual comparison only (different Gaussian counts)")
    print(f"  Saved side-by-side: ours ({len(data_sub['means'])}G) | gsplat ({n_full}G)")
    return None


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Validate renderer against gsplat reference")
    parser.add_argument("--tier", type=int, nargs="+", default=[1, 2],
                        help="Test tiers to run (1=synthetic, 2=H5, 3=Voxel51)")
    parser.add_argument("--h5_file", type=Path, default=Path("gaussian_training_h5s/scene_0000.h5"))
    parser.add_argument("--output_dir", type=Path, default=Path("renderer_validation/gsplat_comparison"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = {}

    if 1 in args.tier:
        results["Tier 1 (synthetic)"] = test_synthetic(args.output_dir, device)

    if 2 in args.tier:
        results["Tier 2 (H5)"] = test_training_h5(args.h5_file, args.output_dir, device)

    if 3 in args.tier:
        results["Tier 3 (Voxel51)"] = test_voxel51(args.output_dir, device)

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    for name, psnr in results.items():
        if psnr is None:
            print(f"  {name}: visual comparison only")
        elif psnr > 30:
            print(f"  {name}: PSNR = {psnr:.1f} dB  *** PASS ***")
        elif psnr > 20:
            print(f"  {name}: PSNR = {psnr:.1f} dB  ** MARGINAL **")
        else:
            print(f"  {name}: PSNR = {psnr:.1f} dB  * FAIL *")

    passing = all(v is None or v > 25 for v in results.values())
    if passing:
        print("\nRenderer validation PASSED. Foundation is correct.")
    else:
        print("\nRenderer validation FAILED. Check difference images for diagnosis.")
        print("  - Shifted positions → camera convention bug")
        print("  - Wrong shapes → covariance projection bug")
        print("  - Wrong colors → compositing bug")


if __name__ == "__main__":
    main()
