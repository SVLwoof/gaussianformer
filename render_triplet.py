"""Render triplet comparisons: RenderFormer GT | gsplat Gaussians | GaussianFormer output.

This is the primary validation tool. For each scene+view, it produces a
side-by-side triplet showing:
  1. RenderFormer ground truth (from training_renders/)
  2. gsplat render of the Gaussian scene data (pre-rendered in gsplat_renders/)
  3. GaussianFormer model output (rendered on-the-fly from checkpoint)

If the model is perfect, columns 1 and 3 should match. Column 2 shows what
information the model has to work with (the Gaussian input tokens, splatted).

Columns 1 and 2 are loaded from pre-rendered PNGs (no GPU needed for those).
Only column 3 requires GPU. Run render_gsplat_precompute.py first to generate
column 2.

Usage:
    uv run python render_triplet.py --checkpoint checkpoints/phase2_epoch_100.pt
    uv run python render_triplet.py --checkpoint checkpoints/phase2_epoch_100.pt --scenes 0 5 50
    uv run python render_triplet.py --checkpoint checkpoints/phase2_epoch_100.pt --max_scenes 10
"""

import argparse
from pathlib import Path

import imageio
import numpy as np
import torch
from simple_ocio import ToneMapper

from gaussianformer.models.config import GaussianFormerConfig
from gaussianformer.models.gaussianformer import GaussianFormer
from gaussianformer.pipelines.rendering_pipeline import GaussianFormerRenderingPipeline
from infer_gaussian import load_single_gaussian_h5_data


def render_gaussianformer(
    pipeline: GaussianFormerRenderingPipeline,
    h5_path: Path,
    view_idx: int,
    resolution: int,
    tone_mapper: ToneMapper | None,
    device: torch.device,
) -> np.ndarray:
    """Render a scene with GaussianFormer at a specific view."""
    data = load_single_gaussian_h5_data(h5_path)
    gaussians = data["gaussians"].unsqueeze(0).to(device)
    mask = data["mask"].unsqueeze(0).to(device)
    c2w = data["c2w"][view_idx:view_idx + 1].unsqueeze(0).to(device)
    fov = data["fov"][view_idx:view_idx + 1].unsqueeze(0).to(device)

    with torch.no_grad():
        rendered = pipeline(
            gaussians=gaussians, mask=mask, c2w=c2w, fov=fov,
            resolution=resolution, torch_dtype=torch.float32,
        )

    hdr_img = rendered[0, 0].cpu().float().numpy()
    if tone_mapper is not None:
        ldr_img = tone_mapper.hdr_to_ldr(hdr_img)
    else:
        ldr_img = np.clip(hdr_img, 0, 1)
    return ldr_img


def load_image(path: Path, resolution: int) -> np.ndarray | None:
    """Load a PNG image, resized to target resolution if needed."""
    if not path.exists():
        return None
    img = imageio.v3.imread(path).astype(np.float32) / 255.0
    if img.shape[0] != resolution or img.shape[1] != resolution:
        from PIL import Image
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        pil_img = pil_img.resize((resolution, resolution), Image.LANCZOS)
        img = np.array(pil_img).astype(np.float32) / 255.0
    return img


def make_triplet(
    gt_img: np.ndarray | None,
    gsplat_img: np.ndarray | None,
    gf_img: np.ndarray,
    resolution: int,
) -> np.ndarray:
    """Create a labeled triplet image: GT | gsplat | GaussianFormer."""
    label_height = 24
    total_h = resolution + label_height
    total_w = resolution * 3 + 4  # 2px gap between panels

    canvas = np.ones((total_h, total_w, 3), dtype=np.float32)

    # Place images (leave top rows for labels)
    for i, img in enumerate([gt_img, gsplat_img, gf_img]):
        x_start = i * (resolution + 2)
        if img is not None:
            canvas[label_height:, x_start:x_start + resolution] = img[:, :, :3]
        else:
            canvas[label_height:, x_start:x_start + resolution] = 0.3

    # Dark label bars
    for i in range(3):
        x_start = i * (resolution + 2)
        canvas[:label_height, x_start:x_start + resolution] = 0.15

    return np.clip(canvas, 0, 1)


def main():
    parser = argparse.ArgumentParser(description="Render triplet comparisons")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="GaussianFormer checkpoint (.pt file)")
    parser.add_argument("--h5_dir", type=Path, default=Path("gaussian_training_h5s"))
    parser.add_argument("--gt_dir", type=Path, default=Path("training_renders"))
    parser.add_argument("--gsplat_dir", type=Path, default=Path("gsplat_renders"))
    parser.add_argument("--output_dir", type=Path, default=Path("triplet_renders"))
    parser.add_argument("--scenes", type=int, nargs="+", default=None,
                        help="Specific scene indices to render (e.g., 0 5 50)")
    parser.add_argument("--max_scenes", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--tone_mapper", type=str, default="agx")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load GaussianFormer model
    print(f"Loading checkpoint: {args.checkpoint}")
    config = GaussianFormerConfig()
    model = GaussianFormer(config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "?")
    loss = ckpt.get("loss", 0)
    print(f"  Epoch {epoch}, loss {loss:.6f}")

    pipeline = GaussianFormerRenderingPipeline(model)
    pipeline.to(device)

    # Tone mapper for GaussianFormer HDR output
    tm_name = "Khronos PBR Neutral" if args.tone_mapper == "pbr_neutral" else args.tone_mapper
    tone_mapper = ToneMapper(tm_name) if args.tone_mapper != "none" else None

    # Select scenes
    all_h5s = sorted(args.h5_dir.glob("*.h5"))
    if args.scenes is not None:
        h5_files = [args.h5_dir / f"scene_{i:04d}.h5" for i in args.scenes]
        h5_files = [f for f in h5_files if f.exists()]
    else:
        h5_files = all_h5s[:args.max_scenes]

    print(f"Rendering {len(h5_files)} scenes as triplets...\n")

    for h5_path in h5_files:
        scene_name = h5_path.stem
        # Count views from H5
        import h5py
        with h5py.File(h5_path, "r") as f:
            n_views = f["c2w"].shape[0]
        print(f"  {scene_name}: {n_views} view(s)")

        for v in range(n_views):
            # 1. RenderFormer ground truth (pre-rendered PNG)
            gt_img = load_image(
                args.gt_dir / f"{scene_name}_view_{v}.png", args.resolution,
            )

            # 2. gsplat render (pre-rendered PNG)
            gsplat_img = load_image(
                args.gsplat_dir / f"{scene_name}_view_{v}.png", args.resolution,
            )

            # 3. GaussianFormer output (rendered now)
            gf_img = render_gaussianformer(
                pipeline, h5_path, v, args.resolution, tone_mapper, device,
            )

            # Build triplet
            triplet = make_triplet(gt_img, gsplat_img, gf_img, args.resolution)
            out_path = args.output_dir / f"{scene_name}_view{v}_triplet.png"
            imageio.v3.imwrite(out_path, (triplet * 255).astype(np.uint8))

            # Also save individual GF panel
            imageio.v3.imwrite(
                args.output_dir / f"{scene_name}_view{v}_gf.png",
                (np.clip(gf_img, 0, 1) * 255).astype(np.uint8),
            )

            gt_status = "ok" if gt_img is not None else "MISSING"
            gs_status = "ok" if gsplat_img is not None else "MISSING"
            print(f"    view {v}: GT={gt_status}, gsplat={gs_status}")

    print(f"\nAll triplets saved to {args.output_dir}/")
    print("Each triplet: [RenderFormer GT] | [gsplat Gaussians] | [GaussianFormer output]")


if __name__ == "__main__":
    main()
