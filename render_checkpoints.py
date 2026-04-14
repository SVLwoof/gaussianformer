"""Render multiple checkpoints to visualize training progression.

Usage:
    uv run python render_checkpoints.py
    uv run python render_checkpoints.py --scenes scene_0000 scene_0050 --epochs 10 50 100
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


def render_scene_with_checkpoint(
    checkpoint_path: Path,
    h5_path: Path,
    output_dir: Path,
    resolution: int,
    tone_mapper,
    device: torch.device,
) -> None:
    """Load a checkpoint and render a scene."""
    config = GaussianFormerConfig()
    model = GaussianFormer(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "?")
    loss = ckpt.get("loss", 0)

    pipeline = GaussianFormerRenderingPipeline(model)
    pipeline.to(device)

    data = load_single_gaussian_h5_data(h5_path)
    gaussians = data["gaussians"].unsqueeze(0).to(device)
    mask = data["mask"].unsqueeze(0).to(device)
    c2w = data["c2w"].unsqueeze(0).to(device)
    fov = data["fov"].unsqueeze(0).unsqueeze(-1).to(device)

    with torch.no_grad():
        rendered_imgs = pipeline(
            gaussians=gaussians, mask=mask, c2w=c2w, fov=fov,
            resolution=resolution, torch_dtype=torch.float32,
        )

    scene_name = h5_path.stem
    nv = c2w.shape[1]
    for i in range(nv):
        hdr_img = rendered_imgs[0, i].cpu().float().numpy()
        if tone_mapper is not None:
            ldr_img = tone_mapper.hdr_to_ldr(hdr_img)
        else:
            ldr_img = np.clip(hdr_img, 0, 1)
        ldr_img = (ldr_img * 255).astype(np.uint8)

        png_path = output_dir / f"{scene_name}_epoch{epoch}_view{i}.png"
        imageio.v3.imwrite(png_path, ldr_img)

    print(f"  epoch {epoch} (loss {loss:.4f}): {nv} views saved")


def main():
    parser = argparse.ArgumentParser(description="Render training progression across checkpoints")
    parser.add_argument("--scenes", nargs="+", default=["scene_0000", "scene_0050", "scene_0100"],
                        help="Scene names to render (without .h5)")
    parser.add_argument("--epochs", nargs="+", type=int, default=[10, 30, 50, 80, 100],
                        help="Phase 2 epochs to render")
    parser.add_argument("--h5_dir", type=Path, default=Path("gaussian_training_h5s"))
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoint_renders/v3"))
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--tone_mapper", type=str, default="agx")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tm_name = "Khronos PBR Neutral" if args.tone_mapper == "pbr_neutral" else args.tone_mapper
    tone_mapper = ToneMapper(tm_name) if args.tone_mapper != "none" else None

    for scene_name in args.scenes:
        h5_path = args.h5_dir / f"{scene_name}.h5"
        if not h5_path.exists():
            print(f"Skipping {scene_name}: {h5_path} not found")
            continue

        print(f"\nRendering {scene_name}:")
        for epoch in args.epochs:
            ckpt_path = args.checkpoint_dir / f"phase2_epoch_{epoch}.pt"
            if not ckpt_path.exists():
                print(f"  Skipping epoch {epoch}: checkpoint not found")
                continue

            render_scene_with_checkpoint(
                ckpt_path, h5_path, args.output_dir,
                args.resolution, tone_mapper, device,
            )


if __name__ == "__main__":
    main()
