"""Quick inference from a training checkpoint to check if the model is learning.

Usage:
    uv run python infer_checkpoint.py --checkpoint checkpoints/phase1_epoch_10.pt --h5_file gaussian_training_h5s/scene_0000.h5
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--h5_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoint_renders"))
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--tone_mapper", type=str, default="agx")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model from checkpoint
    config = GaussianFormerConfig()
    model = GaussianFormer(config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')}, loss {ckpt.get('loss', '?'):.6f})")

    pipeline = GaussianFormerRenderingPipeline(model)
    pipeline.to(device)

    # Tone mapper
    tm_name = "Khronos PBR Neutral" if args.tone_mapper == "pbr_neutral" else args.tone_mapper
    tone_mapper = ToneMapper(tm_name) if args.tone_mapper != "none" else None

    # Load scene
    data = load_single_gaussian_h5_data(args.h5_file)
    gaussians = data["gaussians"].unsqueeze(0).to(device)
    mask = data["mask"].unsqueeze(0).to(device)
    c2w = data["c2w"].unsqueeze(0).to(device)
    fov = data["fov"].unsqueeze(0).unsqueeze(-1).to(device)

    with torch.no_grad():
        rendered_imgs = pipeline(
            gaussians=gaussians, mask=mask, c2w=c2w, fov=fov,
            resolution=args.resolution, torch_dtype=torch.float32,
        )

    # Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_name = args.h5_file.stem
    nv = c2w.shape[1]
    for i in range(nv):
        hdr_img = rendered_imgs[0, i].cpu().float().numpy()
        if tone_mapper is not None:
            ldr_img = tone_mapper.hdr_to_ldr(hdr_img)
        else:
            ldr_img = np.clip(hdr_img, 0, 1)
        ldr_img = (ldr_img * 255).astype(np.uint8)

        png_path = args.output_dir / f"{base_name}_view_{i}.png"
        imageio.v3.imwrite(png_path, ldr_img)
        print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
