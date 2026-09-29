"""Render every camera stored in an HDF5 scene with GaussianFormer and write one PNG per view.

  python infer.py --h5 scene.h5 --out renders/
"""
import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from gaussianformer.pipelines.rendering_pipeline import GaussianFormerRenderingPipeline
from training.dataset import load_gaussians

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", type=Path, required=True, help="scene with means, scales, rotations, colors, opacities, c2w, fov")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="shahafvl/gaussianformer", help="Hugging Face id or local directory")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--precision", choices=DTYPES, default="bf16")
    a = ap.parse_args()

    pipe = GaussianFormerRenderingPipeline.from_pretrained(a.model).to(torch.device("cuda"))
    gaussians, c2w, fov = (torch.from_numpy(x)[None].cuda() for x in load_gaussians(a.h5))
    mask = torch.ones(gaussians.shape[:2], dtype=torch.bool, device="cuda")
    images = pipe(gaussians, mask, c2w, fov, resolution=a.resolution, torch_dtype=DTYPES[a.precision])
    a.out.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images[0].float().clamp(0, 1).cpu().numpy()):
        iio.imwrite(a.out / f"{a.h5.stem}_view_{i}.png", (img * 255).round().astype(np.uint8))
    print(f"wrote {len(images[0])} views to {a.out}")


if __name__ == "__main__":
    main()
