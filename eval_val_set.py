"""Render a handful of val scenes with the final checkpoint, save side-by-side GT|GF."""
import argparse
from pathlib import Path

import h5py
import imageio
import numpy as np
import torch
from simple_ocio import ToneMapper

from gaussianformer.models.config import GaussianFormerConfig
from gaussianformer.models.gaussianformer import GaussianFormer
from gaussianformer.pipelines.rendering_pipeline import GaussianFormerRenderingPipeline
from infer_gaussian import load_single_gaussian_h5_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints_v4/phase2_epoch_100.pt"))
    ap.add_argument("--h5_dir", type=Path, default=Path("data_v2/h5s_val"))
    ap.add_argument("--gt_dir", type=Path, default=Path("data_v2/renders_val"))
    ap.add_argument("--output_dir", type=Path, default=Path("checkpoint_renders/v4_val"))
    ap.add_argument("--scenes", nargs="+", type=int, default=[0, 1, 2, 5, 10, 20, 50])
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--tone_mapper", type=str, default="agx")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = GaussianFormerConfig()
    model = GaussianFormer(config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "?")
    loss = ckpt.get("loss", 0)
    print(f"Loaded {args.checkpoint.name}: epoch {epoch}, loss {loss:.6f}")

    pipeline = GaussianFormerRenderingPipeline(model)
    pipeline.to(device)

    tm_name = "Khronos PBR Neutral" if args.tone_mapper == "pbr_neutral" else args.tone_mapper
    tone_mapper = ToneMapper(tm_name) if args.tone_mapper != "none" else None

    per_scene_psnr = []
    for si in args.scenes:
        h5_path = args.h5_dir / f"scene_{si:04d}.h5"
        if not h5_path.exists():
            print(f"  missing {h5_path}")
            continue
        data = load_single_gaussian_h5_data(h5_path)
        gaussians = data["gaussians"].unsqueeze(0).to(device)
        mask = data["mask"].unsqueeze(0).to(device)
        c2w_all = data["c2w"].to(device)
        fov_all = data["fov"].to(device)
        with h5py.File(h5_path, "r") as f:
            n_views = f["c2w"].shape[0]

        for v in range(n_views):
            c2w = c2w_all[v:v + 1].unsqueeze(0)
            fov = fov_all[v:v + 1].unsqueeze(0)
            with torch.no_grad():
                out = pipeline(
                    gaussians=gaussians, mask=mask, c2w=c2w, fov=fov,
                    resolution=args.resolution, torch_dtype=torch.float32,
                )
            hdr = out[0, 0].cpu().float().numpy()
            if tone_mapper is not None:
                gf_ldr = tone_mapper.hdr_to_ldr(hdr)
            else:
                gf_ldr = np.clip(hdr, 0, 1)
            gf_ldr = np.clip(gf_ldr, 0, 1)

            gt_path = args.gt_dir / f"scene_{si:04d}_view_{v}.png"
            gt = imageio.v3.imread(gt_path).astype(np.float32) / 255.0
            if gt.shape[0] != args.resolution:
                from PIL import Image
                gt_pil = Image.fromarray((gt * 255).astype(np.uint8)).resize(
                    (args.resolution, args.resolution), Image.LANCZOS)
                gt = np.array(gt_pil).astype(np.float32) / 255.0
            if gt.shape[-1] == 4:
                gt = gt[..., :3]

            mse = ((gf_ldr - gt) ** 2).mean()
            psnr = 10 * np.log10(1.0 / (mse + 1e-12))
            per_scene_psnr.append(psnr)

            pad = np.ones((args.resolution, 4, 3), dtype=np.float32)
            pair = np.concatenate([gt, pad, gf_ldr], axis=1)
            out_path = args.output_dir / f"scene_{si:04d}_view{v}_sbs.png"
            imageio.v3.imwrite(out_path, (np.clip(pair, 0, 1) * 255).astype(np.uint8))
            imageio.v3.imwrite(
                args.output_dir / f"scene_{si:04d}_view{v}_gf.png",
                (np.clip(gf_ldr, 0, 1) * 255).astype(np.uint8),
            )
            print(f"  scene_{si:04d} view {v}: PSNR {psnr:.2f} dB")

    if per_scene_psnr:
        arr = np.array(per_scene_psnr)
        print(f"\nMean PSNR across {len(arr)} views: {arr.mean():.2f} dB  (min {arr.min():.2f}, max {arr.max():.2f})")
    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
