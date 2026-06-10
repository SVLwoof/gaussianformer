"""Full val-set eval: PSNR + LPIPS over all scenes/views for one checkpoint.

Designed as the V13-comparator workhorse. Renders each H5 scene under the loaded
checkpoint, tone-maps to LDR (matches GT pipeline), computes PSNR and LPIPS vs
the GT renders, writes per-scene means and a global summary as JSON.

Inputs:
  --checkpoint path/to/phaseN_epoch_M.pt
  --pe_type {rope, nerf}    (must match how the ckpt was trained)
  --h5_dir   data_v9_n20k/h5s_val          (val H5s, any target_n)
  --gt_dir   data_v9/renders_val           (GT PNGs from FULL-scene raster)
  --out_json path/to/result.json

Single-GPU, bs=1, bf16 inference. ~60-90 min for 183 scenes x 14 views at N=20k.
"""
import argparse
import json
import time
from pathlib import Path

import h5py
import imageio.v3 as iio
import lpips
import numpy as np
import torch
from PIL import Image
from simple_ocio import ToneMapper

from gaussianformer.models.config import GaussianFormerConfig
from gaussianformer.models.gaussianformer import GaussianFormer
from gaussianformer.pipelines.rendering_pipeline import GaussianFormerRenderingPipeline
from infer_gaussian import load_single_gaussian_h5_data


def load_gt(gt_path: Path, resolution: int) -> np.ndarray:
    gt = iio.imread(gt_path).astype(np.float32) / 255.0
    if gt.shape[0] != resolution:
        gt = np.asarray(Image.fromarray((gt * 255).astype(np.uint8)).resize(
            (resolution, resolution), Image.LANCZOS)).astype(np.float32) / 255.0
    if gt.shape[-1] == 4:
        gt = gt[..., :3]
    return gt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--pe_type", type=str, default="rope",
                    choices=["rope", "nerf"])
    ap.add_argument("--h5_dir", type=Path, default=Path("data_v9_n20k/h5s_val"))
    ap.add_argument("--gt_dir", type=Path, default=Path("data_v9/renders_val"))
    ap.add_argument("--out_json", type=Path, required=True)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--tone_mapper", type=str, default="none",
                    help="MUST match the GT pipeline. data_v9 GT renders are written "
                    "with NO tone map (gsplat source is already LDR), so 'none' (clip) "
                    "is correct. AGX desaturates and corrupts both the render and the "
                    "PSNR/LPIPS metrics -- do not use it here.")
    ap.add_argument("--max_scenes", type=int, default=None,
                    help="If set, only evaluate first N scenes (smoke test).")
    args = ap.parse_args()

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = GaussianFormerConfig(pe_type=args.pe_type)
    model = GaussianFormer(config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded {args.checkpoint} (epoch {epoch}, pe_type={args.pe_type})")

    model.eval()
    pipeline = GaussianFormerRenderingPipeline(model)
    pipeline.to(device)

    tm_name = "Khronos PBR Neutral" if args.tone_mapper == "pbr_neutral" else args.tone_mapper
    tone_mapper = ToneMapper(tm_name) if args.tone_mapper != "none" else None

    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()

    scene_h5s = sorted(args.h5_dir.glob("scene_*.h5"))
    if args.max_scenes is not None:
        scene_h5s = scene_h5s[: args.max_scenes]
    print(f"Evaluating {len(scene_h5s)} scenes from {args.h5_dir}")

    per_scene = []
    t_start = time.time()
    for si, h5_path in enumerate(scene_h5s):
        data = load_single_gaussian_h5_data(h5_path)
        gaussians = data["gaussians"].unsqueeze(0).to(device)
        mask = data["mask"].unsqueeze(0).to(device)
        c2w_all = data["c2w"].to(device)
        fov_all = data["fov"].to(device)
        with h5py.File(h5_path, "r") as f:
            n_views = f["c2w"].shape[0]

        scene_psnrs, scene_lpips = [], []
        for v in range(n_views):
            c2w = c2w_all[v:v + 1].unsqueeze(0)
            fov = fov_all[v:v + 1].unsqueeze(0)
            with torch.no_grad():
                out = pipeline(
                    gaussians=gaussians, mask=mask, c2w=c2w, fov=fov,
                    resolution=args.resolution, torch_dtype=torch.bfloat16,
                )
            hdr = out[0, 0].cpu().float().numpy()
            gf_ldr = (tone_mapper.hdr_to_ldr(hdr) if tone_mapper is not None
                      else np.clip(hdr, 0, 1))
            gf_ldr = np.clip(gf_ldr, 0, 1).astype(np.float32)

            gt_path = args.gt_dir / f"{h5_path.stem}_view_{v}.png"
            if not gt_path.exists():
                continue
            gt = load_gt(gt_path, args.resolution)

            mse = float(((gf_ldr - gt) ** 2).mean())
            psnr = 10.0 * np.log10(1.0 / (mse + 1e-12))

            pred_t = torch.from_numpy(gf_ldr).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
            gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
            with torch.no_grad():
                lp = float(lpips_fn(pred_t, gt_t).item())

            scene_psnrs.append(psnr)
            scene_lpips.append(lp)

        if not scene_psnrs:
            continue
        psnrs = np.asarray(scene_psnrs)
        lps = np.asarray(scene_lpips)
        per_scene.append({
            "name": h5_path.stem,
            "n_views": int(len(scene_psnrs)),
            "psnr_mean": float(psnrs.mean()), "psnr_min": float(psnrs.min()),
            "psnr_max": float(psnrs.max()),
            "lpips_mean": float(lps.mean()), "lpips_min": float(lps.min()),
            "lpips_max": float(lps.max()),
        })

        if (si + 1) % 10 == 0 or si == len(scene_h5s) - 1:
            elapsed = time.time() - t_start
            rate = (si + 1) / elapsed
            eta = (len(scene_h5s) - si - 1) / rate if rate > 0 else 0
            print(f"  [{si+1}/{len(scene_h5s)}] {h5_path.stem}: "
                  f"PSNR {psnrs.mean():.2f} LPIPS {lps.mean():.3f} "
                  f"({rate:.2f} scene/s, ETA {eta/60:.1f}m)")

    all_psnr = np.asarray([s["psnr_mean"] for s in per_scene])
    all_lpips = np.asarray([s["lpips_mean"] for s in per_scene])
    summary = {
        "n_scenes": int(len(per_scene)),
        "n_views_total": int(sum(s["n_views"] for s in per_scene)),
        "psnr_mean": float(all_psnr.mean()), "psnr_median": float(np.median(all_psnr)),
        "psnr_min": float(all_psnr.min()), "psnr_max": float(all_psnr.max()),
        "psnr_std": float(all_psnr.std()),
        "lpips_mean": float(all_lpips.mean()), "lpips_median": float(np.median(all_lpips)),
        "lpips_min": float(all_lpips.min()), "lpips_max": float(all_lpips.max()),
        "lpips_std": float(all_lpips.std()),
    }

    result = {
        "checkpoint": str(args.checkpoint),
        "epoch": epoch,
        "pe_type": args.pe_type,
        "h5_dir": str(args.h5_dir),
        "gt_dir": str(args.gt_dir),
        "resolution": args.resolution,
        "tone_mapper": args.tone_mapper,
        "wall_time_s": time.time() - t_start,
        "summary": summary,
        "per_scene": per_scene,
    }
    args.out_json.write_text(json.dumps(result, indent=2))

    print(f"\n== SUMMARY ({args.checkpoint.name}, pe_type={args.pe_type}) ==")
    print(f"  scenes:     {summary['n_scenes']}  ({summary['n_views_total']} views)")
    print(f"  PSNR  mean: {summary['psnr_mean']:.3f} dB  median: {summary['psnr_median']:.3f}  "
          f"std: {summary['psnr_std']:.3f}  range: [{summary['psnr_min']:.2f}, {summary['psnr_max']:.2f}]")
    print(f"  LPIPS mean: {summary['lpips_mean']:.4f}  median: {summary['lpips_median']:.4f}  "
          f"std: {summary['lpips_std']:.4f}  range: [{summary['lpips_min']:.4f}, {summary['lpips_max']:.4f}]")
    print(f"  wrote: {args.out_json}")


if __name__ == "__main__":
    main()
