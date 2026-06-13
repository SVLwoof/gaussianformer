"""Novel-view generalisation probe for the overfit models.

The overfit trained on the object's 14 orbit views. This renders the model at HELD-OUT
poses -- azimuths halfway between training views, at an in-between elevation -- that it never
saw, and compares to gsplat-full (real-GT) and gsplat-pruned (20k) at the SAME pose. It
separates "memorised the 14 training views" (novel views collapse) from "learned a renderable
3D representation from the 20k input" (novel views stay sharp). Also renders one training view
as a control (should reproduce the ~50 dB overfit number).

  uv run --frozen python -m experiments.overfit.novel_view \
    --ckpt experiments/overfit/ckpt/tomatoes_v14best/phase2_epoch_1500.pt \
    --scene tomatoes --h5 experiments/overfit/data/tomatoes/h5s/tomatoes_n20000.h5 \
    --label tomatoes_v14best --out_dir experiments/overfit/eval
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
import h5py
import gsplat
import imageio.v3 as iio
import lpips as lpips_lib
from PIL import ImageFont

from data_external.orbit import look_at_blender, c2w_to_viewmat
from data_external.scene_configs import SCENES
from data_external.run_scene import normalize_raw
from render_compare import load_model, ModelSpec, label_panel, compose_row, compose_grid
from infer_gaussian import load_single_gaussian_h5_data

RADIUS, FOV_DEG, N_TRAIN = 1.7, 45.0, 14


def psnr(a, b):
    return 10.0 * np.log10(1.0 / (float(((a - b) ** 2).mean()) + 1e-12))


def K_from_fov(res):
    f = 0.5 * res / np.tan(0.5 * FOV_DEG * np.pi / 180.0)
    return np.array([[f, 0, res / 2], [0, f, res / 2], [0, 0, 1]], dtype=np.float32)


def rasterize(arr, viewmat, K, res, device):
    op = arr["opacities"]
    op = op.squeeze(-1) if op.ndim == 2 else op
    with torch.no_grad():
        img, _, _ = gsplat.rasterization(
            means=arr["means"], quats=arr["quats"], scales=arr["scales"],
            opacities=op, colors=arr["colors"],
            viewmats=viewmat[None], Ks=K[None], width=res, height=res,
            sh_degree=None, eps2d=0.3, render_mode="RGB", near_plane=0.01, packed=True,
        )
    return img[0].clamp(0, 1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--scene", type=str, default="tomatoes")
    ap.add_argument("--h5", type=Path, required=True)
    ap.add_argument("--label", type=str, default="model")
    ap.add_argument("--pe_type", type=str, default="rope")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = args.resolution

    # full splat (real-GT) in the H5's normalised frame
    cfg = SCENES[args.scene]
    full_np = normalize_raw(cfg)
    full = {k: torch.from_numpy(full_np[k]).to(device) for k in ("means", "scales", "quats", "colors", "opacities")}
    # pruned 20k (model input + pruned-GT)
    with h5py.File(args.h5, "r") as f:
        pruned = {
            "means": torch.from_numpy(np.array(f["means"], np.float32)).to(device),
            "scales": torch.from_numpy(np.array(f["scales"], np.float32)).to(device),
            "quats": torch.from_numpy(np.array(f["rotations"], np.float32)).to(device),
            "colors": torch.from_numpy(np.array(f["colors"], np.float32)).to(device),
            "opacities": torch.from_numpy(np.array(f["opacities"], np.float32)).to(device),
        }
    data = load_single_gaussian_h5_data(args.h5)
    gaussians = data["gaussians"].unsqueeze(0).to(device)
    mask = data["mask"].unsqueeze(0).to(device)
    pipe = load_model(ModelSpec(ckpt=args.ckpt, label=args.label, pe_type=args.pe_type), device)
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()

    def lp(a, b):
        ta = torch.from_numpy(a).permute(2, 0, 1)[None].to(device) * 2 - 1
        tb = torch.from_numpy(b).permute(2, 0, 1)[None].to(device) * 2 - 1
        with torch.no_grad():
            return float(lpips_fn(ta, tb).item())

    K = torch.from_numpy(K_from_fov(res)).to(device)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def eye(theta, elev):
        return np.array([RADIUS * np.cos(theta), elev, RADIUS * np.sin(theta)], dtype=np.float32)

    # view set: 2 training controls (exact training poses) + 6 held-out novel poses
    # (azimuth halfway between training views, elevation between the alt-high/low training elevs)
    views = []
    views.append(("train_v0", 2 * np.pi * 0 / N_TRAIN, 0.4 * RADIUS))     # == H5 view 0
    views.append(("train_v1", 2 * np.pi * 1 / N_TRAIN, -0.1 * RADIUS))    # == H5 view 1
    for j in (0, 2, 4, 6, 8, 10):
        views.append((f"novel_a{j}", 2 * np.pi * (j + 0.5) / N_TRAIN, 0.15 * RADIUS))

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    per, rows = [], []
    for name, theta, elev in views:
        c2w_np = look_at_blender(eye(theta, elev), np.zeros(3), up=up)
        viewmat = torch.from_numpy(c2w_to_viewmat(c2w_np)).to(device)
        c2w_t = torch.from_numpy(c2w_np).to(device).unsqueeze(0).unsqueeze(0)   # [1,1,4,4]
        fov_t = torch.tensor([[FOV_DEG]], dtype=torch.float32, device=device)
        with torch.no_grad():
            out = pipe(gaussians=gaussians, mask=mask, c2w=c2w_t, fov=fov_t,
                       resolution=res, torch_dtype=torch.bfloat16)
        model = np.clip(out[0, 0].cpu().float().numpy(), 0, 1).astype(np.float32)
        real = rasterize(full, viewmat, K, res, device)
        prun = rasterize(pruned, viewmat, K, res, device)
        m = dict(view=name, kind=("train" if name.startswith("train") else "novel"),
                 psnr_model_real=psnr(model, real), lpips_model_real=lp(model, real),
                 psnr_model_pruned=psnr(model, prun),
                 psnr_pruned_real=psnr(prun, real))
        per.append(m)
        rows.append(compose_row([
            label_panel(real, f"real-GT {name}", font),
            label_panel(prun, f"pruned-GT  vsReal {m['psnr_pruned_real']:.1f}", font),
            label_panel(model, f"model  vsReal {m['psnr_model_real']:.1f}  vsPruned {m['psnr_model_pruned']:.1f}", font),
        ]))
        print(f"  {name:10s} [{m['kind']:5s}]  model-vs-real {m['psnr_model_real']:6.2f} dB / LPIPS {m['lpips_model_real']:.4f}"
              f"   (pruned-vs-real {m['psnr_pruned_real']:.2f})")

    train = [d for d in per if d["kind"] == "train"]
    novel = [d for d in per if d["kind"] == "novel"]
    summ = {
        "train_model_vs_real_psnr": float(np.mean([d["psnr_model_real"] for d in train])),
        "novel_model_vs_real_psnr": float(np.mean([d["psnr_model_real"] for d in novel])),
        "novel_model_vs_real_lpips": float(np.mean([d["lpips_model_real"] for d in novel])),
        "novel_pruned_vs_real_psnr": float(np.mean([d["psnr_pruned_real"] for d in novel])),
    }
    out = dict(ckpt=str(args.ckpt), label=args.label, summary=summ, per_view=per)
    (args.out_dir / f"{args.label}_novelview_metrics.json").write_text(json.dumps(out, indent=2))
    iio.imwrite(args.out_dir / f"{args.label}_novelview_strip.png",
                (np.clip(compose_grid(rows), 0, 1) * 255).astype(np.uint8))
    print(f"\n=== {args.label} ===")
    print(f"  TRAIN views  model-vs-real: {summ['train_model_vs_real_psnr']:.2f} dB  (memorised control)")
    print(f"  NOVEL views  model-vs-real: {summ['novel_model_vs_real_psnr']:.2f} dB / LPIPS {summ['novel_model_vs_real_lpips']:.4f}")
    print(f"  NOVEL views  pruned-vs-real: {summ['novel_pruned_vs_real_psnr']:.2f} dB  (what a perfect 20k renderer gets)")
    print(f"  wrote {args.out_dir}/{args.label}_novelview_(metrics.json|strip.png)")


if __name__ == "__main__":
    main()
