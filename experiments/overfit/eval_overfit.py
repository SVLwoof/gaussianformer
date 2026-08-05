"""Three-way overfit decomposition for the single-object capacity probe.

    real-GT  --(pruning loss, fixed by N)-->  pruned-GT  --(model loss)-->  model output

For one checkpoint, render the model at all of the object's views and report PSNR/LPIPS vs
both the real-GT (the training target) and the pruned-GT (the N=20k information ceiling the
model's input can possibly reach), plus the fixed pruned-vs-real gap. Writes a metrics JSON
and a real-GT | pruned-GT | model strip montage.

The decisive number is **model vs pruned-GT**: if a fully-overfit model can't match the
rasterisation of its OWN 20k-Gaussian input on a single memorised object, the ceiling is the
architecture (decoder), not the data.

Reuses render_compare internals so the camera + gsplat path is identical to the training eval.

  uv run --frozen python -m experiments.overfit.eval_overfit \
    --ckpt experiments/overfit/ckpt/boxes_base/phase2_epoch_1500.pt \
    --h5 experiments/overfit/data/boxes/h5s/scene_1441.h5 \
    --realgt_dir data_v9/renders --label boxes_base \
    --out_dir experiments/overfit/eval
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
import h5py
import imageio.v3 as iio
import lpips as lpips_lib
from PIL import ImageFont

from render_compare import (
    ORBIT_N_VIEWS, ORBIT_RADIUS, ORBIT_FOV_DEG,
    render_pruned_gt, load_model, ModelSpec, render_view, load_gt,
    label_panel, compose_row, compose_grid, make_orbit_views,
    load_single_gaussian_h5_data,
)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return 10.0 * np.log10(1.0 / (float(((a - b) ** 2).mean()) + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--h5", type=Path, required=True, help="the single object's H5")
    ap.add_argument("--realgt_dir", type=Path, required=True,
                    help="dir containing <h5-stem>_view_<i>.png real-GT images")
    ap.add_argument("--label", type=str, default="model")
    ap.add_argument("--pe_type", type=str, default="rope")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n_strip", type=int, default=5, help="views shown in the strip montage")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stem = args.h5.stem

    pipe = load_model(ModelSpec(ckpt=args.ckpt, label=args.label, pe_type=args.pe_type), device)
    data = load_single_gaussian_h5_data(args.h5)
    for k in ("gaussians", "mask", "c2w", "fov"):
        data[k] = data[k].to(device)
    with h5py.File(args.h5, "r") as f:
        n_views = f["c2w"].shape[0]

    vm, ks = make_orbit_views(ORBIT_N_VIEWS, ORBIT_RADIUS, ORBIT_FOV_DEG, args.resolution, up_axis="y")
    vm = torch.from_numpy(vm).to(device)
    ks = torch.from_numpy(ks).to(device)

    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()

    def lp(a: np.ndarray, b: np.ndarray) -> float:
        ta = torch.from_numpy(a).permute(2, 0, 1)[None].to(device) * 2 - 1
        tb = torch.from_numpy(b).permute(2, 0, 1)[None].to(device) * 2 - 1
        with torch.no_grad():
            return float(lpips_fn(ta, tb).item())

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    per, rows = [], []
    for v in range(n_views):
        real = load_gt(args.realgt_dir / f"{stem}_view_{v}.png", args.resolution)
        pruned = render_pruned_gt(args.h5, v, vm, ks, args.resolution, device)
        model = render_view(pipe, data, v, args.resolution, None)   # tone_mapper=None (clip)
        m = dict(
            view=v,
            psnr_model_real=psnr(model, real), lpips_model_real=lp(model, real),
            psnr_model_pruned=psnr(model, pruned), lpips_model_pruned=lp(model, pruned),
            psnr_pruned_real=psnr(pruned, real), lpips_pruned_real=lp(pruned, real),
        )
        per.append(m)
        if v < args.n_strip:
            rows.append(compose_row([
                label_panel(real, f"real-GT v{v}", font),
                label_panel(pruned, f"pruned-GT  vsReal {m['psnr_pruned_real']:.2f}dB", font),
                label_panel(model, f"{args.label}  vsPruned {m['psnr_model_pruned']:.2f}  vsReal {m['psnr_model_real']:.2f}", font),
            ]))

    summ = {k: float(np.mean([d[k] for d in per])) for k in per[0] if k != "view"}
    out = dict(ckpt=str(args.ckpt), object=stem, n_views=n_views, summary=summ, per_view=per)
    (args.out_dir / f"{stem}_{args.label}_metrics.json").write_text(json.dumps(out, indent=2))
    iio.imwrite(args.out_dir / f"{stem}_{args.label}_strip.png",
                (np.clip(compose_grid(rows), 0, 1) * 255).astype(np.uint8))

    print(f"\n=== {stem}  [{args.label}]  ({args.ckpt.name}) ===")
    print(f"  model  vs real-GT  : {summ['psnr_model_real']:6.2f} dB / LPIPS {summ['lpips_model_real']:.4f}")
    print(f"  model  vs pruned-GT: {summ['psnr_model_pruned']:6.2f} dB / LPIPS {summ['lpips_model_pruned']:.4f}   <-- CAPACITY CEILING")
    print(f"  pruned vs real-GT  : {summ['psnr_pruned_real']:6.2f} dB / LPIPS {summ['lpips_pruned_real']:.4f}   (N=20k pruning loss)")
    print(f"  wrote {args.out_dir}/{stem}_{args.label}_(metrics.json|strip.png)")


if __name__ == "__main__":
    main()
