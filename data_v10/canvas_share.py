"""How much of a canvas-conditioned model's output IS the canvas?

For each (scene, view): PSNR(model, GT), PSNR(rec-GT, GT) and PSNR(model, rec-GT) on the
foreground crop, plus the relative residual energy ||model - rec||/||rec||. rec-GT here is the
gsplat rasterization of the SAME recovered Gaussians the model is fed, i.e. the canvas the
residual head adds to, so PSNR(model, rec-GT) answers "is this a rasterizer with extra steps?":
  high (say > 40 dB)  -> the output is the canvas; the network contributes nothing
  low                 -> the network departs from the canvas; whether that HELPS is the
                         sign of PSNR(model, GT) - PSNR(rec-GT, GT).

  uv run --no-sync python -m data_v10.canvas_share --ckpt CKPT --model_cfg k=v ... \
     --split train --scenes_file data_v10/nsweep/n10_scenes.json --out tmp/share_train.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data_external.orbit import make_orbit_views
from render_compare import load_model, ModelSpec, load_gt
from infer_gaussian import load_single_gaussian_h5_data
from data_v10.model_on_v10 import _fg_crop
from data_v10.ceiling_eval import SPLITS, RADIUS, FOV, RES, psnr, render_rec_all, render_model_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--model_cfg", nargs="*", default=None)
    ap.add_argument("--split", choices=sorted(SPLITS), required=True)
    ap.add_argument("--scenes_file", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--views", default="0,4,7,11")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    device = torch.device("cuda")
    views = [int(v) for v in a.views.split(",")]
    scenes = json.loads(a.scenes_file.read_text())
    if a.limit:
        scenes = scenes[:a.limit]
    h5_dir, ren_dir = SPLITS[a.split]
    pipe = load_model(ModelSpec(ckpt=a.ckpt, label="m", pe_type="rope", model_cfg=a.model_cfg), device)
    vm_np, K_np = make_orbit_views(14, RADIUS, FOV, RES, up_axis="y")
    vm, K = torch.from_numpy(vm_np).to(device), torch.from_numpy(K_np).to(device)

    rows = []
    for n, s in enumerate(scenes):
        h5 = h5_dir / f"scene_{s:04d}.h5"
        if not h5.exists():
            continue
        data = load_single_gaussian_h5_data(h5)
        for k in ("gaussians", "mask", "c2w", "fov"):
            data[k] = data[k].to(device)
        rec = render_rec_all(h5, views, vm, K, RES, device)
        mdl = render_model_all(pipe, data, views, RES, 7)
        for i, v in enumerate(views):
            gt_path = ren_dir / f"scene_{s:04d}_view_{v}.png"
            if not gt_path.exists():
                continue
            gt = load_gt(gt_path, RES)
            g, r, m = _fg_crop([gt, rec[i], mdl[i]], gt, pad=12, out=RES)
            rows.append({"scene": s, "view": v,
                         "psnr_model_gt": psnr(m, g), "psnr_rec_gt": psnr(r, g),
                         "psnr_model_rec": psnr(m, r),
                         "resid_rel": float(np.linalg.norm(m - r) / (np.linalg.norm(r) + 1e-12))})
        if n % 20 == 0:
            print(f"{n}/{len(scenes)}", flush=True)

    agg = {k: float(np.mean([r[k] for r in rows])) for k in
           ("psnr_model_gt", "psnr_rec_gt", "psnr_model_rec", "resid_rel")}
    agg["n_rows"] = len(rows)
    agg["margin_model_minus_rec"] = agg["psnr_model_gt"] - agg["psnr_rec_gt"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"agg": agg, "ckpt": str(a.ckpt), "split": a.split,
                                 "model_cfg": a.model_cfg, "rows": rows}, indent=1))
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
