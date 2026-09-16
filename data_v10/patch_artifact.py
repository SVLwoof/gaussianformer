"""Patch-grid artefact strength: error power at period = patch_size vs its spectral neighbours.

For each (scene, view): e = model - GT on the object's bounding box at NATIVE resolution
(no resize: resizing would move the period). Crop width/height are floored to a multiple of
the patch size so the patch frequency lands exactly on an FFT bin. Row-averaged |rfft| gives
the x spectrum, column-averaged the y spectrum. Score = power at the patch bin / mean power at
the bins +-2 and +-3 away (skipping +-1, which leaks into the peak).
  score ~ 1  -> no grid;  score >> 1 -> the token grid is imprinted on the output.
The rasterizer (rec-GT) is scored too, as the no-grid reference.
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
from data_v10.ceiling_eval import SPLITS, RADIUS, FOV, RES, render_rec_all, render_model_all

PATCH = 8


def score(err: np.ndarray) -> tuple[float, float]:
    out = []
    for axis in (1, 0):                                   # x (along rows), then y
        n = (err.shape[axis] // PATCH) * PATCH
        e = err[:, :n] if axis == 1 else err[:n, :]
        spec = np.abs(np.fft.rfft(e, axis=axis)).mean(axis=1 - axis)
        k = n // PATCH
        nb = [spec[k + d] for d in (-3, -2, 2, 3) if 0 < k + d < len(spec)]
        out.append(float(spec[k] / (np.mean(nb) + 1e-12)))
    return out[0], out[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--model_cfg", nargs="*", default=None)
    ap.add_argument("--split", choices=sorted(SPLITS), required=True)
    ap.add_argument("--scenes_file", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--views", default="0,4,7,11")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    device = torch.device("cuda")
    views = [int(v) for v in a.views.split(",")]
    scenes = json.loads(a.scenes_file.read_text())[: a.limit or None]
    h5_dir, ren_dir = SPLITS[a.split]
    pipe = load_model(ModelSpec(ckpt=a.ckpt, label="m", pe_type="rope", model_cfg=a.model_cfg), device)
    vm_np, K_np = make_orbit_views(14, RADIUS, FOV, RES, up_axis="y")
    vm, K = torch.from_numpy(vm_np).to(device), torch.from_numpy(K_np).to(device)
    rows = []
    for s in scenes:
        h5 = h5_dir / f"scene_{s:04d}.h5"
        if not h5.exists():
            continue
        data = load_single_gaussian_h5_data(h5)
        for k in ("gaussians", "mask", "c2w", "fov"):
            data[k] = data[k].to(device)
        rec = render_rec_all(h5, views, vm, K, RES, device)
        mdl = render_model_all(pipe, data, views, RES, 7)
        for i, v in enumerate(views):
            p = ren_dir / f"scene_{s:04d}_view_{v}.png"
            if not p.exists():
                continue
            gt = load_gt(p, RES)
            ys, xs = np.where(gt.mean(-1) > 0.02)
            if len(ys) < 64:
                continue
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            g = gt[y0:y1, x0:x1].mean(-1)
            mx, my = score(mdl[i][y0:y1, x0:x1].mean(-1) - g)
            rx, ry = score(rec[i][y0:y1, x0:x1].mean(-1) - g)
            rows.append({"scene": s, "view": v, "model_x": mx, "model_y": my, "rec_x": rx, "rec_y": ry})
    agg = {k: float(np.median([r[k] for r in rows])) for k in ("model_x", "model_y", "rec_x", "rec_y")}
    agg["n"] = len(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"agg": agg, "ckpt": str(a.ckpt), "rows": rows}, indent=1))
    print("PATCH_ARTIFACT", a.ckpt, json.dumps(agg))


if __name__ == "__main__":
    main()
