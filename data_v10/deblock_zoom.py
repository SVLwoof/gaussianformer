"""Zoomed, native-resolution comparison for the patch-grid artefact.

For each (scene, view) pick the 96x96 window with the most GT texture (gradient energy), and show
GT | rasterizer | model A | model B | |A-GT| | |B-GT|, each upscaled with NEAREST so an 8-px grid
stays visible (any smoothing resize would hide exactly what we are looking for). Error maps share
one colour scale per row so A and B are directly comparable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from data_external.orbit import make_orbit_views
from render_compare import load_model, ModelSpec, load_gt
from infer_gaussian import load_single_gaussian_h5_data
from data_v10.ceiling_eval import SPLITS, RADIUS, FOV, RES, psnr, render_rec_all, render_model_all

WIN, UP = 96, 4
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def best_window(gt: np.ndarray) -> tuple[int, int]:
    lum = gt.mean(-1)
    g = np.abs(np.diff(lum, axis=0))[:, :-1] + np.abs(np.diff(lum, axis=1))[:-1, :]
    ii = np.pad(g, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    best, pos = -1.0, (0, 0)
    for y in range(0, g.shape[0] - WIN, 8):
        for x in range(0, g.shape[1] - WIN, 8):
            s = ii[y + WIN, x + WIN] - ii[y, x + WIN] - ii[y + WIN, x] + ii[y, x]
            if lum[y:y + WIN, x:x + WIN].mean() > 0.08 and s > best:
                best, pos = s, (y, x)
    return pos


def tile(img: np.ndarray, label: str, font) -> Image.Image:
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((WIN * UP, WIN * UP), Image.NEAREST)
    canvas = Image.new("RGB", (WIN * UP, WIN * UP + 26), (0, 0, 0))
    canvas.paste(im, (0, 26))
    ImageDraw.Draw(canvas).text((4, 3), label, fill=(255, 230, 90), font=font)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs=2, required=True, help="label:ckpt:model_cfg (space-separated key=val)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--scenes", nargs="+", type=int, required=True)
    ap.add_argument("--views", nargs="+", type=int, default=[0, 7])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    device = torch.device("cuda")
    font = ImageFont.truetype(FONT, 17)
    specs = []
    for m in a.models:
        lbl, ck, cfg = m.split(":", 2)
        specs.append((lbl, load_model(ModelSpec(ckpt=Path(ck), label=lbl, pe_type="rope", model_cfg=cfg.split() or None), device)))
    h5_dir, ren_dir = SPLITS[a.split]
    vm_np, K_np = make_orbit_views(14, RADIUS, FOV, RES, up_axis="y")
    vm, K = torch.from_numpy(vm_np).to(device), torch.from_numpy(K_np).to(device)
    rows, stats = [], []
    for s in a.scenes:
        h5 = h5_dir / f"scene_{s:04d}.h5"
        data = load_single_gaussian_h5_data(h5)
        for k in ("gaussians", "mask", "c2w", "fov"):
            data[k] = data[k].to(device)
        rec = render_rec_all(h5, a.views, vm, K, RES, device)
        outs = [render_model_all(p, data, a.views, RES, 7) for _, p in specs]
        for i, v in enumerate(a.views):
            gt = load_gt(ren_dir / f"scene_{s:04d}_view_{v}.png", RES)
            y, x = best_window(gt)
            c = lambda im: im[y:y + WIN, x:x + WIN]
            g = c(gt)
            ea, eb = np.abs(c(outs[0][i]) - g), np.abs(c(outs[1][i]) - g)
            scale = max(ea.max(), eb.max(), 1e-6)
            pa, pb, pr = psnr(c(outs[0][i]), g), psnr(c(outs[1][i]), g), psnr(c(rec[i]), g)
            stats.append({"scene": s, "view": v, "crop_psnr_rec": pr, specs[0][0]: pa, specs[1][0]: pb})
            tiles = [tile(g, f"GT s{s} v{v}", font), tile(c(rec[i]), f"raster {pr:.1f}", font),
                     tile(c(outs[0][i]), f"{specs[0][0]} {pa:.1f}", font), tile(c(outs[1][i]), f"{specs[1][0]} {pb:.1f}", font),
                     tile(ea / scale, f"|{specs[0][0]}-GT|", font), tile(eb / scale, f"|{specs[1][0]}-GT|", font)]
            row = Image.new("RGB", (sum(t.width for t in tiles) + 4 * (len(tiles) - 1), tiles[0].height), (40, 40, 40))
            xx = 0
            for t in tiles:
                row.paste(t, (xx, 0)); xx += t.width + 4
            rows.append(row)
    sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows) + 4 * (len(rows) - 1)), (40, 40, 40))
    yy = 0
    for r in rows:
        sheet.paste(r, (0, yy)); yy += r.height + 4
    a.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.out)
    a.out.with_suffix(".json").write_text(json.dumps(stats, indent=1))
    print("DEBLOCK_ZOOM wrote", a.out, sheet.size)


if __name__ == "__main__":
    main()
