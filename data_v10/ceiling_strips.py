"""Render GT | rec-GT | V17 strips for the peculiar cases ceiling_report surfaces.

The numbers say where V17 sits relative to the recovered-input ceiling; only looking at the pixels
says WHY. In particular, cases where V17 scores at or above rec-GT need eyes on them before they
get quoted -- the likely mechanism is that rec-GT carries pruning artefacts (floaters, dropped
detail) that V17 smooths away, which would mean rec-GT bounds the available INFORMATION but not
per-pixel agreement with the GT render.

Panels are foreground-cropped with the same `_fg_crop` used for the metrics, so what is shown is
what was scored.

  uv run --frozen python -m data_v10.ceiling_strips --cases data_v10/ceiling/cases.json \
    --out meeting_material/ceiling_cases
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import imageio.v3 as iio
import lpips as lpips_lib
from PIL import Image, ImageDraw, ImageFont

from data_external.orbit import make_orbit_views
from render_compare import load_model, ModelSpec, load_gt
from infer_gaussian import load_single_gaussian_h5_data
from data_v10.model_on_v10 import _fg_crop
from data_v10.ceiling_eval import SPLITS, RADIUS, FOV, RES, psnr, render_rec_all, render_model_all

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def band(img: np.ndarray, text: str, font, colour=(255, 230, 0)) -> np.ndarray:
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 30], fill=(0, 0, 0))
    d.text((5, 4), text, fill=colour, font=font)
    return np.asarray(im).astype(np.float32) / 255.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True,
                    help='JSON list of {"split":..,"scene":..,"view":..,"tag":..}')
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints_v17_512lp/phase2_epoch_36.pt"))
    ap.add_argument("--view_chunk", type=int, default=7)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    cases = json.loads(args.cases.read_text())
    vm_np, K_np = make_orbit_views(14, RADIUS, FOV, RES, up_axis="y")
    vm, K = torch.from_numpy(vm_np).to(device), torch.from_numpy(K_np).to(device)
    pipe = load_model(ModelSpec(ckpt=args.ckpt, label="V17", pe_type="rope"), device)
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
    font = ImageFont.truetype(FONT, 22)

    def lp(a, b):
        ta = torch.from_numpy(a).permute(2, 0, 1)[None].to(device) * 2 - 1
        tb = torch.from_numpy(b).permute(2, 0, 1)[None].to(device) * 2 - 1
        with torch.no_grad():
            return float(lpips_fn(ta, tb).item())

    # Group by (split, scene) so each object is encoded once even with several views requested.
    groups: dict[tuple[str, int], list[dict]] = {}
    for c in cases:
        groups.setdefault((c["split"], c["scene"]), []).append(c)

    rows, index = [], []
    for (split, scene), cs in groups.items():
        h5_dir, ren_dir = SPLITS[split]
        h5 = h5_dir / f"scene_{scene:04d}.h5"
        if not h5.exists():
            print(f"skip {split}/{scene}: no h5", flush=True)
            continue
        data = load_single_gaussian_h5_data(h5)
        for k in ("gaussians", "mask", "c2w", "fov"):
            data[k] = data[k].to(device)
        views = [c["view"] for c in cs]
        recs = render_rec_all(h5, views, vm, K, RES, device)
        mdls = render_model_all(pipe, data, views, RES, args.view_chunk)

        for j, c in enumerate(cs):
            gt = load_gt(ren_dir / f"scene_{scene:04d}_view_{c['view']}.png", RES)
            c_gt, c_rec, c_mdl = _fg_crop([gt, recs[j], mdls[j]], gt, pad=12, out=RES)
            p_rec, p_mdl = psnr(c_rec, c_gt), psnr(c_mdl, c_gt)
            l_rec, l_mdl = lp(c_rec, c_gt), lp(c_mdl, c_gt)
            tag = c.get("tag", "")
            rows.append(np.concatenate([
                band(c_gt, f"GT  {split} s{scene} v{c['view']}  {tag}", font, (255, 255, 255)),
                band(c_rec, f"rec-GT  {p_rec:.1f}dB  LPIPS {l_rec:.4f}", font),
                band(c_mdl, f"V17  {p_mdl:.1f}dB  LPIPS {l_mdl:.4f}", font,
                     (120, 255, 160) if p_mdl >= p_rec else (255, 230, 0)),
            ], axis=1))
            index.append(dict(split=split, scene=scene, view=c["view"], tag=tag,
                              rec_psnr=round(p_rec, 3), mdl_psnr=round(p_mdl, 3),
                              rec_lpips=round(l_rec, 5), mdl_lpips=round(l_mdl, 5)))
            print(f"{split} s{scene} v{c['view']} {tag}: rec-GT {p_rec:.2f} / V17 {p_mdl:.2f} "
                  f"({p_rec - p_mdl:+.2f} dB)", flush=True)

    if rows:
        grid = (np.clip(np.concatenate(rows, axis=0), 0, 1) * 255).astype(np.uint8)
        iio.imwrite(args.out / "ceiling_cases.png", grid)
        (args.out / "ceiling_cases.json").write_text(json.dumps(index, indent=1))
        print(f"wrote {args.out}/ceiling_cases.png ({grid.shape[1]}x{grid.shape[0]})", flush=True)
    print("DONE_STRIPS", flush=True)


if __name__ == "__main__":
    main()
