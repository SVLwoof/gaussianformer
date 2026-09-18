"""Render sweep for the 2026-09-17 supervisor report (many objects, several models).

Modes:
  ladder  heldout scenes x views {0,7}: GT | rec-GT | V18 seed | P2+fg c3 | canvas c4, FG-cropped,
          one PNG per scene + a JSON of crop PSNRs; also a contact sheet (view 0, rec-GT | canvas c4)
          over --sheet_scenes.
  codec   one scale-out object (gopro / scene_1423): its held-out codec views through
          GT | rasterizer | canvas-c3 base (no adapter) | adapter, full frame + a 96-px zoom row
          per view (best-texture window, NEAREST x4, error maps on a shared scale).

  PYTHONPATH=. uv run --no-sync python data_v10/report_renders.py ladder --scenes 7 23 ... --out docs/report/renders
  PYTHONPATH=. uv run --no-sync python data_v10/report_renders.py codec --scene gopro --out docs/report/renders
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from data_external.orbit import c2w_to_viewmat, make_orbit_views, orbit_c2w
from data_v10.ceiling_eval import FOV, RADIUS, RES, SPLITS, psnr, render_model_all, render_rec_all
from data_v10.model_on_v10 import _fg_crop
from data_v10.prune_recovery import rasterize
from infer_gaussian import load_single_gaussian_h5_data
from render_compare import ModelSpec, load_gt, load_model

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
WIN, UP = 96, 4  # zoom window (px) and nearest-neighbour upscale


def best_window(gt: np.ndarray) -> tuple[int, int]:
    """Top-left of the WIN x WIN window with the most GT gradient energy (and some brightness)."""
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
CANVAS = "canvas_cond=true proj_rope_2d=true"
LADDER = [  # label, ckpt, model_cfg
    ("V18 seed", "checkpoints_v18_256/phase2_epoch_30.pt", ""),
    ("P2+fg c3", "checkpoints_probe_p2r_fg_c3/phase2_epoch_3000.pt", "proj_rope_2d=true"),
    ("canvas c4", "checkpoints_probe_p1p2_fg_c4/phase2_epoch_3000.pt", CANVAS),
]
CODEC = {  # scene -> (base ckpt, adapter ckpt)
    "gopro": ("checkpoints_probe_p1p2_fg_c3/phase2_epoch_3000.pt", "checkpoints_lora_p1p2fgc3_gopro_r4/lora_final.pt"),
    "scene_1423": ("checkpoints_probe_p1p2_fg_c3/phase2_epoch_3000.pt", "checkpoints_lora_p1p2fgc3_scene_1423_r4/lora_final.pt"),
}


def band(img: np.ndarray, text: str, font, colour=(255, 230, 0), size: int = 384) -> Image.Image:
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    if im.width != size:
        im = im.resize((size, size), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, size, 24], fill=(0, 0, 0))
    d.text((4, 3), text, fill=colour, font=font)
    return im


def hstack(tiles: list[Image.Image], gap: int = 4) -> Image.Image:
    w = sum(t.width for t in tiles) + gap * (len(tiles) - 1)
    out = Image.new("RGB", (w, max(t.height for t in tiles)), (40, 40, 40))
    x = 0
    for t in tiles:
        out.paste(t, (x, 0))
        x += t.width + gap
    return out


def vstack(rows: list[Image.Image], gap: int = 4) -> Image.Image:
    h = sum(r.height for r in rows) + gap * (len(rows) - 1)
    out = Image.new("RGB", (max(r.width for r in rows), h), (40, 40, 40))
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height + gap
    return out


def zoom_tile(img: np.ndarray, label: str, font) -> Image.Image:
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((WIN * UP, WIN * UP), Image.NEAREST)
    out = Image.new("RGB", (WIN * UP, WIN * UP + 26), (0, 0, 0))
    out.paste(im, (0, 26))
    ImageDraw.Draw(out).text((4, 3), label, fill=(255, 230, 90), font=font)
    return out


def zoom_row(gt: np.ndarray, others: list[tuple[str, np.ndarray]], font, tag: str) -> tuple[Image.Image, dict]:
    y, x = best_window(gt)
    c = lambda im: im[y:y + WIN, x:x + WIN]
    g = c(gt)
    errs = [np.abs(c(im) - g) for _, im in others]
    scale = max(max(e.max() for e in errs), 1e-6)
    ps = {lab: psnr(c(im), g) for lab, im in others}
    tiles = [zoom_tile(g, f"GT {tag}", font)] + [zoom_tile(c(im), f"{lab} {ps[lab]:.1f}", font) for lab, im in others]
    tiles += [zoom_tile(e / scale, f"|{lab}-GT|", font) for (lab, _), e in zip(others[1:], errs[1:])]
    return hstack(tiles), ps


def ladder(a: argparse.Namespace, device) -> None:
    global RES, RADIUS, LADDER
    RES, RADIUS = a.res, a.radius
    if a.models:  # label:ckpt:cfg (cfg space-separated key=val, may be empty)
        LADDER = [tuple(m.split(":", 2)) for m in a.models]
    font, small = ImageFont.truetype(FONT, 17), ImageFont.truetype(FONT, 12)
    h5_dir, ren_dir = SPLITS["val"]
    if a.renders_dir is not None:
        ren_dir = a.renders_dir
    vm_np, K_np = make_orbit_views(14, RADIUS, FOV, RES, up_axis="y")
    vm, K = torch.from_numpy(vm_np).to(device), torch.from_numpy(K_np).to(device)
    scenes = sorted(set(a.scenes) | set(a.sheet_scenes))
    views = sorted(set(a.views) | {0})
    renders: dict[str, dict[tuple[int, int], np.ndarray]] = {lab: {} for lab, _, _ in LADDER}
    for lab, ck, cfg in LADDER:
        pipe = load_model(ModelSpec(ckpt=Path(ck), label=lab, pe_type="rope", model_cfg=cfg.split() or None), device)
        for s in scenes:
            data = load_single_gaussian_h5_data(h5_dir / f"scene_{s:04d}.h5")
            for k in ("gaussians", "mask", "c2w", "fov"):
                data[k] = data[k].to(device)
            if RADIUS != 1.7:  # the H5's cameras are the r=1.7 orbit; re-aim to the requested radius
                data["c2w"] = torch.from_numpy(orbit_c2w(14, RADIUS)).to(device, data["c2w"].dtype)
            vs = views if s in a.scenes else [0]
            out = render_model_all(pipe, data, vs, RES, 7)
            for j, v in enumerate(vs):
                renders[lab][(s, v)] = out[j]
        del pipe
        torch.cuda.empty_cache()
        print("rendered", lab, flush=True)
    stats, sheet_rows, sheet_cells = [], [], []
    for s in scenes:
        vs = views if s in a.scenes else [0]
        recs = render_rec_all(h5_dir / f"scene_{s:04d}.h5", vs, vm, K, RES, device)
        rows, zrows = [], []
        for j, v in enumerate(vs):
            gt = load_gt(ren_dir / f"scene_{s:04d}_view_{v}.png", RES)
            imgs = [gt, recs[j]] + [renders[lab][(s, v)] for lab, _, _ in LADDER]
            crops = _fg_crop(imgs, gt, pad=12, out=RES)
            p_rec = psnr(crops[1], crops[0])
            rec = {"scene": s, "view": v, "rec": p_rec}
            if s in a.scenes and v in a.views:
                panels = [band(crops[0], f"GT  s{s} v{v}", font, (255, 255, 255), a.panel),
                          band(crops[1], f"rasterizer  {p_rec:.1f} dB", font, (255, 230, 0), a.panel)]
                for (lab, _, _), c in zip(LADDER, crops[2:]):
                    p = psnr(c, crops[0])
                    rec[lab] = p
                    panels.append(band(c, f"{lab}  {p:.1f} dB ({p - p_rec:+.1f})", font,
                                       (120, 255, 160) if p >= p_rec else (255, 160, 120), a.panel))
                rows.append(hstack(panels))
                zr, zps = zoom_row(gt, [("raster", recs[j])] + [(lab, renders[lab][(s, v)]) for lab, _, _ in LADDER],
                                   font, f"s{s} v{v}")
                zrows.append(zr)
                rec["zoom"] = zps
            if v == 0 and s in a.sheet_scenes:
                p_c4 = psnr(crops[-1], crops[0])
                rec.setdefault("canvas c4", p_c4)
                sheet_cells.append(hstack([band(crops[1], f"s{s} raster {p_rec:.1f}", small, (255, 230, 0), a.sheet_panel),
                                           band(crops[-1], f"canvas c4 {p_c4:.1f}", small, (255, 160, 120), a.sheet_panel)], 2))
            stats.append(rec)
            print(json.dumps(rec), flush=True)
        if rows:
            vstack(rows).save(a.out / f"ladder_s{s:04d}.png")
            vstack(zrows).save(a.out / f"zoom_s{s:04d}.png")
    if sheet_cells:
        per = a.sheet_cols
        sheet = vstack([hstack(sheet_cells[i:i + per], 6) for i in range(0, len(sheet_cells), per)], 6)
        sheet.save(a.out / "contact_sheet.png")
    (a.out / "ladder.json").write_text(json.dumps(stats, indent=1))
    print("LADDER_DONE", flush=True)


def codec(a: argparse.Namespace, device) -> None:
    font = ImageFont.truetype(FONT, 20)
    base_ck, ada_ck = CODEC[a.scene]
    D = Path(f"experiments/overfit/data/codec_scaleout/{a.scene}")
    h5 = D / "h5s" / f"{a.scene}.h5"
    data = load_single_gaussian_h5_data(h5)
    with h5py.File(h5, "r") as f:
        rec = {k: torch.as_tensor(np.array(f[k], np.float32), device=device) for k in ("means", "scales", "rotations", "colors", "opacities")}
    recp = dict(means=rec["means"], quats=rec["rotations"], scales=rec["scales"], colors=rec["colors"], opacities=rec["opacities"].reshape(-1))
    focal = 0.5 * RES / np.tan(0.5 * np.radians(FOV))
    K = torch.as_tensor(np.array([[focal, 0, RES / 2], [0, focal, RES / 2], [0, 0, 1]], np.float32), device=device)
    models = [("base c3", load_model(ModelSpec(ckpt=Path(base_ck), label="base", pe_type="rope", model_cfg=CANVAS.split()), device)),
              ("adapter r4", load_model(ModelSpec(ckpt=Path(ada_ck), label="adapter", pe_type="rope"), device))]
    stats = []
    for name, n in (("novel_close", a.n_close), ("novel_rand", a.n_rand), ("novel_far", a.n_far)):
        c2ws = np.load(D / f"codec_eval_{name}_c2w.npy")
        vm = torch.as_tensor(np.stack([c2w_to_viewmat(c) for c in c2ws]), device=device)
        Ks = K[None].expand(len(c2ws), 3, 3)
        rows, zrows = [], []
        for i in range(min(n, len(c2ws))):
            gt = load_gt(D / "renders" / f"codec_eval_{name}" / f"gt_view_{i}.png", RES)
            rg = rasterize(recp, vm, Ks, [i], RES)[0].clamp(0, 1).cpu().numpy()
            outs = []
            for lab, pipe in models:
                c2w_t = torch.as_tensor(c2ws[i], device=device)[None]
                with torch.no_grad():
                    o = pipe(gaussians=data["gaussians"][None].to(device), mask=data["mask"][None].to(device),
                             c2w=c2w_t[None], fov=torch.tensor([[FOV]], device=device), resolution=RES, torch_dtype=torch.bfloat16)
                outs.append((lab, np.clip(o[0, 0].cpu().float().numpy(), 0, 1)))
            pr = psnr(rg, gt)
            rec_ = {"scene": a.scene, "range": name, "view": i, "rec": pr}
            panels = [band(gt, f"GT {name} v{i}", font, (255, 255, 255), RES), band(rg, f"rasterizer {pr:.1f}", font, (255, 230, 0), RES)]
            for lab, im in outs:
                p = psnr(im, gt)
                rec_[lab] = p
                panels.append(band(im, f"{lab} {p:.1f} ({p - pr:+.1f})", font, (120, 255, 160) if p >= pr else (255, 160, 120), RES))
            rows.append(hstack(panels))
            zr, zps = zoom_row(gt, [("raster", rg)] + outs, font, f"{name} v{i}")
            zrows.append(zr)
            rec_["zoom"] = zps
            stats.append(rec_)
            print(json.dumps(rec_), flush=True)
        vstack(rows).save(a.out / f"codec_{a.scene}_{name}.png")
        vstack(zrows).save(a.out / f"codec_{a.scene}_{name}_zoom.png")
    (a.out / f"codec_{a.scene}.json").write_text(json.dumps(stats, indent=1))
    print("CODEC_DONE", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    l = sub.add_parser("ladder")
    l.add_argument("--scenes", nargs="+", type=int, required=True)
    l.add_argument("--sheet_scenes", nargs="*", type=int, default=[])
    l.add_argument("--views", nargs="+", type=int, default=[0, 7])
    l.add_argument("--panel", type=int, default=384)
    l.add_argument("--sheet_panel", type=int, default=192)
    l.add_argument("--sheet_cols", type=int, default=4)
    l.add_argument("--out", type=Path, required=True)
    l.add_argument("--res", type=int, default=RES, help="render/eval resolution (GT resized)")
    l.add_argument("--radius", type=float, default=RADIUS, help="orbit radius; pair with --renders_dir of GT at that radius")
    l.add_argument("--renders_dir", type=Path, default=None)
    l.add_argument("--models", nargs="*", default=None, help="override the ladder: label:ckpt:cfg")
    c = sub.add_parser("codec")
    c.add_argument("--scene", choices=sorted(CODEC), required=True)
    c.add_argument("--n_close", type=int, default=8)
    c.add_argument("--n_rand", type=int, default=6)
    c.add_argument("--n_far", type=int, default=4)
    c.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    (ladder if a.mode == "ladder" else codec)(a, device)


if __name__ == "__main__":
    main()
