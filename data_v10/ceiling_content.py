"""Stratify the ceiling sweep by CONTENT, to find where the 14.5 dB actually lives.

The sweep gives a mean margin per slice, but a mean cannot distinguish "uniformly 14.5 dB on every
object" from "10 dB on ordinary objects plus a small emissive population failing at 30 dB". The
tails suggested the latter (glowing chests at 20 dB against a 52 dB ceiling), and that difference
decides whether the HDR-loss defect is a headline or a footnote.

For every (split, scene, view) already measured, this recomputes the SAME foreground crop used for
the metrics (`_fg_crop`, imported so it cannot drift) and characterises the GT content in it:

  sat_frac   fraction of crop pixels at/near display saturation  -> emissive / HDR proxy
  mean_lum   mean luminance of the crop                          -> overall brightness
  hf_energy  mean gradient magnitude / mean luminance            -> relative fine-detail content
  edge_frac  fraction of pixels above a gradient threshold       -> texture coverage

Content stats are computed on the NATIVE crop, before the 512 NEAREST resize. The resize factor
depends on object size, so measuring gradients after it would confound texture with scale --
`fg_frac` is kept as the separate scale covariate.

CPU only; no GPU, no model. Reads GT PNGs.

  uv run --no-sync python -m data_v10.ceiling_content --out meeting_material/ceiling_content.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from render_compare import load_gt
from data_v10.model_on_v10 import _fg_crop
from data_v10.ceiling_eval import SPLITS, RES


def native_crop(gt: np.ndarray) -> np.ndarray:
    """The same box _fg_crop picks, but WITHOUT the resize to RES (see module docstring)."""
    lum = gt.mean(-1)
    ys, xs = np.where(lum > 0.02)
    if len(ys) < 10:
        return gt
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    half = max(y1 - y0, x1 - x0) // 2 + 12
    H = gt.shape[0]
    return gt[max(0, cy - half):min(H, cy + half), max(0, cx - half):min(H, cx + half)]


def content_of(key: tuple[str, int, int]) -> dict | None:
    split, scene, view = key
    _, ren_dir = SPLITS[split]
    p = ren_dir / f"scene_{scene:04d}_view_{view}.png"
    if not p.exists():
        return None
    c = native_crop(load_gt(p, RES))
    lum = c.mean(-1)
    if lum.size < 16:
        return None
    gy, gx = np.gradient(lum)
    grad = np.hypot(gx, gy)
    m = float(lum.mean()) + 1e-6
    return dict(split=split, scene=scene, view=view,
                sat_frac=float((lum > 0.98).mean()),
                mean_lum=float(lum.mean()),
                hf_energy=float(grad.mean() / m),
                edge_frac=float((grad > 0.05).mean()),
                crop_px=int(lum.shape[0]))


def describe_bins(rows: list[dict], key: str, nbins: int = 5) -> list[str]:
    """Mean margin per quantile bin of `key`, at OBJECT level."""
    vals = np.array([r[key] for r in rows])
    edges = np.unique(np.percentile(vals, np.linspace(0, 100, nbins + 1)))
    if len(edges) < 3:
        return [f"(all objects share one value of {key})"]
    idx = np.clip(np.digitize(vals, edges[1:-1]), 0, len(edges) - 2)
    out = [f"| {key} range | n | rec-GT dB | V17 dB | margin dB |", "|---|---|---|---|---|"]
    for b in range(len(edges) - 1):
        sel = [r for r, i in zip(rows, idx) if i == b]
        if not sel:
            continue
        out.append(f"| {edges[b]:.4f} – {edges[b+1]:.4f} | {len(sel)} | "
                   f"{np.mean([r['rec_psnr_fg'] for r in sel]):.2f} | "
                   f"{np.mean([r['mdl_psnr_fg'] for r in sel]):.2f} | "
                   f"**{np.mean([r['margin_psnr_fg'] for r in sel]):.2f}** |")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", type=Path, default=[Path("data_v10/ceiling")])
    ap.add_argument("--out", type=Path, default=Path("meeting_material/ceiling_content.md"))
    ap.add_argument("--model", default="v17")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--cache", type=Path, default=Path("data_v10/ceiling/content_stats.jsonl"))
    args = ap.parse_args()

    seen, rows = set(), []
    for d in args.dirs:
        for p in sorted(d.glob("v17_*.jsonl")):
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("model", "v17") != args.model:
                    continue
                k = (r["split"], r["scene"], r["view"])
                if k not in seen:
                    seen.add(k)
                    rows.append(r)
    print(f"{len(rows)} metric rows", flush=True)

    cached: dict[tuple[str, int, int], dict] = {}
    if args.cache.exists():
        for line in args.cache.read_text().splitlines():
            if line.strip():
                c = json.loads(line)
                cached[(c["split"], c["scene"], c["view"])] = c
        print(f"{len(cached)} content rows cached", flush=True)

    todo = [k for k in seen if k not in cached]
    if todo:
        print(f"computing content for {len(todo)} renders on {args.workers} workers...", flush=True)
        with Pool(args.workers) as pool, args.cache.open("a") as fh:
            for i, c in enumerate(pool.imap_unordered(content_of, todo, chunksize=64)):
                if c is not None:
                    cached[(c["split"], c["scene"], c["view"])] = c
                    fh.write(json.dumps(c) + "\n")
                if (i + 1) % 5000 == 0:
                    print(f"  {i+1}/{len(todo)}", flush=True)

    merged = []
    for r in rows:
        c = cached.get((r["split"], r["scene"], r["view"]))
        if c is not None:
            merged.append({**r, **{k: c[k] for k in
                                   ("sat_frac", "mean_lum", "hf_energy", "edge_frac", "crop_px")}})
    print(f"{len(merged)} merged rows", flush=True)

    # Object level: views of one object are not independent.
    acc: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in merged:
        acc[(r["split"], r["scene"])].append(r)
    num = ["rec_psnr_fg", "mdl_psnr_fg", "margin_psnr_fg", "rec_lpips_fg", "mdl_lpips_fg",
           "fg_frac", "sat_frac", "mean_lum", "hf_energy", "edge_frac", "crop_px"]
    objs = [{"split": k[0], "scene": k[1],
             **{f: float(np.mean([v[f] for v in vs])) for f in num}} for k, vs in acc.items()]

    L = ["# Where the 14.5 dB lives — content stratification", "",
         f"{len(merged)} renders / {len(objs)} objects. Object-level means throughout.", ""]

    for sp in ("val", "train", "unseen2x"):
        o = [x for x in objs if x["split"] == sp]
        if not o:
            continue
        L += [f"## {sp}  (n={len(o)} objects)", "",
              "### Correlation of margin with content", "",
              "| covariate | corr with margin |", "|---|---|"]
        m = np.array([x["margin_psnr_fg"] for x in o])
        for f in ("sat_frac", "mean_lum", "hf_energy", "edge_frac", "fg_frac", "crop_px"):
            v = np.array([x[f] for x in o])
            c = np.corrcoef(m, v)[0, 1] if v.std() > 0 else float("nan")
            L.append(f"| {f} | {c:+.3f} |")
        L.append("")
        for f in ("sat_frac", "hf_energy", "fg_frac"):
            L += [f"### margin by {f} quintile", ""] + describe_bins(o, f) + [""]

        # Emissive subpopulation and its contribution to the mean.
        for thr in (0.01, 0.05, 0.10):
            hi = [x for x in o if x["sat_frac"] > thr]
            lo = [x for x in o if x["sat_frac"] <= thr]
            if not hi or not lo:
                continue
            L.append(f"- `sat_frac > {thr:g}`: **{len(hi)}/{len(o)}** objects "
                     f"({100*len(hi)/len(o):.1f}%), margin **{np.mean([x['margin_psnr_fg'] for x in hi]):.2f} dB** "
                     f"vs **{np.mean([x['margin_psnr_fg'] for x in lo]):.2f} dB** for the rest "
                     f"(overall {m.mean():.2f}); excluding them moves the mean by "
                     f"{np.mean([x['margin_psnr_fg'] for x in lo]) - m.mean():+.2f} dB")
        L.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
