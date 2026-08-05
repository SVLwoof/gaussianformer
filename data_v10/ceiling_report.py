"""Aggregate ceiling_eval JSONL shards into the numbers we actually reason about.

Two deliberate choices:

* **The unit is the object, not the (scene, view) row.** 14 renders of one object are ~1 sample,
  not 14 -- they share geometry, colour and pruning outcome. Every summary statistic is computed
  over per-object means, so 1,806 val objects give n=1806, not n=25,284. Row-level stats are still
  printed for the tails, where individual renders are the thing being inspected.
* **Foreground-cropped is the headline; whole-image is printed beside it as the foil.** The gap
  between the two columns is the metric-inflation argument, quantified.

  uv run --frozen python -m data_v10.ceiling_report --out meeting_material/ceiling_report.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SLICE_LABEL = {
    "val": "TEST (held-out)",
    "train": "TRAIN (memorised, idx<15000)",
    "unseen2x": "UNSEEN-2x (idx>=15000, never seen by V17)",
}


def load(dirs: list[Path], model: str) -> list[dict]:
    seen, rows = set(), []
    for d in dirs:
        for p in sorted(d.glob("*.jsonl")):
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("model", "v17") != model:
                    continue
                k = (r["split"], r["scene"], r["view"])
                if k in seen:          # a requeued shard can re-emit its last flush
                    continue
                seen.add(k)
                rows.append(r)
    return rows


def per_object(rows: list[dict]) -> dict[tuple[str, int], dict[str, float]]:
    """Collapse each object's views to its mean. This is the analysis unit."""
    acc: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        acc[(r["split"], r["scene"])].append(r)
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float)) and k != "view"]
    return {k: {f: float(np.mean([v[f] for v in vs])) for f in keys} for k, vs in acc.items()}


def describe(vals: np.ndarray) -> str:
    q = np.percentile(vals, [10, 50, 90])
    return (f"{vals.mean():7.3f} | {q[1]:7.3f} | {q[0]:7.3f} | {q[2]:7.3f} | "
            f"{vals.min():7.3f} | {vals.max():7.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", type=Path, default=[Path("data_v10/ceiling")])
    ap.add_argument("--out", type=Path, default=Path("meeting_material/ceiling_report.md"))
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--model", default="v17", help="model_tag to report on (v17, v18, ...)")
    args = ap.parse_args()

    rows = load(args.dirs, args.model)
    if not rows:
        print(f"no rows found for model={args.model}"); return
    objs = per_object(rows)
    L: list[str] = [f"# rec-GT ceiling vs {args.model.upper()}", ""]

    by_split: dict[str, list[dict]] = defaultdict(list)
    for (sp, _), o in objs.items():
        by_split[sp].append(o)
    row_split: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        row_split[r["split"]].append(r)

    L += ["## Coverage", "",
          "| slice | objects | renders |", "|---|---|---|"]
    for sp in ("val", "train", "unseen2x"):
        if sp in by_split:
            L.append(f"| {SLICE_LABEL[sp]} | {len(by_split[sp])} | {len(row_split[sp])} |")

    L += ["", "## Per-object means (n = objects, NOT renders)", ""]
    for sp in ("val", "train", "unseen2x"):
        if sp not in by_split:
            continue
        o = by_split[sp]
        L += [f"### {SLICE_LABEL[sp]}  (n={len(o)})", "",
              "| quantity | mean | p50 | p10 | p90 | min | max |", "|---|---|---|---|---|---|---|"]
        for field, name in [
            ("rec_psnr_fg", "rec-GT PSNR (FG)"),
            ("mdl_psnr_fg", "V17 PSNR (FG)"),
            ("margin_psnr_fg", "margin dB (rec-GT − V17)"),
            ("rec_lpips_fg", "rec-GT LPIPS (FG)"),
            ("mdl_lpips_fg", "V17 LPIPS (FG)"),
            ("margin_lpips_fg", "margin LPIPS (V17 − rec-GT)"),
            ("rec_psnr_full", "rec-GT PSNR (whole-image)"),
            ("mdl_psnr_full", "V17 PSNR (whole-image)"),
            ("fg_frac", "object pixel fraction"),
        ]:
            v = np.array([x[field] for x in o])
            L.append(f"| {name} | {describe(v)} |")
        # how often does the model reach or beat the ceiling?
        m = np.array([x["margin_psnr_fg"] for x in o])
        ml = np.array([x["margin_lpips_fg"] for x in o])
        rr = np.array([r["margin_psnr_fg"] for r in row_split[sp]])
        L += ["",
              f"- objects where **V17 ≥ rec-GT** (PSNR, FG): **{int((m <= 0).sum())}/{len(m)}** "
              f"({100 * (m <= 0).mean():.2f}%)",
              f"- objects where **V17 LPIPS ≤ rec-GT** : **{int((ml <= 0).sum())}/{len(ml)}** "
              f"({100 * (ml <= 0).mean():.2f}%)",
              f"- individual renders where V17 ≥ rec-GT: **{int((rr <= 0).sum())}/{len(rr)}** "
              f"({100 * (rr <= 0).mean():.2f}%)",
              f"- corr(margin, object pixel fraction) = "
              f"{np.corrcoef(m, [x['fg_frac'] for x in o])[0, 1]:+.3f}",
              f"- corr(margin, rec-GT PSNR) = "
              f"{np.corrcoef(m, [x['rec_psnr_fg'] for x in o])[0, 1]:+.3f}", ""]

    # Tails -- the cases worth looking at, at ROW level so a strip can be rendered directly.
    L += ["## Peculiar cases (row level: one scene, one view)", ""]
    for sp in ("val", "train", "unseen2x"):
        if sp not in row_split:
            continue
        rs = sorted(row_split[sp], key=lambda r: r["margin_psnr_fg"])
        L += [f"### {SLICE_LABEL[sp]}", "",
              f"**Smallest margin / V17 closest to (or above) the ceiling** — top {args.topk}:", "",
              "| scene | view | rec-GT dB | V17 dB | margin dB | rec LPIPS | V17 LPIPS |",
              "|---|---|---|---|---|---|---|"]
        for r in rs[:args.topk]:
            L.append(f"| {r['scene']} | {r['view']} | {r['rec_psnr_fg']:.2f} | "
                     f"{r['mdl_psnr_fg']:.2f} | {r['margin_psnr_fg']:+.2f} | "
                     f"{r['rec_lpips_fg']:.4f} | {r['mdl_lpips_fg']:.4f} |")
        L += ["", f"**Largest margin / V17 furthest below the ceiling** — top {args.topk}:", "",
              "| scene | view | rec-GT dB | V17 dB | margin dB | rec LPIPS | V17 LPIPS |",
              "|---|---|---|---|---|---|---|"]
        for r in rs[-args.topk:][::-1]:
            L.append(f"| {r['scene']} | {r['view']} | {r['rec_psnr_fg']:.2f} | "
                     f"{r['mdl_psnr_fg']:.2f} | {r['margin_psnr_fg']:+.2f} | "
                     f"{r['rec_lpips_fg']:.4f} | {r['mdl_lpips_fg']:.4f} |")
        L.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    # machine-readable companion for the strip renderer
    args.out.with_suffix(".json").write_text(json.dumps(
        {f"{sp}": {"objects": len(by_split[sp]), "rows": len(row_split[sp])} for sp in by_split},
        indent=1))
    print("\n".join(L))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
