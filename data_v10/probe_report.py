"""Summarise N=10 probe verdicts from ceiling_eval rows: per-object means, then the mean.

  uv run --no-sync python data_v10/probe_report.py [--dir data_v10/ceiling] [--tags probe_p3_hf8 ...]
Prints, per tag and split (train = fit on the 10 objects, heldout = heldout300):
  margin = rec-GT PSNR_fg - model PSNR_fg (lower is better), model PSNR_fg, LPIPS margin.
Reference (record): nsweep_n10 baseline fit 7.57 / heldout 20.47.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", type=Path, default=Path("data_v10/ceiling"))
ap.add_argument("--tags", nargs="*", default=None)
a = ap.parse_args()
files = sorted(a.dir.glob("probe_*_train.jsonl")) + sorted(a.dir.glob("nsweep_n10*_train.jsonl"))
tags = a.tags or sorted({f.name[: -len("_train.jsonl")] for f in files})
print(f"{'tag':28s} {'split':8s} {'objs':>5s} {'margin dB':>10s} {'model PSNR':>11s} {'rec-GT':>8s} {'LPIPS m':>8s}")
for tag in tags:
    for split in ("train", "heldout"):
        p = a.dir / f"{tag}_{split}.jsonl"
        if not p.exists():
            continue
        per = defaultdict(list)
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            per[r["scene"]].append(r)
        if not per:
            continue
        m = np.array([np.mean([r["margin_psnr_fg"] for r in rs]) for rs in per.values()])
        mp = np.array([np.mean([r["mdl_psnr_fg"] for r in rs]) for rs in per.values()])
        rp = np.array([np.mean([r["rec_psnr_fg"] for r in rs]) for rs in per.values()])
        ml = np.array([np.mean([r["margin_lpips_fg"] for r in rs]) for rs in per.values()])
        print(f"{tag:28s} {split:8s} {len(per):5d} {m.mean():10.2f} {mp.mean():11.2f} {rp.mean():8.2f} {ml.mean():8.4f}")
