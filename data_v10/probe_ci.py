"""Bootstrap CIs and paired tests for the N=10 probe margins.

The headline margins are means over 40 (train) or 1200 (heldout) rows that are NOT independent:
4 views of each scene share an object. Resampling SCENES (a cluster bootstrap) rather than rows
is what makes an interval honest here. Paired differences between two arms reuse the same scene
resampling, which is much tighter than comparing two independent CIs.

  uv run --no-sync python -m data_v10.probe_ci --arms probe_p1p2_fg probe_p1p2_fg_c4 ...
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CEIL = Path("data_v10/ceiling")


def load(arm: str, split: str) -> dict[int, list[float]]:
    f = CEIL / f"{arm}_{split}.jsonl"
    by = defaultdict(list)
    for line in f.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by[r["scene"]].append(r["margin_psnr_fg"])
    return dict(by)


def boot(by: dict[int, list[float]], idx: np.ndarray) -> np.ndarray:
    scenes = np.array(sorted(by))
    per = {s: float(np.mean(by[s])) for s in scenes}
    vals = np.array([per[s] for s in scenes])
    return vals[idx].mean(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    data = {arm: load(arm, a.split) for arm in a.arms}
    common = sorted(set.intersection(*(set(d) for d in data.values())))
    assert common, "arms share no scenes"
    rng = np.random.default_rng(a.seed)
    idx = rng.integers(0, len(common), size=(a.n_boot, len(common)))
    curves = {}
    for arm, d in data.items():
        sub = {s: d[s] for s in common}
        curves[arm] = boot(sub, idx)
        m = np.mean([np.mean(sub[s]) for s in common])
        lo, hi = np.percentile(curves[arm], [2.5, 97.5])
        print(f"{arm:28s} {a.split:8s} n_scenes={len(common):4d}  margin {m:6.2f}  95% CI [{lo:6.2f}, {hi:6.2f}]")

    print("\npaired differences (arm_j - arm_i), same scene resampling:")
    for i in range(len(a.arms)):
        for j in range(i + 1, len(a.arms)):
            d = curves[a.arms[j]] - curves[a.arms[i]]
            lo, hi = np.percentile(d, [2.5, 97.5])
            p = 2 * min((d > 0).mean(), (d < 0).mean())
            sig = "*" if lo * hi > 0 else " "
            print(f"  {a.arms[j]:26s} - {a.arms[i]:26s} = {d.mean():6.2f}  95% CI [{lo:6.2f}, {hi:6.2f}] p={p:.4f} {sig}")


if __name__ == "__main__":
    main()
