"""Compose a horizontal strip of cherry-picked views for the report.

Layout per output PNG: [full|pruned|gf] for each chosen view, concatenated
horizontally with gutters, single row.

Usage:
  python data_external/build_tomatoes_strip.py
"""
import os
from pathlib import Path
import numpy as np
import imageio.v3 as iio

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "data_external/tomatoes/renders"
# Set REPORT_FIGURES_DIR to write the strips elsewhere (e.g. a sibling report repo).
OUT = Path(os.environ.get("REPORT_FIGURES_DIR", REPO_ROOT / "figures"))

VIEWS = [0, 1, 8]
N = 30000

PHASES = {
    "v6":   ROOT / f"gaussianformer_n{N}",
    "v9":   ROOT / f"gaussianformer_n{N}_v9_ep60",
    "v10b": ROOT / f"gaussianformer_n{N}_v10b_ep26",
}

GUTTER_TRIPLE = 6
GUTTER_BETWEEN = 24


def load_view(d: Path, i: int) -> np.ndarray:
    return iio.imread(d / f"view_{i:02d}.png")


def main():
    full_dir = ROOT / "gsplat_full"
    pruned_dir = ROOT / f"gsplat_n{N}"
    OUT.mkdir(parents=True, exist_ok=True)

    for tag, gf_dir in PHASES.items():
        triples = []
        for v in VIEWS:
            full = load_view(full_dir, v)
            pruned = load_view(pruned_dir, v)
            gf = load_view(gf_dir, v)
            H = full.shape[0]
            gutter = np.full((H, GUTTER_TRIPLE, 3), 255, dtype=np.uint8)
            triples.append(np.concatenate([full, gutter, pruned, gutter, gf], axis=1))

        H = triples[0].shape[0]
        between = np.full((H, GUTTER_BETWEEN, 3), 255, dtype=np.uint8)
        strip = triples[0]
        for t in triples[1:]:
            strip = np.concatenate([strip, between, t], axis=1)

        out_path = OUT / f"tomatoes_{tag}_n{N}_strip.png"
        iio.imwrite(out_path, strip)
        print(f"wrote {out_path}  ({strip.shape[1]}x{strip.shape[0]})")


if __name__ == "__main__":
    main()
