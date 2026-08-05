"""Where in SPATIAL FREQUENCY does V17 lose energy relative to GT?

Two hypotheses for the ~14.5 dB gap make opposite spectral predictions:

  (a) TOKENISATION limit -- ray tokens are 8x8 patches, so a 512^2 render passes through a 64x64
      token grid. If that is the binding constraint, V17 should track GT up to ~0.125 cycles/px
      (1/8) and collapse beyond it: a KNEE at the patch scale.
  (b) ROUTING failure -- each ray token must resolve which of ~20k Gaussians land in its 8x8
      footprint via cross-attention. If that is the constraint, the deficit is BROADBAND with a
      smooth rolloff and no feature at 8 px.

Note (a) is already in trouble: the June overfit probe hit 49-53 dB at resolution 512 with this
same patch_size=8, so the tokenisation cannot impose a ~30 dB ceiling. This measures where the
energy actually goes.

Everything is computed on the NATIVE 512 pixel grid (foreground box, NO resize). The cropped+
resized panels used for the metrics rescale spatial frequency per object and would destroy the
link to the 8 px patch scale -- the whole point here.

CPU only; reads existing showcase renders.

  uv run --no-sync python -m data_v10.spectrum_probe --out meeting_material/spectrum_probe.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from render_compare import load_gt

SHOWCASE = Path("data_v10/showcase")
RENDERS = Path("data_v10/renders")
RES = 512


def native_box(gt: np.ndarray, pad: int = 12) -> tuple[int, int, int, int]:
    lum = gt.mean(-1)
    ys, xs = np.where(lum > 0.02)
    if len(ys) < 10:
        return 0, gt.shape[0], 0, gt.shape[1]
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    half = max(y1 - y0, x1 - x0) // 2 + pad
    H = gt.shape[0]
    return max(0, cy - half), min(H, cy + half), max(0, cx - half), min(H, cx + half)


def radial_cross(gt: np.ndarray, md: np.ndarray, nbins: int = 64):
    """Per-frequency-band cross terms for MTF *and* coherence.

    MTF (|M|^2 / |G|^2) says whether the model has the right AMOUNT of energy at a frequency.
    Coherence |sum G M*| / sqrt(sum|G|^2 sum|M|^2) says whether that energy is in the right PLACE:
    it is 1 when the model's structure is phase-aligned with GT and ~0 when the model emits
    equally-strong but uncorrelated detail. A model with MTF~0.8 and coherence~0.3 is not blurring
    -- it is inventing.
    """
    n = min(gt.shape[0], gt.shape[1], md.shape[0], md.shape[1])
    w = np.hanning(n)
    win = w[:, None] * w[None, :]
    G = np.fft.fftshift(np.fft.fft2((gt[:n, :n].mean(-1) - gt[:n, :n].mean()) * win))
    M = np.fft.fftshift(np.fft.fft2((md[:n, :n].mean(-1) - md[:n, :n].mean()) * win))
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = (np.hypot(yy - c, xx - c) / n).ravel()
    bins = np.linspace(0, 0.5, nbins + 1)
    idx = np.digitize(r, bins) - 1
    Gf, Mf = G.ravel(), M.ravel()
    gg = np.zeros(nbins); mm = np.zeros(nbins); gm = np.zeros(nbins, dtype=complex)
    for b in range(nbins):
        s = idx == b
        if s.any():
            gg[b] = (np.abs(Gf[s]) ** 2).sum()
            mm[b] = (np.abs(Mf[s]) ** 2).sum()
            gm[b] = (Gf[s] * np.conj(Mf[s])).sum()
    return 0.5 * (bins[:-1] + bins[1:]), gg, mm, gm


def radial_power(img: np.ndarray, nbins: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged power spectrum of the luminance, in cycles/pixel (0 .. 0.5)."""
    lum = img.mean(-1)
    n = min(lum.shape)
    lum = lum[:n, :n]
    # Hann window: the crop has hard edges, which would otherwise smear energy across all freqs.
    w = np.hanning(n)
    lum = (lum - lum.mean()) * (w[:, None] * w[None, :])
    P = np.abs(np.fft.fftshift(np.fft.fft2(lum))) ** 2
    cy = cx = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(yy - cy, xx - cx) / n            # 0 .. ~0.707, in cycles/pixel
    bins = np.linspace(0, 0.5, nbins + 1)
    idx = np.digitize(r.ravel(), bins) - 1
    Pf = P.ravel()
    out = np.full(nbins, np.nan)
    for b in range(nbins):
        m = idx == b
        if m.any():
            out[b] = Pf[m].mean()
    return 0.5 * (bins[:-1] + bins[1:]), out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="V17-ep36")
    ap.add_argument("--out", type=Path, default=Path("meeting_material/spectrum_probe.md"))
    ap.add_argument("--nbins", type=int, default=64)
    args = ap.parse_args()

    # hf_energy per (scene, view) from the content pass, to stratify by detail content.
    hf: dict[tuple[int, int], float] = {}
    p = Path("data_v10/ceiling/content_stats.jsonl")
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                c = json.loads(line)
                if c["split"] in ("train", "unseen2x"):
                    hf[(c["scene"], c["view"])] = c["hf_energy"]

    files = sorted(SHOWCASE.glob(f"s*_v*_{args.model}.png"))
    print(f"{len(files)} {args.model} renders", flush=True)

    curves: list[tuple[float, np.ndarray, np.ndarray]] = []
    freqs = None
    for f in files:
        scene = int(f.name[1:6])
        view = int(f.name[8:10])
        gt_p = RENDERS / f"scene_{scene:04d}_view_{view}.png"
        if not gt_p.exists():
            continue
        gt = load_gt(gt_p, RES)
        md = load_gt(f, RES)
        y0, y1, x0, x1 = native_box(gt)
        if min(y1 - y0, x1 - x0) < 64:
            continue
        fr, pg = radial_power(gt[y0:y1, x0:x1], args.nbins)
        _, pm = radial_power(md[y0:y1, x0:x1], args.nbins)
        freqs = fr
        curves.append((hf.get((scene, view), float("nan")), pg, pm))
    print(f"{len(curves)} usable pairs", flush=True)

    def mtf(sel: list[tuple[float, np.ndarray, np.ndarray]]) -> np.ndarray:
        """Ratio of model power to GT power per frequency -- an empirical MTF.
        Averaged as a ratio of sums (energy-weighted), not a mean of ratios, so a few
        near-zero-GT-power bins cannot dominate."""
        G = np.nansum(np.stack([c[1] for c in sel]), axis=0)
        M = np.nansum(np.stack([c[2] for c in sel]), axis=0)
        return M / (G + 1e-20)

    L = ["# Spatial-frequency probe: where does V17 lose energy?", "",
         f"{len(curves)} renders, native 512 grid, foreground box, no resize. "
         "MTF = model power / GT power per radial frequency (1.0 = perfect).", "",
         f"**Patch scale = 1/8 = 0.125 cycles/px.** A knee there implicates tokenisation; "
         "a smooth broadband rolloff does not.", ""]

    valid = [c for c in curves if not np.isnan(c[0])]
    groups = [("ALL", curves)]
    if len(valid) > 30:
        h = np.array([c[0] for c in valid])
        lo, hi = np.percentile(h, [33.3, 66.7])
        groups += [("hf LOW (smooth)", [c for c in valid if c[0] <= lo]),
                   ("hf MID", [c for c in valid if lo < c[0] <= hi]),
                   ("hf HIGH (detail)", [c for c in valid if c[0] > hi])]

    L += ["| freq (cyc/px) | period (px) | " + " | ".join(f"MTF {g}" for g, _ in groups) + " |",
          "|---" * (2 + len(groups)) + "|"]
    mtfs = [mtf(g) for _, g in groups]
    for i, fq in enumerate(freqs):
        if fq < 0.008:
            continue
        row = f"| {fq:.3f} | {1/fq:5.1f} | " + " | ".join(f"{m[i]:.3f}" for m in mtfs) + " |"
        L.append(row)

    L += ["", "## Half-power points (MTF crosses 0.5)", ""]
    for (name, _), m in zip(groups, mtfs):
        below = np.where(m < 0.5)[0]
        f50 = freqs[below[0]] if len(below) else float("nan")
        L.append(f"- **{name}**: MTF < 0.5 beyond **{f50:.3f} cyc/px** "
                 f"(period {1/f50:.1f} px)" if len(below) else f"- **{name}**: never drops below 0.5")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    print("\n".join(L[:40]))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
