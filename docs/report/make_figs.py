"""Figures for the 2026-09-16 supervisor report. matplotlib Agg, no display needed."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from pathlib import Path
D = Path("docs/report"); D.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3,
                     "figure.dpi": 200, "savefig.bbox": "tight"})
B = 56.0  # bytes per Gaussian (14 float32)

# --- 1. rate-distortion: slipper -----------------------------------------
gs_n   = np.array([2000, 5000, 20000])
gs_psnr= np.array([31.1, 32.0, 34.8])          # measured rec-GT bars
gs_mb  = gs_n * B / 1e6
fit = np.polyfit(np.log2(gs_n), gs_psnr, 1)     # ~1.2 dB per doubling
ext_n = np.array([20000, 50000, 120000]); ext_mb = ext_n*B/1e6
ext_p = np.polyval(fit, np.log2(ext_n))
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(gs_mb, gs_psnr, "o-", color="#444", label="gsplat rasterizer (measured)")
ax.plot(ext_mb, ext_p, "--", color="#999", lw=1.1, label="gsplat extrapolated (+1.2 dB/doubling)")
mb20, p20 = 20000*B/1e6, 34.80
mb_win = (20000*B + 5234688)/1e6; p_win = 35.40
mb_r1 = (20000*B + 479232)/1e6
ax.plot(mb20, p20, "o", ms=9, mfc="none", mec="#444", mew=1.6)
ax.annotate("20k splat, rasterized\n1.12 MB / 34.80 dB", (mb20, p20), (0.13, 35.9),
            fontsize=7.5, color="#444", arrowprops=dict(arrowstyle="->", color="#444", lw=.7))
ax.plot(mb_win, p_win, "*", ms=15, color="#1a7f37")
ax.annotate("ours: 20k splat + r=4 adapter\n6.35 MB / 35.40 dB (+0.60 vs its own splat)",
            (mb_win, p_win), (1.6, 32.6), fontsize=7.5, color="#1a7f37",
            arrowprops=dict(arrowstyle="->", color="#1a7f37", lw=.8))
p_equal = np.polyval(fit, np.log2(mb_win*1e6/B))
ax.plot([mb_win, mb_win], [p_win, p_equal], color="#c00", lw=1.4)
ax.annotate(f"equal-bytes gap\n{p_equal - p_win:.1f} dB in the rasterizer's favour",
            (mb_win, (p_win+p_equal)/2), (2.0, 37.0), fontsize=7.5, color="#c00",
            arrowprops=dict(arrowstyle="->", color="#c00", lw=.8))
ax.axvline(mb_r1, color="#b36b00", ls=":", lw=1.3)
ax.annotate("r=1 attention-only adapter\nwould sit here (1.60 MB) - untested",
            (mb_r1, 31.4), (0.115, 31.0), fontsize=7.5, color="#b36b00",
            arrowprops=dict(arrowstyle="->", color="#b36b00", lw=.7))
ax.set_xscale("log"); ax.set_xlim(0.09, 9); ax.set_ylim(30.6, 38.2)
ax.set_xlabel("bytes per object (MB, log scale)"); ax.set_ylabel("FG-crop PSNR vs GT (dB)")
ax.set_title("Rate-distortion, slipper (unseen real scan)")
ax.legend(loc="lower right", fontsize=7.5, framealpha=.95)
fig.savefig(D/"fig_rate_distortion.png"); plt.close(fig)

# --- 2. heldout ladder (N=10, one cycle each) ----------------------------
arms = ["V18 baseline","+P2 (proj RoPE)","+P1 (canvas)","P1+P2","P1+P2+fg","P1+P2+fg, 4 cycles"]
held = [20.44, 19.07, 17.47, 16.95, 17.27, 17.06]
ci   = [(20.22,20.67),(18.84,19.29),(17.24,17.70),(16.73,17.18),(17.04,17.49),(16.78,17.34)]
err  = np.array([[h-l for h,(l,_) in zip(held,ci)],[u-h for h,(_,u) in zip(held,ci)]])
fig, ax = plt.subplots(figsize=(6.2, 2.9))
c = ["#888","#888","#1a7f37","#1a7f37","#1a7f37","#1a7f37"]
ax.barh(range(len(arms)), held, xerr=err, color=c, height=.62, error_kw=dict(lw=1, capsize=2.5))
ax.set_yticks(range(len(arms))); ax.set_yticklabels(arms); ax.invert_yaxis()
ax.set_xlim(15.5, 21); ax.set_xlabel("held-out margin below the rasterizer (dB; lower = better)")
ax.set_title("Generalisation to 300 unseen objects (N=10 training set, 95% CI)")
for i,h in enumerate(held): ax.text(h+0.30, i, f"{h:.2f}", va="center", fontsize=8)
fig.savefig(D/"fig_heldout_ladder.png"); plt.close(fig)

# --- 3. scaling: the canvas gap grows with data --------------------------
fig, ax = plt.subplots(figsize=(6.2, 3.0))
N = [10, 100]
ax.plot(N, [20.44, 17.93], "o-", color="#888", label="baseline (V18 recipe)")
ax.plot(N, [19.39, 16.38], "s-", color="#c05621", label="P2 + fg (no canvas)")
ax.plot(N, [17.27, 11.25], "^-", color="#1a7f37", label="P1 + P2 + fg (canvas)")
ax.plot([1000],[16.70], "x", color="#888", ms=8); ax.annotate("baseline at N=1000", (1000,16.70), (300,18.4), fontsize=7.5, color="#888", arrowprops=dict(arrowstyle="->", color="#888", lw=.7))
for n,a,b in [(10,19.39,17.27),(100,16.38,11.25)]:
    ax.annotate("", (n,a), (n,b), arrowprops=dict(arrowstyle="<->", color="#1a7f37", lw=1))
    ax.text(n*1.1, (a+b)/2, f"{a-b:.2f} dB", color="#1a7f37", fontsize=8, va="center")
ax.set_xscale("log"); ax.set_xlabel("training objects"); ax.set_ylabel("held-out margin (dB, lower = better)")
ax.set_title("The canvas advantage grows with data (equal compute)"); ax.legend(fontsize=7.5); ax.invert_yaxis()
fig.savefig(D/"fig_scaling.png"); plt.close(fig)
print("wrote", *[p.name for p in sorted(D.glob("fig_*.png"))])
