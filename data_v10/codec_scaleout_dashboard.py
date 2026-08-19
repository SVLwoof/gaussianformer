"""Aggregate all codec scale-out verdicts into one table + trajectory plot (PIL, no mpl).

Collects experiments/overfit/data/codec_scaleout/*/*_verdict.json, orders checkpoints by
cumulative optimizer steps, prints the latest verdict per object, and draws per-object
trajectories (avg delta vs rec-GT over cumulative ksteps) with the tomato ladder as
reference. Output: data_v10/codec_scaleout_dashboard.png.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("experiments/overfit/data/codec_scaleout")
LABEL = {
    "scene_0262": "plate", "scene_0772": "sandal", "scene_1078": "figure",
    "scene_1342": "boxing ring", "scene_0031": "seahorse", "scene_0223": "tent",
    "scene_1423": "molecule", "scene_1223": "doll", "scene_0959": "apple (S)",
    "scene_0874": "vase (S)",
}
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
          "#f032e6", "#9a6324", "#469990", "#808000"]
SETS = ("novel_rand", "novel_close", "novel_far")

def stats(d):
    won = sum(int(d[k]["beats_rec"].split("/")[0]) for k in SETS)
    n = sum(int(d[k]["beats_rec"].split("/")[1]) for k in SETS)
    avg = sum(d[k]["delta"] * int(d[k]["beats_rec"].split("/")[1]) for k in SETS) / n
    return avg, won, n

def ksteps(scene, ckpt_path):
    ep = int(re.search(r"phase2_epoch_(\d+)", ckpt_path).group(1))
    cyc2 = "_r2/" in ckpt_path
    if scene == "scene_0874" and not cyc2:   # vase c1: 6xbs1 -> 1500 steps/epoch
        return ep * 1.5
    return (30.0 if cyc2 else 0.0) + ep * 1.125

def annealed(scene, ckpt_path):
    """True only at a cycle's FINAL epoch, where the cosine LR has annealed back down.

    Each cycle is a warm restart: LR jumps back to phase2_lr at epoch 1, so mid-cycle
    checkpoints are measured mid-disruption and are NOT comparable to end-of-cycle
    points (see the apple's c2-ep9 dip, 2026-08-20).
    """
    ep = int(re.search(r"phase2_epoch_(\d+)", ckpt_path).group(1))
    final = 20 if (scene == "scene_0874" and "_r2/" not in ckpt_path) else 27
    return ep == final

traj = {}
for f in sorted(ROOT.glob("*/*_verdict.json")):
    d = json.loads(f.read_text())
    avg, won, n = stats(d)
    traj.setdefault(f.parent.name, []).append((ksteps(f.parent.name, d["ckpt"]), avg, won, n,
                                               d["ckpt"], annealed(f.parent.name, d["ckpt"])))

TOMATO = [(60, -1.38), (120, +0.36), (150, +0.97), (180, +1.30)]

W, H, ML, MR, MT, MB = 1100, 700, 70, 260, 50, 60
X0, X1 = 0.0, max(60.0, max((p[0] for pts in traj.values() for p in pts), default=60) + 10)
X1 = max(X1, 185)
Y0, Y1 = -14.0, 3.0
img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
fontb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)

def px(x, y):
    return (ML + (x - X0) / (X1 - X0) * (W - ML - MR),
            MT + (Y1 - y) / (Y1 - Y0) * (H - MT - MB))

for gy in range(int(Y0), int(Y1) + 1, 2):
    (x0, y0), (x1, _) = px(X0, gy), px(X1, gy)
    d.line([x0, y0, x1, y0], fill=(230, 230, 230))
    d.text((8, y0 - 8), f"{gy:+d}", fill=(80, 80, 80), font=font)
for gx in range(0, int(X1) + 1, 30):
    (x0, y0), (_, y1) = px(gx, Y0), px(gx, Y1)
    d.line([x0, y0, x0, y1], fill=(235, 235, 235))
    d.text((x0 - 10, H - MB + 8), f"{gx}", fill=(80, 80, 80), font=font)
(zx0, zy), (zx1, _) = px(X0, 0), px(X1, 0)
d.line([zx0, zy, zx1, zy], fill=(0, 0, 0), width=2)
d.text((zx1 - 190, zy - 22), "rec-GT parity", fill=(0, 0, 0), font=fontb)

def draw_line(pts, color, width=2, marks=None):
    for a, b in zip(pts, pts[1:]):
        d.line([*px(*a), *px(*b)], fill=color, width=width)
    for i, a in enumerate(pts):
        x, y = px(*a)
        solid = True if marks is None else marks[i]
        if solid:   # end of cycle: LR annealed, comparable across objects
            d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=color)
        else:       # mid-cycle: measured inside the warm-restart dip
            d.ellipse([x - 4, y - 4, x + 4, y + 4], outline=color, width=2, fill=(255, 255, 255))

ly = MT
print(f"{'object':14s} {'ksteps':>7s} {'avg dB':>8s} {'won':>6s}")
for i, (scene, pts) in enumerate(sorted(traj.items())):
    pts.sort()
    c = COLORS[i % len(COLORS)]
    draw_line([(p[0], p[1]) for p in pts], c, width=3 if scene in ("scene_0959", "scene_0874") else 2,
              marks=[p[5] for p in pts])
    k, avg, won, n, ck, _ann = pts[-1]
    lbl = f"{LABEL.get(scene, scene)}  {avg:+.2f} dB, {won}/{n}"
    d.rectangle([W - MR + 10, ly, W - MR + 26, ly + 14], fill=c)
    d.text((W - MR + 32, ly - 1), lbl, fill=(0, 0, 0), font=font)
    ly += 24
    print(f"{LABEL.get(scene, scene):14s} {k:7.1f} {avg:+8.2f} {won:>3d}/{n:<2d}")
draw_line(TOMATO, (200, 30, 60), width=2)
d.rectangle([W - MR + 10, ly, W - MR + 26, ly + 14], fill=(200, 30, 60))
d.text((W - MR + 32, ly - 1), "tomatoes (ref, +1.30)", fill=(0, 0, 0), font=font)
d.text((ML, 12), "Codec scale-out: avg PSNR delta vs rec-GT over cumulative optimizer ksteps",
       fill=(0, 0, 0), font=fontb)
d.text((W // 2 - 90, H - 26), "cumulative ksteps", fill=(0, 0, 0), font=font)
d.text((ML, 30), "solid = end of cycle (LR annealed, comparable);  hollow = mid-cycle (warm-restart dip, NOT comparable)", fill=(90, 90, 90), font=font)
img.save("data_v10/codec_scaleout_dashboard.png")
print("-> data_v10/codec_scaleout_dashboard.png")
