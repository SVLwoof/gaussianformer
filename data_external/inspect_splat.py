"""Quick look at an arbitrary 3DGS ply: orbit renders + isolation stats.

Used to vet candidate scans before committing them to the codec pipeline — the key
question is whether the capture is an ISOLATED object (like the tomatoes) or a scene
with background, which the pipeline cannot use without cropping.

  PYTHONPATH=. python data_external/inspect_splat.py --ply path/to.ply --out tmp/inspect
"""
from __future__ import annotations
import argparse
import numpy as np, torch, imageio.v3 as iio
from pathlib import Path
from plyfile import PlyData
from data_external.orbit import look_at_blender, c2w_to_viewmat
from data_v10.prune_recovery import rasterize

C0 = 0.28209479177387814

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--radius", type=float, default=None, help="orbit radius (default: 2.2x median extent)")
    ap.add_argument("--keep_pct", type=float, default=100.0,
                    help="drop gaussians beyond this percentile of distance from the median centre")
    ap.add_argument("--drop_faint_large", action="store_true",
                    help="drop low-opacity, large-scale gaussians (typical floater/plume signature)")
    args = ap.parse_args()

    v = PlyData.read(args.ply)["vertex"]
    means = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float32)
    raw_scales = np.stack([v[f"scale_{i}"] for i in range(3)], -1).astype(np.float32)
    scales = raw_scales if raw_scales.min() >= 0 else np.exp(raw_scales)
    rots = np.stack([v[f"rot_{i}"] for i in range(4)], -1).astype(np.float32)
    colors = np.clip(0.5 + C0 * np.stack([v[f"f_dc_{i}"] for i in range(3)], -1), 0, 1).astype(np.float32)
    op = np.asarray(v["opacity"], np.float32)
    opac = op if raw_scales.min() >= 0 else 1 / (1 + np.exp(-op))

    # centre and normalise to the tomato convention: median-centred, extent ~[-0.45, 0.45]
    med = np.median(means, 0)
    means = means - med

    keep = np.ones(len(means), bool)
    if args.keep_pct < 100.0:
        r = np.linalg.norm(means, axis=1)
        keep &= r <= np.percentile(r, args.keep_pct)
    if args.drop_faint_large:
        smax = scales.max(1)
        keep &= ~((smax > np.percentile(smax, 98)) & (opac < 0.3))
    if not keep.all():
        print(f"filter keeps {keep.sum()}/{len(keep)} ({100 * keep.mean():.1f}%)")
        means, scales, rots, colors, opac = (a[keep] for a in (means, scales, rots, colors, opac))
    p95 = np.percentile(np.linalg.norm(means, axis=1), 95)
    s = 0.45 / p95
    means, scales = means * s, scales * s
    print(f"{len(means)} gaussians | p95 radius {p95:.3f} -> scaled by {s:.3f}")

    device = "cuda"
    g = dict(means=torch.as_tensor(means, device=device), quats=torch.as_tensor(rots, device=device),
             scales=torch.as_tensor(scales, device=device), colors=torch.as_tensor(colors, device=device),
             opacities=torch.as_tensor(opacs := opac.reshape(-1), device=device))
    RES, FOV = args.res, 45.0
    focal = 0.5 * RES / np.tan(0.5 * np.radians(FOV))
    K = np.array([[focal, 0, RES / 2], [0, focal, RES / 2], [0, 0, 1]], np.float32)
    r = args.radius or 1.7
    c2ws = []
    for i in range(6):
        az = i * np.pi / 3
        el = np.radians(20 if i % 2 == 0 else -5)
        eye = np.array([r * np.cos(el) * np.cos(az), r * np.sin(el), r * np.cos(el) * np.sin(az)], np.float32)
        c2ws.append(look_at_blender(eye, np.zeros(3), np.array([0., 1., 0.], np.float32)))
    vm = torch.as_tensor(np.stack([c2w_to_viewmat(c) for c in c2ws]), device=device)
    Kt = torch.as_tensor(np.tile(K[None], (len(c2ws), 1, 1)), device=device)
    imgs = rasterize(g, vm, Kt, list(range(len(c2ws))), RES)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    strip = np.concatenate([np.clip(im.cpu().numpy(), 0, 1) for im in imgs], axis=1)
    iio.imwrite(out / "orbit_strip.png", (strip * 255).astype(np.uint8))
    # fraction of non-black pixels tells us how much background/enclosure there is
    fg = (strip.max(-1) > 0.03).mean()
    print(f"non-black pixel fraction across 6 views: {fg:.3f} (tomato-like isolated object ~0.05-0.20)")
    print("->", out / "orbit_strip.png")

if __name__ == "__main__":
    main()
