"""Prep an external raw 3DGS scan (superspl.at etc.) for the codec scale-out pipeline.

Chain: raw PLY -> normalize (median-center, optional 180-deg X flip, max|pos|->0.45)
-> significance prune to 20k + gsplat recovery FT (tomatoes_rebuild recipe) -> the
codec_scaleout/<name> layout that train_codec_scaleout.sh / codec_scaleout_eval.py
already consume: h5s/<name>.h5 (rec splat + 9000 train c2w), renders/ (9000 FULL-splat
rasterizations, codec5 view distribution), codec_eval_* sets (codec v1 conventions).

Seed bases start at 200000 + 100*index (disjoint from the val scenes' 100000+10*sid).
--probe renders 3 elevated views per flip state and exits (pick the upright one).

Usage: python data_external/prep_external_codec.py --name octopus --seed_base 200000 [--flip_x]
"""
from __future__ import annotations
import argparse
import numpy as np, torch, h5py, imageio.v3 as iio
from pathlib import Path
from plyfile import PlyData
from data_external.orbit import look_at_blender, c2w_to_viewmat, make_orbit_views
from data_v10.prune_recovery import significance_topk, recover, rasterize, to_dev

RES, FOV = 512, 45.0
UP = np.array([0., 1., 0.], np.float32)
device = "cuda"


def load_normalized(name: str, flip_x: bool) -> dict[str, np.ndarray]:
    v = PlyData.read(f"data_external/{name}/{name}.ply")["vertex"]
    means = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float32)
    quats = np.stack([v[f"rot_{i}"] for i in range(4)], -1).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-9
    C0 = 0.28209479177387814
    colors = np.clip(0.5 + C0 * np.stack([v[f"f_dc_{i}"] for i in range(3)], -1), 0, 1).astype(np.float32)
    opacities = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float32)))
    scales = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], -1).astype(np.float32))

    means -= np.median(means, axis=0)
    if flip_x:
        means[:, 1] *= -1.0
        means[:, 2] *= -1.0
        qw, qx, qy, qz = (quats[:, i].copy() for i in range(4))
        quats = np.stack([-qx, qw, -qz, qy], axis=-1).astype(np.float32)
    s = 0.45 / max(float(np.abs(means).max()), 1e-9)
    means *= s
    scales *= s
    print(f"{name}: {len(means)} gaussians, flip_x={flip_x}, scaled by {s:.4f}")
    return dict(means=means, scales=scales, rotations=quats, colors=colors,
                opacities=opacities.astype(np.float32).reshape(-1))


def cams(n, seed, radii, el_range=(-15, 55)):
    rng = np.random.default_rng(seed)
    c2ws = []
    for _ in range(n):
        az = rng.uniform(0, 2 * np.pi)
        el = rng.uniform(np.radians(el_range[0]), np.radians(el_range[1]))
        r = rng.uniform(*radii) if isinstance(radii, tuple) else radii
        eye = np.array([r * np.cos(el) * np.cos(az), r * np.sin(el), r * np.cos(el) * np.sin(az)], np.float32)
        c2ws.append(look_at_blender(eye, np.zeros(3), UP))
    return np.stack(c2ws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed_base", type=int, required=True)
    ap.add_argument("--flip_x", action="store_true")
    ap.add_argument("--probe", action="store_true", help="render 3 elevated views per flip state and exit")
    args = ap.parse_args()
    name, base = args.name, args.seed_base

    focal = 0.5 * RES / np.tan(0.5 * np.radians(FOV))
    K = np.array([[focal, 0, RES / 2], [0, focal, RES / 2], [0, 0, 1]], np.float32)

    def full_tensors(a):
        return to_dev({"means": a["means"], "scales": a["scales"], "quats": a["rotations"],
                       "colors": a["colors"], "opacities": a["opacities"]}, device)

    if args.probe:
        rows = []
        for flip in (False, True):
            t = full_tensors(load_normalized(name, flip))
            c2ws = cams(3, seed=7, radii=1.7, el_range=(40, 40))
            vm = torch.as_tensor(np.stack([c2w_to_viewmat(c) for c in c2ws]), device=device)
            Kt = torch.as_tensor(np.tile(K[None], (3, 1, 1)), device=device)
            imgs = rasterize(t, vm, Kt, [0, 1, 2], RES).clamp(0, 1).cpu().numpy()
            rows.append(np.concatenate(list(imgs), axis=1))
        out = Path(f"tmp/orient_probe_{name}.png")
        out.parent.mkdir(exist_ok=True)
        iio.imwrite(out, (np.concatenate(rows, axis=0) * 255).astype(np.uint8))
        print(f"probe (top row: no flip, bottom row: flip_x), el +40 -> {out}")
        return

    a = load_normalized(name, args.flip_x)
    full_t = full_tensors(a)

    # 20k significance prune + recovery against 64-view orbit of the full splat
    vm64, K64 = make_orbit_views(64, 1.7, 45.0, RES, up_axis="y")
    vm64, K64 = torch.from_numpy(vm64).to(device), torch.from_numpy(K64).to(device)
    pruned = significance_topk(a, vm64, K64, 20000, device)
    pruned["opacities"] = np.asarray(pruned["opacities"]).reshape(-1)
    pruned = {k: (v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)) for k, v in pruned.items()}
    targets = torch.cat([rasterize(full_t, vm64, K64, list(range(i, min(i + 8, 64))), RES)
                         for i in range(0, 64, 8)])
    rec = recover(pruned, targets, vm64, K64, 1500, device)

    root = Path(f"experiments/overfit/data/codec_scaleout/{name}")
    (root / "h5s").mkdir(parents=True, exist_ok=True)
    train_c2w = np.concatenate([
        cams(4500, seed=base + 1, radii=(1.05, 2.55)),
        cams(4500, seed=base + 2, radii=(1.05, 2.55), el_range=(-15, 20)),
    ])
    with h5py.File(root / "h5s" / f"{name}.h5", "w") as f:
        f.create_dataset("means", data=rec["means"]); f.create_dataset("scales", data=rec["scales"])
        f.create_dataset("rotations", data=rec["rotations"]); f.create_dataset("colors", data=rec["colors"])
        f.create_dataset("opacities", data=np.asarray(rec["opacities"]).reshape(-1, 1))
        f.create_dataset("c2w", data=train_c2w)
        f.create_dataset("fov", data=np.full(len(train_c2w), FOV, np.float32))

    def render_set(c2ws, outdir, stem):
        outdir.mkdir(parents=True, exist_ok=True)
        vm = torch.as_tensor(np.stack([c2w_to_viewmat(c) for c in c2ws]), device=device)
        Kt = torch.as_tensor(np.tile(K[None], (len(c2ws), 1, 1)), device=device)
        for i in range(0, len(c2ws), 8):
            idxs = [j for j in range(i, min(i + 8, len(c2ws))) if not (outdir / f"{stem}_view_{j}.png").exists()]
            if not idxs:
                continue
            imgs = rasterize(full_t, vm, Kt, idxs, RES)
            for j, im in zip(idxs, imgs):
                iio.imwrite(outdir / f"{stem}_view_{j}.png", (im.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))

    render_set(train_c2w, root / "renders", name)
    for ename, n, seed, radii in (("novel_rand", 24, base + 3, (1.35, 2.15)),
                                  ("novel_close", 8, base + 4, 1.15),
                                  ("novel_far", 8, base + 5, 2.45)):
        c = cams(n, seed, radii)
        np.save(root / f"codec_eval_{ename}_c2w.npy", c)
        render_set(c, root / "renders" / f"codec_eval_{ename}", "gt")
    print(f"{name}: 9000 train views + 40 eval views -> {root}")


if __name__ == "__main__":
    main()
