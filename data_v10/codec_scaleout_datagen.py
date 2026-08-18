"""Per-object codec training data for the 10-object scale-out (2026-08-18 locked list).

Mirrors the tomato codec recipe on a val object: input = its recovered 20k splat
(h5s_20k_rec_val), supervision = rasterizations of its FULL 50k splat (full_h5s_val).
Train views = 4500 uniform (el -15..55) + 4500 grazing (el -15..20), r 1.05-2.55 --
the codec5 distribution; a codec4-style run can train on views [:4500]. Eval sets follow
codec v1 conventions: rand 24 @ r 1.35-2.15, close 8 @ 1.15, far 8 @ 2.45, disjoint seeds.
Val objects share the tomato normalization scale (bbox ~[-0.45,0.45], orbit r 1.7, fov 45),
so the radius ranges carry over. Resume-safe: existing renders are skipped.

Usage: python data_v10/codec_scaleout_datagen.py --scene scene_0959
"""
from __future__ import annotations
import argparse
import numpy as np, torch, h5py, imageio.v3 as iio
from pathlib import Path
from data_external.orbit import look_at_blender, c2w_to_viewmat
from data_v10.prune_recovery import rasterize

RES, FOV = 512, 45.0
UP = np.array([0., 1., 0.], np.float32)
device = "cuda"

def cams(n, seed, radii, el_range=(-15, 55)):
    rng = np.random.default_rng(seed)
    c2ws = []
    for _ in range(n):
        az = rng.uniform(0, 2*np.pi)
        el = rng.uniform(np.radians(el_range[0]), np.radians(el_range[1]))
        r = rng.uniform(*radii) if isinstance(radii, tuple) else radii
        eye = np.array([r*np.cos(el)*np.cos(az), r*np.sin(el), r*np.cos(el)*np.sin(az)], np.float32)
        c2ws.append(look_at_blender(eye, np.zeros(3), UP))
    return np.stack(c2ws)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    args = ap.parse_args()
    scene = args.scene
    sid = int(scene.split("_")[1])
    base = 100_000 + 10 * sid  # per-scene seed block, disjoint across objects

    with h5py.File(f"data_v10/full_h5s_val/{scene}.h5") as f:
        full = {k: np.array(f[k], np.float32) for k in ("means", "scales", "rotations", "colors", "opacities")}
    with h5py.File(f"data_v10/h5s_20k_rec_val/{scene}.h5") as f:
        rec = {k: np.array(f[k], np.float32) for k in ("means", "scales", "rotations", "colors", "opacities")}

    full_t = {k: torch.as_tensor(v, device=device) for k, v in
              dict(means=full["means"], quats=full["rotations"], scales=full["scales"],
                   colors=full["colors"], opacities=full["opacities"].reshape(-1)).items()}
    focal = 0.5 * RES / np.tan(0.5 * np.radians(FOV))
    K = np.array([[focal, 0, RES/2], [0, focal, RES/2], [0, 0, 1]], np.float32)

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

    root = Path(f"experiments/overfit/data/codec_scaleout/{scene}")
    (root / "h5s").mkdir(parents=True, exist_ok=True)

    train_c2w = np.concatenate([
        cams(4500, seed=base + 1, radii=(1.05, 2.55)),
        cams(4500, seed=base + 2, radii=(1.05, 2.55), el_range=(-15, 20)),
    ])
    with h5py.File(root / "h5s" / f"{scene}.h5", "w") as f:
        for k, v in rec.items():
            f.create_dataset(k, data=v)
        f.create_dataset("c2w", data=train_c2w)
        f.create_dataset("fov", data=np.full(len(train_c2w), FOV, np.float32))
    render_set(train_c2w, root / "renders", f"{scene}")

    for name, n, seed, radii in (("novel_rand", 24, base + 3, (1.35, 2.15)),
                                 ("novel_close", 8, base + 4, 1.15),
                                 ("novel_far", 8, base + 5, 2.45)):
        c = cams(n, seed, radii)
        np.save(root / f"codec_eval_{name}_c2w.npy", c)
        render_set(c, root / "renders" / f"codec_eval_{name}", "gt")
    print(f"{scene}: 9000 train views (4500 uniform + 4500 grazing) + 40 eval views -> {root}")

if __name__ == "__main__":
    main()
