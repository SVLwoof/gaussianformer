"""Randomized-view training data for the tomato 'neural codec' overfit.

Goal (2026-08-13): beat rec-GT on EVERY novel view of one object -- if a single-object model
reads/renders better than rasterizing its own compressed splat, it is a useful per-scene codec.
Train views are RANDOMIZED over azimuth, elevation AND radius (zoom 1.35-2.15); eval views come
from a disjoint seed plus radius EXTRAPOLATION (1.15, 2.45) so view-interpolation alone cannot
explain success. Targets rasterized from the FULL 219k splat; input = recovered 20k.
"""
from __future__ import annotations
import numpy as np, torch, h5py, imageio.v3 as iio
from pathlib import Path
from data_external.orbit import look_at_blender, c2w_to_viewmat
from data_v10.prune_recovery import rasterize
from plyfile import PlyData
# inline ply loader (importing tomatoes_rebuild would EXECUTE its module-level pipeline)
_v = PlyData.read(str(Path("data_external/tomatoes/normalized.ply")))["vertex"]
_C0 = 0.28209479177387814
_raw_scales = np.stack([_v[f"scale_{i}"] for i in range(3)], -1).astype(np.float32)
FULL = dict(
    means=np.stack([_v["x"], _v["y"], _v["z"]], -1).astype(np.float32),
    scales=_raw_scales if _raw_scales.min() >= 0 else np.exp(_raw_scales),
    rotations=np.stack([_v[f"rot_{i}"] for i in range(4)], -1).astype(np.float32),
    colors=np.clip(0.5 + _C0 * np.stack([_v[f"f_dc_{i}"] for i in range(3)], -1), 0, 1).astype(np.float32),
    opacities=(np.asarray(_v["opacity"], np.float32) if _raw_scales.min() >= 0
               else 1/(1+np.exp(-np.asarray(_v["opacity"], np.float32)))),
)

D = Path("data_external/tomatoes"); device = "cuda"; RES, FOV = 512, 45.0
UP = np.array([0., 1., 0.], np.float32)
focal = 0.5 * RES / np.tan(0.5 * np.radians(FOV))
K = np.array([[focal, 0, RES/2], [0, focal, RES/2], [0, 0, 1]], np.float32)

def cams(n, seed, radii):
    rng = np.random.default_rng(seed)
    c2ws = []
    for _ in range(n):
        az = rng.uniform(0, 2*np.pi); el = rng.uniform(np.radians(-15), np.radians(55))
        r = rng.uniform(*radii) if isinstance(radii, tuple) else radii
        eye = np.array([r*np.cos(el)*np.cos(az), r*np.sin(el), r*np.cos(el)*np.sin(az)], np.float32)
        c2ws.append(look_at_blender(eye, np.zeros(3), UP))
    return np.stack(c2ws)

full_t = {k: torch.as_tensor(v, device=device) for k, v in
          dict(means=FULL["means"], quats=FULL["rotations"], scales=FULL["scales"],
               colors=FULL["colors"], opacities=FULL["opacities"].reshape(-1)).items()}

def render_set(c2ws, outdir, stem):
    outdir.mkdir(parents=True, exist_ok=True)
    vm = torch.as_tensor(np.stack([c2w_to_viewmat(c) for c in c2ws]), device=device)
    Kt = torch.as_tensor(np.tile(K[None], (len(c2ws), 1, 1)), device=device)
    for i in range(0, len(c2ws), 8):
        imgs = rasterize(full_t, vm, Kt, list(range(i, min(i+8, len(c2ws)))), RES)
        for j, im in enumerate(imgs):
            iio.imwrite(outdir / f"{stem}_view_{i+j}.png", (im.clamp(0,1).cpu().numpy()*255).astype(np.uint8))

# --- train: 200 randomized views incl. zoom ---
train_c2w = cams(200, seed=11, radii=(1.35, 2.15))
troot = Path("experiments/overfit/data/tomatoes_codec")
(troot/"h5s").mkdir(parents=True, exist_ok=True)
with h5py.File(D/"h5"/"tomatoes_rec20000.h5", "r") as f:
    g = {k: np.array(f[k]) for k in ("means","scales","rotations","colors","opacities")}
with h5py.File(troot/"h5s"/"tomatoes_codec.h5", "w") as f:
    for k, v in g.items(): f.create_dataset(k, data=v)
    f.create_dataset("c2w", data=train_c2w)
    f.create_dataset("fov", data=np.full(len(train_c2w), FOV, np.float32))
render_set(train_c2w, troot/"renders", "tomatoes_codec")
print(f"train: 200 randomized views (r 1.35-2.15) -> {troot}")

# --- eval: 40 disjoint random views + zoom extrapolation ---
for name, n, seed, radii in (("novel_rand", 24, 99, (1.35, 2.15)),
                             ("novel_close", 8, 100, 1.15), ("novel_far", 8, 101, 2.45)):
    c = cams(n, seed, radii)
    np.save(D/f"codec_eval_{name}_c2w.npy", c)
    render_set(c, D/"renders"/f"codec_eval_{name}", "gt")
print("eval sets written (novel_rand / novel_close / novel_far)")
