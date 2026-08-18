"""Codec v5 training data: 9000 views = codec4's 4500 + 4500 grazing-elevation views.

Codec v4 beat rec-GT on average (+0.36 dB, 22/40 views) but its losses concentrate at
grazing angles (close v1: -3.5 dB, an edge-on plate view) and marginal rand/far views.
v5 doubles the view budget with the new half drawn from LOW elevations (-15..20 deg,
same az/radius coverage) to shore up exactly where the model loses. The old 4500 renders
are reused via relative symlinks; only the new half is rasterized (full 219k splat).
Eval sets stay frozen (codec_eval_*) so verdicts remain comparable across v1-v5.
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

full_t = {k: torch.as_tensor(v, device=device) for k, v in
          dict(means=FULL["means"], quats=FULL["rotations"], scales=FULL["scales"],
               colors=FULL["colors"], opacities=FULL["opacities"].reshape(-1)).items()}

def render_set(c2ws, outdir, stem, start=0):
    outdir.mkdir(parents=True, exist_ok=True)
    vm = torch.as_tensor(np.stack([c2w_to_viewmat(c) for c in c2ws]), device=device)
    Kt = torch.as_tensor(np.tile(K[None], (len(c2ws), 1, 1)), device=device)
    for i in range(0, len(c2ws), 8):
        imgs = rasterize(full_t, vm, Kt, list(range(i, min(i+8, len(c2ws)))), RES)
        for j, im in enumerate(imgs):
            iio.imwrite(outdir / f"{stem}_view_{start+i+j}.png", (im.clamp(0,1).cpu().numpy()*255).astype(np.uint8))

with h5py.File("experiments/overfit/data/tomatoes_codec4/h5s/tomatoes_codec4.h5", "r") as f:
    old_c2w = np.array(f["c2w"])                       # the 4500 codec4 views, verbatim
assert len(old_c2w) == 4500
new_c2w = cams(4500, seed=51, radii=(1.05, 2.55), el_range=(-15, 20))
train_c2w = np.concatenate([old_c2w, new_c2w])

troot = Path("experiments/overfit/data/tomatoes_codec5")
(troot/"h5s").mkdir(parents=True, exist_ok=True)
with h5py.File(D/"h5"/"tomatoes_rec20000.h5", "r") as f:
    g = {k: np.array(f[k]) for k in ("means","scales","rotations","colors","opacities")}
with h5py.File(troot/"h5s"/"tomatoes_codec5.h5", "w") as f:
    for k, v in g.items(): f.create_dataset(k, data=v)
    f.create_dataset("c2w", data=train_c2w)
    f.create_dataset("fov", data=np.full(len(train_c2w), FOV, np.float32))

(troot/"renders").mkdir(parents=True, exist_ok=True)
for i in range(4500):
    link = troot/"renders"/f"tomatoes_codec5_view_{i}.png"
    if not link.exists():
        link.symlink_to(f"../../tomatoes_codec4/renders/tomatoes_codec4_view_{i}.png")
render_set(new_c2w, troot/"renders", "tomatoes_codec5", start=4500)
print(f"train: 4500 reused + 4500 grazing (el -15..20, r 1.05-2.55) = 9000 -> {troot}")
