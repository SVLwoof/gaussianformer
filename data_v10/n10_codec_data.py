"""Dense randomized-view training data for N=10 (codec method at multi-object scale).
Per scene: input = recovered 20k h5; targets = FULL splat rasterized at 150 random cams
(az free, el -15..55, radius 1.05-2.55). Standard 14-view orbit renders stay as the
novel-view eval (in-range but never trained on)."""
from __future__ import annotations
import json, numpy as np, torch, h5py, imageio.v3 as iio
from pathlib import Path
from data_external.orbit import look_at_blender, c2w_to_viewmat
from data_v10.prune_recovery import load_full, rasterize

device = "cuda"; RES, FOV = 512, 45.0
UP = np.array([0., 1., 0.], np.float32)
focal = 0.5*RES/np.tan(0.5*np.radians(FOV))
K = np.array([[focal,0,RES/2],[0,focal,RES/2],[0,0,1]], np.float32)
scenes = json.loads(Path("data_v10/nsweep/n10_scenes.json").read_text())
root = Path("experiments/overfit/data/n10_codec"); (root/"h5s").mkdir(parents=True, exist_ok=True); (root/"renders").mkdir(exist_ok=True)
rng = np.random.default_rng(77)
for s in scenes:
    full = load_full(Path(f"data_v10/full_h5s/scene_{s:04d}.h5"))
    ft = {k: torch.as_tensor(v, device=device) for k, v in dict(
        means=full["means"], quats=full["rotations"], scales=full["scales"],
        colors=full["colors"], opacities=full["opacities"].reshape(-1)).items()}
    c2ws = []
    for _ in range(150):
        az = rng.uniform(0, 2*np.pi); el = rng.uniform(np.radians(-15), np.radians(55))
        r = rng.uniform(1.05, 2.55)
        eye = np.array([r*np.cos(el)*np.cos(az), r*np.sin(el), r*np.cos(el)*np.sin(az)], np.float32)
        c2ws.append(look_at_blender(eye, np.zeros(3), UP))
    c2ws = np.stack(c2ws)
    vm = torch.as_tensor(np.stack([c2w_to_viewmat(c) for c in c2ws]), device=device)
    Ks = torch.as_tensor(np.tile(K[None], (150,1,1)), device=device)
    for i in range(0, 150, 8):
        for j, im in enumerate(rasterize(ft, vm, Ks, list(range(i, min(i+8,150))), RES)):
            iio.imwrite(root/"renders"/f"scene_{s:04d}_view_{i+j}.png", (im.clamp(0,1).cpu().numpy()*255).astype(np.uint8))
    with h5py.File(f"data_v10/h5s_20k_rec/scene_{s:04d}.h5", "r") as fsrc, \
         h5py.File(root/"h5s"/f"scene_{s:04d}.h5", "w") as f:
        for k in ("means","scales","rotations","colors","opacities"): f.create_dataset(k, data=np.array(fsrc[k]))
        f.create_dataset("c2w", data=c2ws); f.create_dataset("fov", data=np.full(150, FOV, np.float32))
    print(f"scene_{s:04d}: 150 views", flush=True)
print("N10_CODEC_DATA_DONE")
