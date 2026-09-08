"""GPU check for P1's render_canvas: rasterize an n10 object with the training pipeline's canvas
helper for its own training camera and compare with the stored GT render of the FULL splat.
Same convention => high PSNR (rec-GT level, ~40+ dB whole-image); a flipped axis => ~10-15 dB.

  PYTHONPATH=. uv run --no-sync python data_v10/check_canvas.py
"""
import json
import numpy as np, torch, h5py, imageio.v3 as iio
from pathlib import Path
from gaussianformer.utils.canvas import render_canvas

scenes = json.load(open("data_v10/nsweep/n10_scenes.json"))
scene = scenes[0] if isinstance(scenes, list) else list(scenes)[0]
h5 = Path("data_v10/nsweep/n10_h5") / f"{scene}.h5"
with h5py.File(h5) as f:
    g = np.concatenate([np.array(f[k], np.float32).reshape(len(f["means"]), -1)
                        for k in ("means", "scales", "rotations", "colors", "opacities")], -1)
    c2w, fov = np.array(f["c2w"], np.float32), np.array(f["fov"], np.float32)
dev = "cuda"
gt_path = Path("data_v10/nsweep/n10_renders") / f"{scene}_view_0.png"
gt = iio.imread(gt_path).astype(np.float32) / 255
res = gt.shape[0]
G = torch.as_tensor(g, device=dev)[None]
canvas = render_canvas(G, torch.ones(1, G.shape[1], dtype=torch.bool, device=dev),
                       torch.as_tensor(c2w[:1], device=dev)[None], torch.as_tensor(fov[:1], device=dev)[None], res)
img = (10 ** canvas[0] - 1).clamp(0, 1).cpu().numpy()
gt_r = np.asarray(iio.imread(gt_path)).astype(np.float32) / 255 if gt.shape[0] == res else gt
psnr = 10 * np.log10(1 / max(((img - gt_r) ** 2).mean(), 1e-12))
flipped = 10 * np.log10(1 / max(((img[:, ::-1] - gt_r) ** 2).mean(), 1e-12))
print(f"{scene}: canvas vs GT full-splat render, view 0: PSNR {psnr:.2f} dB (horizontally flipped: {flipped:.2f})")
iio.imwrite("tmp/check_canvas.png", (np.concatenate([img, gt_r], 1) * 255).astype(np.uint8))
assert psnr > 25, "canvas convention mismatch"
print("CANVAS_OK")
