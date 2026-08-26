"""Rate-distortion pilot: at what splat budget K does the trained codec beat rasterization?

For one object and its trained codec checkpoint, re-prune the FULL splat to each K with the
fleet recipe (significance top-K on a 64-view orbit + 1500-iter recovery), then on the frozen
eval sets compare the model (fed the K-splat, no retraining) with rasterizing that same
K-splat -- both against renders of the full splat.

Env: SCENE, CKPT, KS (comma list, default 20000,10000,5000,2500), FLIP_X=1 for superspl.at scans.
Writes experiments/overfit/data/codec_scaleout/<SCENE>/ksweep_<tag>.json and a CSV row per K.
"""
from __future__ import annotations
import json, os, numpy as np, torch, h5py
from pathlib import Path
from render_compare import load_model, ModelSpec, load_gt
from infer_gaussian import load_single_gaussian_h5_data
from data_external.orbit import c2w_to_viewmat, make_orbit_views
from data_v10.prune_recovery import significance_topk, recover, rasterize, to_dev

SCENE = os.environ["SCENE"]; CKPT = Path(os.environ["CKPT"])
KS = [int(k) for k in os.environ.get("KS", "20000,10000,5000,2500").split(",")]
TAG = os.environ.get("TAG") or CKPT.parent.name.replace("checkpoints_codec_so_", "")
D = Path(f"experiments/overfit/data/codec_scaleout/{SCENE}")
device = "cuda"; RES, FOV, RADIUS = 512, 45.0, 1.7
torch.manual_seed(0)

if SCENE.startswith("scene_"):
    with h5py.File(f"data_v10/full_h5s_val/{SCENE}.h5") as f:
        full = {k: np.array(f[k], np.float32) for k in ("means", "scales", "rotations", "colors", "opacities")}
    full["opacities"] = full["opacities"].reshape(-1)
else:
    from data_external.prep_external_codec import load_normalized
    full = load_normalized(SCENE, flip_x=bool(int(os.environ.get("FLIP_X", "1"))))
full_t = to_dev(dict(means=full["means"], scales=full["scales"], quats=full["rotations"],
                     colors=full["colors"], opacities=full["opacities"]), device)
print(f"{SCENE}: full splat N={len(full['means'])}", flush=True)

vm64, K64 = make_orbit_views(64, RADIUS, FOV, RES, up_axis="y")
vm64, K64 = torch.from_numpy(vm64).to(device), torch.from_numpy(K64).to(device)
targets = torch.cat([rasterize(full_t, vm64, K64, list(range(i, min(i + 8, 64))), RES) for i in range(0, 64, 8)])

pipe = load_model(ModelSpec(ckpt=CKPT, label="codec", pe_type="rope"), device)
focal = 0.5 * RES / np.tan(0.5 * np.radians(FOV))
Kmat = torch.as_tensor(np.array([[focal, 0, RES / 2], [0, focal, RES / 2], [0, 0, 1]], np.float32), device=device)
def psnr(a, b): return 10 * np.log10(1 / max(((a - b) ** 2).mean(), 1e-12))

out = {"scene": SCENE, "ckpt": str(CKPT), "full_n": int(len(full["means"])), "K": {}}
tmp = Path("tmp/k_sweep"); tmp.mkdir(parents=True, exist_ok=True)
for K in KS:
    pruned = significance_topk(full, vm64, K64, K, device)
    pruned = {k: (v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)) for k, v in pruned.items()}
    pruned["opacities"] = pruned["opacities"].reshape(-1)
    rec = recover(pruned, targets, vm64, K64, 1500, device)
    h5 = tmp / f"{SCENE}_{K}.h5"
    with h5py.File(h5, "w") as f:
        for k in ("means", "scales", "rotations", "colors"): f.create_dataset(k, data=rec[k])
        f.create_dataset("opacities", data=np.asarray(rec["opacities"]).reshape(-1, 1))
        f.create_dataset("c2w", data=np.eye(4, dtype=np.float32)[None]); f.create_dataset("fov", data=np.array([FOV], np.float32))
    data = load_single_gaussian_h5_data(h5)
    recp = to_dev(dict(means=rec["means"], quats=rec["rotations"], scales=rec["scales"],
                       colors=rec["colors"], opacities=np.asarray(rec["opacities"]).reshape(-1)), device)
    res = {}
    for name in ("novel_rand", "novel_close", "novel_far"):
        c2ws = np.load(D / f"codec_eval_{name}_c2w.npy")
        vm = torch.as_tensor(np.stack([c2w_to_viewmat(c) for c in c2ws]), device=device)
        Ks_ = Kmat[None].expand(len(c2ws), 3, 3)
        pm, pr = [], []
        for i in range(len(c2ws)):
            gt = load_gt(D / "renders" / f"codec_eval_{name}" / f"gt_view_{i}.png", RES)
            rg = rasterize(recp, vm, Ks_, [i], RES)[0].clamp(0, 1).cpu().numpy()
            with torch.no_grad():
                o = pipe(gaussians=data["gaussians"][None].to(device), mask=data["mask"][None].to(device),
                         c2w=torch.as_tensor(c2ws[i], device=device)[None][None],
                         fov=torch.tensor([[FOV]], device=device), resolution=RES, torch_dtype=torch.bfloat16)
            md = np.clip(o[0, 0].cpu().float().numpy(), 0, 1)
            pm.append(psnr(md, gt)); pr.append(psnr(rg, gt))
        pm, pr = np.array(pm), np.array(pr)
        res[name] = dict(model=float(pm.mean()), raster=float(pr.mean()), delta=float((pm - pr).mean()),
                         won=f"{int((pm > pr).sum())}/{len(pm)}")
    w = {"novel_rand": 24, "novel_close": 8, "novel_far": 8}
    res["avg_delta"] = sum(res[n]["delta"] * w[n] for n in w) / 40
    res["avg_model"] = sum(res[n]["model"] * w[n] for n in w) / 40
    res["avg_raster"] = sum(res[n]["raster"] * w[n] for n in w) / 40
    out["K"][str(K)] = res
    print(f"{SCENE} K={K:6d} ({out['full_n']/K:5.1f}x): model {res['avg_model']:.2f}  raster {res['avg_raster']:.2f}  "
          f"delta {res['avg_delta']:+.2f}  | rand {res['novel_rand']['delta']:+.2f} close {res['novel_close']['delta']:+.2f} "
          f"far {res['novel_far']['delta']:+.2f}", flush=True)
    (D / f"ksweep_{TAG}.json").write_text(json.dumps(out, indent=1))
    h5.unlink()
print("done", D / f"ksweep_{TAG}.json")
