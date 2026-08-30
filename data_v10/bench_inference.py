"""Per-view inference cost: trained codec (GaussianFormer) vs gsplat rasterization of the SAME
20k recovered splat. Env: SCENE, CKPT, RES (512), N (timed views, 40). Prints a table + JSON."""
from __future__ import annotations
import json, os, time, numpy as np, torch, h5py
from pathlib import Path
from render_compare import load_model, ModelSpec
from infer_gaussian import load_single_gaussian_h5_data
from data_external.orbit import c2w_to_viewmat
from data_v10.prune_recovery import rasterize

SCENE, CKPT = os.environ["SCENE"], Path(os.environ["CKPT"])
RES, N, FOV = int(os.environ.get("RES", 512)), int(os.environ.get("N", 40)), 45.0
dev = "cuda"; D = Path(f"experiments/overfit/data/codec_scaleout/{SCENE}")
h5 = D / "h5s" / f"{SCENE}.h5"
data = load_single_gaussian_h5_data(h5)
with h5py.File(h5) as f:
    rec = {k: torch.as_tensor(np.array(f[k], np.float32), device=dev) for k in ("means", "scales", "rotations", "colors", "opacities")}
recp = dict(means=rec["means"], quats=rec["rotations"], scales=rec["scales"], colors=rec["colors"], opacities=rec["opacities"].reshape(-1))
c2ws = np.concatenate([np.load(D / f"codec_eval_{n}_c2w.npy") for n in ("novel_rand", "novel_close", "novel_far")])[:N]
vm = torch.as_tensor(np.stack([c2w_to_viewmat(c) for c in c2ws]), device=dev)
focal = 0.5 * RES / np.tan(0.5 * np.radians(FOV))
K = torch.as_tensor(np.array([[focal, 0, RES / 2], [0, focal, RES / 2], [0, 0, 1]], np.float32), device=dev)[None].expand(len(c2ws), 3, 3).contiguous()
G = data["gaussians"][None].to(dev); M = data["mask"][None].to(dev)
print(f"GPU {torch.cuda.get_device_name(0)} | {SCENE}: N_gaussians={int(M.sum())} res={RES} views={len(c2ws)}", flush=True)

def timed(fn, n, warm=5):
    for _ in range(warm): fn(0)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    ts = []
    for i in range(n):
        torch.cuda.synchronize(); t = time.perf_counter(); fn(i); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    peak = torch.cuda.max_memory_allocated() - base
    ts = np.array(ts) * 1e3
    return dict(ms_mean=float(ts.mean()), ms_median=float(np.median(ts)), ms_p95=float(np.percentile(ts, 95)), peak_extra_MB=peak / 2**20)

out = {"scene": SCENE, "ckpt": str(CKPT), "gpu": torch.cuda.get_device_name(0), "res": RES, "n_gaussians": int(M.sum())}
# --- gsplat ---
out["gsplat_1view"] = timed(lambda i: rasterize(recp, vm, K, [i], RES), len(c2ws))
out["gsplat_8view_batch"] = timed(lambda i: rasterize(recp, vm, K, list(range(0, 8)), RES), 10)
out["gsplat_8view_batch"]["ms_per_view"] = out["gsplat_8view_batch"]["ms_mean"] / 8
out["gsplat_resident_MB"] = sum(v.numel() * 4 for v in recp.values()) / 2**20

# --- model ---
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
m0 = torch.cuda.memory_allocated()
pipe = load_model(ModelSpec(ckpt=CKPT, label="codec", pe_type="rope"), dev)
out["model_params_M"] = sum(p.numel() for p in pipe.model.parameters()) / 1e6
out["model_weights_MB"] = (torch.cuda.memory_allocated() - m0) / 2**20
fovt = torch.tensor([[FOV]], device=dev)
def model_view(i, nv=1):
    c = torch.as_tensor(c2ws[i:i + nv], device=dev)[None]
    return pipe(gaussians=G, mask=M, c2w=c, fov=fovt.expand(1, nv), resolution=RES, torch_dtype=torch.bfloat16)
out["model_1view"] = timed(lambda i: model_view(i), len(c2ws))
# scene encoder alone (view-independent stage; amortisable across views of the same scene)
def scene_only(_):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        seq, vmp, pos = pipe.model.construct_sequence(G, M); pipe.model.transformer(seq, src_key_padding_mask=vmp, spatial_pos=pos)
out["model_scene_stage"] = timed(scene_only, 20)
out["model_view_stage_ms"] = out["model_1view"]["ms_mean"] - out["model_scene_stage"]["ms_mean"]
for nv in (2, 4, 8):
    try:
        r = timed(lambda i: model_view(0, nv), 10); r["ms_per_view"] = r["ms_mean"] / nv; out[f"model_{nv}view_batch"] = r
    except torch.cuda.OutOfMemoryError:
        out[f"model_{nv}view_batch"] = "OOM"; torch.cuda.empty_cache(); break
try:
    out["model_fp32_1view"] = timed(lambda i: pipe(gaussians=G, mask=M, c2w=torch.as_tensor(c2ws[i:i+1], device=dev)[None], fov=fovt, resolution=RES, torch_dtype=torch.float32), 10)
except RuntimeError as e:
    out["model_fp32_1view"] = f"n/a ({str(e)[:40]})"

g, m = out["gsplat_1view"], out["model_1view"]
print(f"\n{'':28s}{'gsplat':>14s}{'codec model':>14s}{'ratio':>10s}")
print(f"{'latency / view (ms, mean)':28s}{g['ms_mean']:14.2f}{m['ms_mean']:14.2f}{m['ms_mean']/g['ms_mean']:10.0f}x")
print(f"{'latency / view (ms, median)':28s}{g['ms_median']:14.2f}{m['ms_median']:14.2f}{m['ms_median']/g['ms_median']:10.0f}x")
print(f"{'views / s (single)':28s}{1000/g['ms_mean']:14.1f}{1000/m['ms_mean']:14.2f}")
print(f"{'peak working VRAM (MB)':28s}{g['peak_extra_MB']:14.0f}{m['peak_extra_MB']:14.0f}{m['peak_extra_MB']/max(g['peak_extra_MB'],1):10.0f}x")
print(f"{'resident (MB)':28s}{out['gsplat_resident_MB']:14.1f}{out['model_weights_MB']:14.0f}")
print(f"scene stage {out['model_scene_stage']['ms_mean']:.1f} ms (once per scene) + view stage {out['model_view_stage_ms']:.1f} ms/view; params {out['model_params_M']:.1f}M")
for k in ("gsplat_8view_batch", "model_2view_batch", "model_4view_batch", "model_8view_batch", "model_fp32_1view"):
    v = out.get(k); print(k, v if isinstance(v, str) else {kk: round(vv, 2) for kk, vv in v.items()} if v else "-")
print("JSON", json.dumps(out))
