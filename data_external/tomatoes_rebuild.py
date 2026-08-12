"""Rebuild tomatoes with the MODERN pipeline: normalized.ply -> significance prune 50k->20k ->
gsplat recovery FT -> recovered h5 (+ rec-GT renders). One-off for the 4-way GT | rec-GT |
overfit | V17 comparison. Reuses run_scene's normalization output (normalized.ply is already
centered/rotated/scaled, activation-space handled below) and prune_recovery's exact score+recover.
"""
from __future__ import annotations
import numpy as np, torch, h5py, imageio.v3 as iio
from pathlib import Path
from plyfile import PlyData
from data_v10.prune_recovery import significance_topk, recover, rasterize, to_dev
from data_external.orbit import make_orbit_views

D = Path("data_external/tomatoes")
device = "cuda"
v = PlyData.read(str(D / "normalized.ply"))["vertex"]
C0 = 0.28209479177387814
a = dict(
    means=np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float32),
    scales=np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], -1).astype(np.float32)),
    rotations=np.stack([v[f"rot_{i}"] for i in range(4)], -1).astype(np.float32),
    colors=np.clip(0.5 + C0 * np.stack([v[f"f_dc_{i}"] for i in range(3)], -1), 0, 1).astype(np.float32),
    opacities=(1 / (1 + np.exp(-np.asarray(v["opacity"], np.float32)))),
)
# normalized.ply from write_pruned_ply may already store activation-space scales/opacity;
# detect: activation-space scales are tiny positives; log-space are negative.
raw_scales = np.stack([v[f"scale_{i}"] for i in range(3)], -1).astype(np.float32)
if raw_scales.min() >= 0:            # already activation space -> undo exp/sigmoid assumptions
    a["scales"] = raw_scales
    a["opacities"] = np.asarray(v["opacity"], np.float32)
print(f"{len(a['means'])} gaussians, scale range {a['scales'].min():.2e}..{a['scales'].max():.2e}, "
      f"opacity range {a['opacities'].min():.3f}..{a['opacities'].max():.3f}")
a["opacities"] = a["opacities"].reshape(-1)

# cameras: same 14-view orbit as the existing h5 (copy c2w/fov verbatim)
with h5py.File(D / "h5" / "tomatoes_n20000.h5", "r") as f:
    c2w, fov = np.array(f["c2w"]), np.array(f["fov"])
# gsplat viewmats for scoring/recovery targets: 64-view orbit like prune_recovery
vm64, K64 = make_orbit_views(64, 1.7, 45.0, 512, up_axis="y")
vm64, K64 = torch.from_numpy(vm64).to(device), torch.from_numpy(K64).to(device)

pruned = significance_topk(a, vm64, K64, 20000, device)
full_t = to_dev({"means": a["means"], "scales": a["scales"], "quats": a["rotations"],
                 "colors": a["colors"], "opacities": a["opacities"]}, device)
targets = torch.cat([rasterize(full_t, vm64, K64, list(range(i, min(i+8, 64))), 512)
                     for i in range(0, 64, 8)])
pruned["opacities"] = np.asarray(pruned["opacities"]).reshape(-1)
pruned = {k: (v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)) for k, v in pruned.items()}
rec = recover(pruned, targets, vm64, K64, 1500, device)
out = D / "h5" / "tomatoes_rec20000.h5"
with h5py.File(out, "w") as f:
    f.create_dataset("means", data=rec["means"]); f.create_dataset("scales", data=rec["scales"])
    f.create_dataset("rotations", data=rec["rotations"]); f.create_dataset("colors", data=rec["colors"])
    f.create_dataset("opacities", data=rec["opacities"].reshape(-1, 1))
    f.create_dataset("c2w", data=c2w); f.create_dataset("fov", data=fov)
print(f"wrote {out}")
# rec-GT renders on the 14 eval views
vm14, K14 = make_orbit_views(14, 1.7, 45.0, 512, up_axis="y")
vm14, K14 = torch.from_numpy(vm14).to(device), torch.from_numpy(K14).to(device)
rp = {"means": torch.as_tensor(rec["means"], device=device),
      "quats": torch.as_tensor(rec["rotations"], device=device),
      "scales": torch.as_tensor(rec["scales"], device=device),
      "colors": torch.as_tensor(rec["colors"], device=device),
      "opacities": torch.as_tensor(rec["opacities"], device=device).reshape(-1)}
imgs = torch.cat([rasterize(rp, vm14, K14, list(range(i, min(i+7, 14))), 512) for i in (0, 7)])
od = D / "renders" / "gsplat_rec20000"; od.mkdir(parents=True, exist_ok=True)
for i in range(14):
    iio.imwrite(od / f"view_{i:02d}.png", (imgs[i].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
print(f"wrote 14 rec-GT renders -> {od}")
