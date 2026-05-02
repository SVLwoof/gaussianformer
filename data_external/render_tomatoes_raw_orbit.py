"""Render the full (un-pruned) tomatoes PLY on the v7-truck 14-view orbit rig.

Normalization (must match every downstream tool):
  - center on median position
  - rotate 180 deg about X (positions (x,y,z) -> (x,-y,-z), quats updated):
    the source PLY has gravity-up == -Y (verified from the orbit's first
    two views: dome of the bowl appeared at +Y, tomatoes at -Y). After
    this proper rotation the scene matches the standard +Y-up convention
    and the orbit rig with up_axis="y" frames the bowl right-side-up.
  - scale by NORM_SCALE so the long axis fits ~[-0.45, 0.45]

Cameras: 14 orbit views via make_orbit_views(up_axis="y"), radius=1.7,
FOV=45 deg, 512 px. Same rig as the v7 truck comparison so eventual
GaussianFormer + pruned-gsplat outputs share the same coords.
"""
from pathlib import Path

import gsplat
import imageio
import numpy as np
import torch
from plyfile import PlyData

from data_external.orbit import make_orbit_views


PLY = Path("data_external/tomatoes/raw.ply")
OUT_DIR = Path("data_external/tomatoes/renders/gsplat_full")
N_VIEWS = 14
RESOLUTION = 512
FOV_DEG = 45.0
RADIUS = 1.7
NORM_SCALE = 2.0


def main():
    device = torch.device("cuda")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {PLY}...")
    plydata = PlyData.read(str(PLY))
    v = plydata["vertex"]
    n = v.count
    print(f"  {n:,} Gaussians")

    means = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    scales_log = np.stack([v[f"scale_{i}"] for i in range(3)], axis=-1).astype(np.float32)
    quats = np.stack([v[f"rot_{i}"] for i in range(4)], axis=-1).astype(np.float32)
    quats = quats / (np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-9)
    f_dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], axis=-1).astype(np.float32)
    C0 = 0.28209479177387814
    colors = np.clip(0.5 + C0 * f_dc, 0, 1).astype(np.float32)
    opacity_logit = np.asarray(v["opacity"], dtype=np.float32)
    opacities = (1.0 / (1.0 + np.exp(-opacity_logit))).astype(np.float32)

    median = np.median(means, axis=0)
    means = means - median
    # 180 deg rotation about X axis: (x, y, z) -> (x, -y, -z)
    means[:, 1] *= -1.0
    means[:, 2] *= -1.0
    # Quaternion composition q_new = q_rot * q_old, q_rot = (w=0, x=1, y=0, z=0).
    # Hamilton product yields q_new = (-x, w, -z, y) for q_old = (w, x, y, z).
    w, x, y, z = quats[:, 0].copy(), quats[:, 1].copy(), quats[:, 2].copy(), quats[:, 3].copy()
    quats = np.stack([-x, w, -z, y], axis=-1).astype(np.float32)
    means = means * NORM_SCALE
    scales_linear = np.exp(scales_log) * NORM_SCALE
    print(f"  centered on median {median.tolist()}, rotated 180deg about X, scaled x{NORM_SCALE}")
    print(f"  normalized means range: [{means.min():.3f}, {means.max():.3f}]")
    print(f"  normalized scales range: [{scales_linear.min():.5f}, {scales_linear.max():.5f}]")

    means_t = torch.from_numpy(means).to(device)
    scales_t = torch.from_numpy(scales_linear).to(device)
    quats_t = torch.from_numpy(quats).to(device)
    opacities_t = torch.from_numpy(opacities).to(device)
    colors_t = torch.from_numpy(colors).to(device)

    viewmats_np, Ks_np = make_orbit_views(N_VIEWS, RADIUS, FOV_DEG, RESOLUTION, up_axis="y")
    viewmats_t = torch.from_numpy(viewmats_np).to(device)
    Ks_t = torch.from_numpy(Ks_np).to(device)

    print(f"\nRendering {N_VIEWS} orbit views @ {RESOLUTION}px, radius={RADIUS}, fov={FOV_DEG}deg, up=Y...")
    rendered = []
    for v_idx in range(N_VIEWS):
        with torch.no_grad():
            img, _, _ = gsplat.rasterization(
                means=means_t, quats=quats_t, scales=scales_t,
                opacities=opacities_t, colors=colors_t,
                viewmats=viewmats_t[v_idx:v_idx+1], Ks=Ks_t[v_idx:v_idx+1],
                width=RESOLUTION, height=RESOLUTION,
                sh_degree=None, eps2d=0.3, render_mode="RGB",
                near_plane=0.01, packed=True,
            )
        rgb = img[0].clamp(0, 1).cpu().numpy()
        out_path = OUT_DIR / f"view_{v_idx:02d}.png"
        imageio.v3.imwrite(out_path, (rgb * 255).astype(np.uint8))
        rendered.append((rgb * 255).astype(np.uint8))
        print(f"  view {v_idx:2d} -> {out_path.name}")

    # 7x2 overview (2 rows of 7) so all 14 views are inspectable in one shot.
    rows = [np.concatenate(rendered[i*7:(i+1)*7], axis=1) for i in range(2)]
    overview = np.concatenate(rows, axis=0)
    overview_path = OUT_DIR / "overview_7x2.png"
    imageio.v3.imwrite(overview_path, overview)
    print(f"\n7x2 overview -> {overview_path}")
    print(f"Wrote {N_VIEWS} renders + overview to {OUT_DIR}/")


if __name__ == "__main__":
    main()
