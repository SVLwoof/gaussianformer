"""V10 data gen: per-object Objaverse_Splats -> FULL (un-pruned) H5 + GT renders.

Differs from data_v9/process_objaverse.py in two ways:
  1. ROTATION FIX: applies a fixed -90 deg about X (Z-up -> Y-up) to means + quaternions in
     normalisation, so objects stand upright in our Y-up orbit rig (the source is Z-up; the
     v9 pipeline omitted this -> objects rendered lying on their side). Verified by tmp/rotation_probe.
  2. NO PRUNING: stores ALL ~50k Gaussians (SH f_rest already dropped -- we render sh_degree=None),
     so the better-pruning exploration can run on the full splats later. GT renders are from the
     full splat (the real target), identical for any future N.

Output (split=train): {out}/full_h5s/scene_NNNN.h5  +  {out}/renders/scene_NNNN_view_V.png
        (split=val):   {out}/full_h5s_val/...        +  {out}/renders_val/...

  uv run --frozen python -m data_v10.process_full --object_list data_v10/object_list_train.json --split train
"""
from __future__ import annotations
import argparse, io, json, time, zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import gsplat
import roma
import h5py
import imageio.v3 as iio
from plyfile import PlyData
from huggingface_hub import hf_hub_download

from data_external.orbit import make_orbit_views, look_at_blender

REPO_ID = "ShapeSplats/Objaverse_Splats"
C0 = 0.28209479177387814
# Z-up -> Y-up: -90 deg about X. Verified upright in tmp/rotation_probe.png (Rx-90).
R_FIX = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--object_list", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, default=Path("data_v10"))
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--n_views", type=int, default=14)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--orbit_radius", type=float, default=1.7)
    p.add_argument("--orbit_fov_deg", type=float, default=45.0)
    p.add_argument("--max_opacity_min", type=float, default=0.4)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=-1)
    p.add_argument("--rm_zips", action="store_true",
                   help="Delete each chunk zip after use (bounds transient disk). OFF by "
                   "default so parallel shards can safely share cached chunks without a "
                   "delete-vs-read race.")
    return p.parse_args()


def load_normalized_rot(ply_bytes: bytes) -> dict | None:
    """Decode, center, ROTATE (Rx-90, Z-up->Y-up), scale. Return None on quality-fail."""
    v = PlyData.read(io.BytesIO(ply_bytes))["vertex"]
    means = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float32)
    scales = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], -1)).astype(np.float32)
    quats = np.stack([v[f"rot_{i}"] for i in range(4)], -1).astype(np.float32)  # wxyz
    quats /= (np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-9)
    colors = np.clip(0.5 + C0 * np.stack([v[f"f_dc_{i}"] for i in range(3)], -1), 0, 1).astype(np.float32)
    opacities = (1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], np.float32)))).astype(np.float32)

    means = means - np.median(means, axis=0)            # center first
    # rotate about origin: positions by R, each Gaussian's frame quat by R
    means = (means @ R_FIX.T).astype(np.float32)
    qx = torch.from_numpy(quats)[:, [1, 2, 3, 0]]        # wxyz -> xyzw
    M = roma.unitquat_to_rotmat(qx)
    quats = roma.rotmat_to_unitquat(torch.from_numpy(R_FIX) @ M)[:, [3, 0, 1, 2]].numpy()  # -> wxyz

    aabb = float(np.abs(means).max())
    if aabb < 1e-4:
        return None
    ns = 0.45 / aabb
    means = (means * ns).astype(np.float32)
    scales = (scales * ns).astype(np.float32)
    return dict(means=means, scales=scales, quats=quats, colors=colors, opacities=opacities)


def build_c2w(n_views, radius, fov_deg):
    c2w = []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        elev = 0.4 * radius if i % 2 == 0 else -0.1 * radius
        eye = np.array([radius * np.cos(theta), elev, radius * np.sin(theta)], np.float32)
        c2w.append(look_at_blender(eye, np.zeros(3, np.float32), up=np.array([0, 1, 0], np.float32)))
    return np.stack(c2w).astype(np.float32), np.full(n_views, fov_deg, np.float32)


def render_full(arr, viewmats, Ks, res, device):
    t = {k: torch.from_numpy(np.ascontiguousarray(arr[k])).to(device) for k in ("means", "scales", "quats", "colors", "opacities")}
    op = t["opacities"].squeeze(-1) if t["opacities"].ndim == 2 else t["opacities"]
    imgs = []
    for vmi in range(viewmats.shape[0]):
        with torch.no_grad():
            img, _, _ = gsplat.rasterization(
                means=t["means"], quats=t["quats"], scales=t["scales"], opacities=op, colors=t["colors"],
                viewmats=viewmats[vmi:vmi+1], Ks=Ks[vmi:vmi+1], width=res, height=res,
                sh_degree=None, eps2d=0.3, render_mode="RGB", near_plane=0.01, packed=True)
        imgs.append(np.clip(img[0].cpu().numpy(), 0, 1))
    return imgs


def write_full_h5(arr, h5_path, c2w, fov):
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("means", data=arr["means"])
        f.create_dataset("scales", data=arr["scales"])
        f.create_dataset("rotations", data=arr["quats"])
        f.create_dataset("colors", data=arr["colors"])
        f.create_dataset("opacities", data=arr["opacities"][:, None])
        f.create_dataset("c2w", data=c2w)
        f.create_dataset("fov", data=fov)


def main():
    args = parse_args()
    h5_dir = args.out_dir / ("full_h5s" if args.split == "train" else "full_h5s_val")
    renders_dir = args.out_dir / ("renders" if args.split == "train" else "renders_val")
    h5_dir.mkdir(parents=True, exist_ok=True)
    renders_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    objects = json.loads(args.object_list.read_text())
    objects = [o for o in objects if o["scene_idx"] >= args.start_idx and (args.end_idx < 0 or o["scene_idx"] < args.end_idx)]
    print(f"Loaded {len(objects)} objects from {args.object_list} (split={args.split})", flush=True)

    by_chunk = defaultdict(list)
    for o in objects:
        by_chunk[o["chunk"]].append(o)

    vm_np, Ks_np = make_orbit_views(args.n_views, args.orbit_radius, args.orbit_fov_deg, args.resolution, up_axis="y")
    viewmats = torch.from_numpy(vm_np).to(device)
    Ks = torch.from_numpy(Ks_np).to(device)
    c2w_all, fov_all = build_c2w(args.n_views, args.orbit_radius, args.orbit_fov_deg)

    done = low_op = failed = 0
    t0 = time.time()
    for chunk in sorted(by_chunk):
        objs = by_chunk[chunk]
        print(f"\n=== Chunk {chunk} ({len(objs)} objects) ===", flush=True)
        zip_path = hf_hub_download(REPO_ID, f"{chunk}.zip", repo_type="dataset")
        with zipfile.ZipFile(zip_path) as zf:
            for obj in objs:
                scene = f"scene_{obj['scene_idx']:04d}"
                h5_out = h5_dir / f"{scene}.h5"
                if h5_out.exists() and (renders_dir / f"{scene}_view_0.png").exists():
                    done += 1
                    continue
                try:
                    ply_bytes = zf.read(f"{chunk}/{obj['uid']}/ckpts/point_cloud_15000.ply")
                except KeyError:
                    print(f"  [{scene}] MISSING ply"); failed += 1; continue
                arr = load_normalized_rot(ply_bytes)
                if arr is None:
                    print(f"  [{scene}] degenerate"); failed += 1; continue
                if float(arr["opacities"].max()) < args.max_opacity_min:
                    low_op += 1; continue
                for vi, img in enumerate(render_full(arr, viewmats, Ks, args.resolution, device)):
                    iio.imwrite(renders_dir / f"{scene}_view_{vi}.png", (img * 255).astype(np.uint8))
                write_full_h5(arr, h5_out, c2w_all, fov_all)
                done += 1
                if done % 50 == 0:
                    rate = done / max(time.time() - t0, 1e-6)
                    print(f"  {scene} done ({done} total | {rate:.1f}/s | low_op {low_op} fail {failed})", flush=True)
        if args.rm_zips:
            try:
                Path(zip_path).unlink()
            except OSError:
                pass
        print(f"  chunk {chunk} complete{' (zip removed)' if args.rm_zips else ''}", flush=True)

    print(f"\nDONE_PROCESS split={args.split}: {done} written, {low_op} low-opacity skipped, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
