"""V9 Phase A: per-object Objaverse_Splats -> H5 + GT renders.

Reads an object list JSON (from build_object_list.py), and for each entry:
  1. Open the chunk zip from the HF cache (cached on first download).
  2. Extract the PLY in-memory.
  3. Normalize: subtract median, scale so max(|pos|) == 0.45.
  4. Drop objects whose max-opacity is too low (filters degenerate fits).
  5. Score with LightGaussian on a 64-view orbit (radius 1.7, FOV 45deg, +Y up).
  6. Top-N prune to target_n Gaussians (default 5000).
  7. Render 14 orbit views @ 512px from the FULL 50K Gaussians (the GT) and
     save as uint8 PNG. Matches v6 GT format (gsplat -> clamp -> uint8 -> PNG,
     no tonemap, since 3DGS source already trained on LDR images).
  8. Save H5 with means/scales/rotations/colors/opacities + c2w/fov for the
     same 14 views. H5 stores the PRUNED 5K Gaussians; renders are from the
     full 50K. The model learns to fill in detail it doesn't have via LPIPS.

Output layout:
  {out_dir}/h5s/scene_NNNN.h5
  {out_dir}/renders/scene_NNNN_view_V.png  (V = 0..13)
  {out_dir}/metadata.json   (scene_idx -> uid, chunk, psnr, lpips, caption)

Usage:
  python -m data_v9.process_objaverse \\
      --object_list data_v9/object_list_train.json \\
      --out_dir data_v9 --split train

  python -m data_v9.process_objaverse \\
      --object_list data_v9/object_list_val.json \\
      --out_dir data_v9 --split val
"""
from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import gsplat
import h5py
import imageio
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from plyfile import PlyData

from data_external.orbit import look_at_blender, make_orbit_views


REPO_ID = "ShapeSplats/Objaverse_Splats"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--object_list", type=Path, required=True,
                   help="JSON list from build_object_list.py")
    p.add_argument("--out_dir", type=Path, default=Path("data_v9"))
    p.add_argument("--split", choices=["train", "val"], required=True,
                   help="Determines output subdir names: h5s/renders for train, "
                   "h5s_val/renders_val for val.")
    p.add_argument("--target_n", type=int, default=5000)
    p.add_argument("--n_views", type=int, default=14)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--orbit_radius", type=float, default=1.7)
    p.add_argument("--orbit_fov_deg", type=float, default=45.0)
    p.add_argument("--score_views", type=int, default=64)
    p.add_argument("--score_resolution", type=int, default=256,
                   help="Lower than render resolution for speed; doesn't affect quality "
                   "of the top-N selection materially.")
    p.add_argument("--gamma", type=float, default=0.1, help="LightGaussian volume exponent.")
    p.add_argument("--max_opacity_min", type=float, default=0.4,
                   help="Drop objects whose maximum Gaussian opacity is below this. "
                   "Filters degenerate ghost-fits (smoke test row 3, sushi).")
    p.add_argument("--start_idx", type=int, default=0,
                   help="Skip entries with scene_idx < start_idx (resume support).")
    p.add_argument("--end_idx", type=int, default=-1,
                   help="Stop at this scene_idx (exclusive). -1 = process to end.")
    return p.parse_args()


def load_normalized(ply_bytes: bytes) -> dict | None:
    """Decode + normalize PLY. Return None on quality-fail."""
    plydata = PlyData.read(io.BytesIO(ply_bytes))
    v = plydata["vertex"]
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
    centered = means - median
    aabb_max = float(np.abs(centered).max())
    if aabb_max < 1e-4:
        return None
    norm_scale = 0.45 / aabb_max
    means_n = (centered * norm_scale).astype(np.float32)
    scales_n = (np.exp(scales_log) * norm_scale).astype(np.float32)

    return {
        "means": means_n,
        "scales": scales_n,
        "quats": quats,
        "colors": colors,
        "opacities": opacities,
        "aabb_max_raw": aabb_max,
        "norm_scale": norm_scale,
    }


def lightgaussian_scores(arr: dict, viewmats: torch.Tensor, Ks: torch.Tensor,
                         resolution: int, gamma: float, device: torch.device) -> np.ndarray:
    means = torch.from_numpy(arr["means"]).to(device)
    scales = torch.from_numpy(arr["scales"]).to(device)
    quats = torch.from_numpy(arr["quats"]).to(device)
    opacities = torch.from_numpy(arr["opacities"]).to(device)
    colors = torch.from_numpy(arr["colors"]).to(device)
    n = means.shape[0]
    score = torch.zeros(n, device=device, dtype=torch.float64)

    for v in range(viewmats.shape[0]):
        with torch.no_grad():
            _, _, info = gsplat.rasterization(
                means=means, quats=quats, scales=scales,
                opacities=opacities, colors=colors,
                viewmats=viewmats[v:v + 1], Ks=Ks[v:v + 1],
                width=resolution, height=resolution,
                sh_degree=None, eps2d=0.3, render_mode="RGB",
                near_plane=0.01, packed=True,
            )
        gids = info["gaussian_ids"]
        contrib = (
            info["opacities"].double()
            * info["radii"][:, 0].double()
            * info["radii"][:, 1].double()
        )
        score.scatter_add_(0, gids, contrib)

    max_scale = scales.max(dim=1).values.double()
    return (score * (max_scale ** gamma)).cpu().numpy()


def render_full(arr: dict, viewmats: torch.Tensor, Ks: torch.Tensor,
                resolution: int, device: torch.device) -> np.ndarray:
    means = torch.from_numpy(arr["means"]).to(device)
    scales = torch.from_numpy(arr["scales"]).to(device)
    quats = torch.from_numpy(arr["quats"]).to(device)
    opacities = torch.from_numpy(arr["opacities"]).to(device)
    colors = torch.from_numpy(arr["colors"]).to(device)
    n_views = viewmats.shape[0]
    out = np.empty((n_views, resolution, resolution, 3), dtype=np.float32)
    for v in range(n_views):
        with torch.no_grad():
            img, _, _ = gsplat.rasterization(
                means=means, quats=quats, scales=scales,
                opacities=opacities, colors=colors,
                viewmats=viewmats[v:v + 1], Ks=Ks[v:v + 1],
                width=resolution, height=resolution,
                sh_degree=None, eps2d=0.3, render_mode="RGB",
                near_plane=0.01, packed=True,
            )
        out[v] = img[0].clamp(0, 1).cpu().numpy()
    return out


def write_h5(arr: dict, keep_idx: np.ndarray, h5_path: Path,
             c2w_all: np.ndarray, fov_all: np.ndarray) -> None:
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("means", data=arr["means"][keep_idx])
        f.create_dataset("scales", data=arr["scales"][keep_idx])
        f.create_dataset("rotations", data=arr["quats"][keep_idx])
        f.create_dataset("colors", data=arr["colors"][keep_idx])
        f.create_dataset("opacities", data=arr["opacities"][keep_idx][:, None])
        f.create_dataset("c2w", data=c2w_all)
        f.create_dataset("fov", data=fov_all)


def build_c2w(n_views: int, radius: float, fov_deg: float) -> tuple[np.ndarray, np.ndarray]:
    c2w_list = []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        elev = 0.4 * radius if i % 2 == 0 else -0.1 * radius
        eye = np.array([radius * np.cos(theta), elev, radius * np.sin(theta)],
                       dtype=np.float32)
        c2w_list.append(look_at_blender(eye, np.zeros(3, dtype=np.float32),
                                          up=np.array([0, 1, 0], dtype=np.float32)))
    c2w_all = np.stack(c2w_list).astype(np.float32)
    fov_all = np.full(n_views, fov_deg, dtype=np.float32)
    return c2w_all, fov_all


def main():
    args = parse_args()
    h5_dir = args.out_dir / ("h5s" if args.split == "train" else "h5s_val")
    renders_dir = args.out_dir / ("renders" if args.split == "train" else "renders_val")
    metadata_path = args.out_dir / (
        "metadata_train.json" if args.split == "train" else "metadata_val.json"
    )
    h5_dir.mkdir(parents=True, exist_ok=True)
    renders_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    objects = json.loads(args.object_list.read_text())
    print(f"Loaded {len(objects)} objects from {args.object_list}")

    # Group by chunk so we open each zip once.
    by_chunk: dict[str, list[dict]] = defaultdict(list)
    for obj in objects:
        if obj["scene_idx"] < args.start_idx:
            continue
        if args.end_idx >= 0 and obj["scene_idx"] >= args.end_idx:
            continue
        by_chunk[obj["chunk"]].append(obj)
    print(f"To process: {sum(len(v) for v in by_chunk.values())} entries across "
          f"{len(by_chunk)} chunks")

    # Camera rig (shared across all objects).
    viewmats_np, Ks_np = make_orbit_views(
        args.n_views, args.orbit_radius, args.orbit_fov_deg, args.resolution, up_axis="y"
    )
    viewmats = torch.from_numpy(viewmats_np).to(device)
    Ks = torch.from_numpy(Ks_np).to(device)

    score_viewmats_np, score_Ks_np = make_orbit_views(
        args.score_views, args.orbit_radius, args.orbit_fov_deg,
        args.score_resolution, up_axis="y"
    )
    score_viewmats = torch.from_numpy(score_viewmats_np).to(device)
    score_Ks = torch.from_numpy(score_Ks_np).to(device)

    c2w_all, fov_all = build_c2w(args.n_views, args.orbit_radius, args.orbit_fov_deg)

    # Load existing metadata so we accumulate (resume-friendly).
    metadata: dict[str, dict] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        print(f"Resumed metadata.json: {len(metadata)} entries already processed")

    skipped_low_op = 0
    skipped_existing = 0
    processed = 0
    failed = 0
    t_start = time.time()

    for chunk in sorted(by_chunk.keys()):
        chunk_objs = by_chunk[chunk]
        print(f"\n=== Chunk {chunk} ({len(chunk_objs)} objects) ===")
        zip_path = hf_hub_download(REPO_ID, f"{chunk}.zip", repo_type="dataset")
        print(f"  zip @ {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as zf:
            for obj in chunk_objs:
                uid = obj["uid"]
                scene_name = f"scene_{obj['scene_idx']:04d}"

                # Skip if already processed (H5 + at least one render).
                h5_out = h5_dir / f"{scene_name}.h5"
                first_view = renders_dir / f"{scene_name}_view_0.png"
                if h5_out.exists() and first_view.exists() and scene_name in metadata:
                    skipped_existing += 1
                    continue

                ply_member = f"{chunk}/{uid}/ckpts/point_cloud_15000.ply"
                try:
                    ply_bytes = zf.read(ply_member)
                except KeyError:
                    print(f"  [{scene_name}] MISSING: {ply_member}")
                    failed += 1
                    continue

                arr = load_normalized(ply_bytes)
                if arr is None:
                    print(f"  [{scene_name}] degenerate (zero AABB)")
                    failed += 1
                    continue

                max_op = float(arr["opacities"].max())
                if max_op < args.max_opacity_min:
                    skipped_low_op += 1
                    continue

                # Score + prune.
                scores = lightgaussian_scores(
                    arr, score_viewmats, score_Ks, args.score_resolution,
                    args.gamma, device,
                )
                rank = np.argsort(-scores)
                target = min(args.target_n, len(rank))
                keep_idx = np.sort(rank[:target])

                # GT renders: from the FULL Gaussians (model learns to fill detail).
                full_imgs = render_full(arr, viewmats, Ks, args.resolution, device)
                for v in range(args.n_views):
                    out_png = renders_dir / f"{scene_name}_view_{v}.png"
                    imageio.v3.imwrite(out_png, (full_imgs[v] * 255).astype(np.uint8))

                # H5: pruned subset + cameras.
                write_h5(arr, keep_idx, h5_out, c2w_all, fov_all)

                metadata[scene_name] = {
                    "uid": uid,
                    "chunk": chunk,
                    "psnr": obj["psnr"],
                    "lpips": obj["lpips"],
                    "caption": obj.get("caption", "")[:300],
                    "n_raw": int(arr["means"].shape[0]),
                    "n_kept": int(target),
                    "max_opacity": max_op,
                    "norm_scale": arr["norm_scale"],
                }
                processed += 1
                if processed % 25 == 0:
                    metadata_path.write_text(json.dumps(metadata, indent=2))
                    elapsed = time.time() - t_start
                    rate = processed / elapsed
                    eta_s = (len(objects) - args.start_idx - processed) / max(rate, 1e-6)
                    print(f"  [{scene_name}] processed (running {processed} | "
                          f"{rate:.2f} obj/s | ETA {eta_s/60:.1f} min)")

    metadata_path.write_text(json.dumps(metadata, indent=2))
    elapsed = time.time() - t_start
    print(f"\nDone. processed={processed}  skipped_low_op={skipped_low_op}  "
          f"skipped_existing={skipped_existing}  failed={failed}  "
          f"elapsed={elapsed/60:.1f} min")
    print(f"  H5 dir:     {h5_dir}/")
    print(f"  Renders:    {renders_dir}/")
    print(f"  Metadata:   {metadata_path}")


if __name__ == "__main__":
    main()
