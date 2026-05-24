"""Prep one or more Objaverse_Splats scenes for the run_scene.py eval pipeline.

Steps per scene:
  1. Download the chunk zip from `ShapeSplats/Objaverse_Splats` (HF) if not cached.
  2. Extract `<chunk>/<uid>/ckpts/point_cloud_15000.ply` and save as
     `data_external/<scene>/raw.ply`.
  3. Normalize per the scene's SceneConfig (median-center + auto AABB scaling).
  4. Render cfg.n_views full-scene gsplat reference views into
     `data_external/<scene>/renders/gsplat_full/view_{i:02d}.png`.

After this completes, the scene is ready for
`python -m data_external.run_scene --scene <name> --checkpoint <...>`.

Usage:
  python -m data_external.prep_objaverse_scene --scene house
  python -m data_external.prep_objaverse_scene --scene house --scene dragon --scene cartoon
"""
from __future__ import annotations

import argparse
import zipfile

import imageio
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from data_external.orbit import make_orbit_views
from data_external.run_scene import normalize_raw, render_gsplat_subset
from data_external.scene_configs import SCENES, SceneConfig


REPO_ID = "ShapeSplats/Objaverse_Splats"


def download_raw_ply(cfg: SceneConfig) -> None:
    """Download the chunk zip and extract this scene's PLY to cfg.raw_ply."""
    if cfg.objaverse_uid is None or cfg.objaverse_chunk is None:
        raise ValueError(
            f"Scene {cfg.name!r} is not an Objaverse scene "
            "(needs objaverse_uid + objaverse_chunk in SCENES)."
        )
    cfg.scene_dir.mkdir(parents=True, exist_ok=True)
    if cfg.raw_ply.exists():
        print(f"[{cfg.name}] raw.ply already exists "
              f"({cfg.raw_ply.stat().st_size/1e6:.1f} MB); skipping download")
        return

    print(f"[{cfg.name}] downloading chunk {cfg.objaverse_chunk}.zip from {REPO_ID}...")
    zip_path = hf_hub_download(REPO_ID, f"{cfg.objaverse_chunk}.zip", repo_type="dataset")
    print(f"[{cfg.name}] zip @ {zip_path}")

    ply_member = f"{cfg.objaverse_chunk}/{cfg.objaverse_uid}/ckpts/point_cloud_15000.ply"
    with zipfile.ZipFile(zip_path, "r") as zf:
        ply_bytes = zf.read(ply_member)

    cfg.raw_ply.write_bytes(ply_bytes)
    print(f"[{cfg.name}] wrote {cfg.raw_ply} ({len(ply_bytes)/1e6:.2f} MB)")


def render_full_reference(cfg: SceneConfig, device: torch.device) -> None:
    """Render the (normalized, unpruned) Gaussians via gsplat across cfg.n_views cameras."""
    if all((cfg.full_gsplat_dir / f"view_{i:02d}.png").exists() for i in range(cfg.n_views)):
        print(f"[{cfg.name}] full-gsplat references already at {cfg.full_gsplat_dir}/; skipping")
        return

    arrays = normalize_raw(cfg)
    n_total = arrays["means"].shape[0]
    keep_idx = np.arange(n_total)

    viewmats_np, Ks_np = make_orbit_views(
        cfg.n_views, cfg.orbit_radius, cfg.orbit_fov_deg, cfg.resolution, up_axis="y"
    )
    viewmats = torch.from_numpy(viewmats_np).to(device)
    Ks = torch.from_numpy(Ks_np).to(device)

    print(f"[{cfg.name}] rendering {cfg.n_views} full-scene gsplat references "
          f"@ {cfg.resolution}px (radius {cfg.orbit_radius}, fov {cfg.orbit_fov_deg})...")
    imgs = render_gsplat_subset(arrays, keep_idx, viewmats, Ks, cfg.resolution, device)

    cfg.full_gsplat_dir.mkdir(parents=True, exist_ok=True)
    for i in range(cfg.n_views):
        out = cfg.full_gsplat_dir / f"view_{i:02d}.png"
        imageio.v3.imwrite(out, (imgs[i] * 255).astype(np.uint8))
    print(f"[{cfg.name}] wrote {cfg.n_views} reference views -> {cfg.full_gsplat_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene", action="append", required=True,
        choices=sorted(s for s, c in SCENES.items() if c.objaverse_uid is not None),
        help="Scene name(s) to prep. Repeat the flag for multiple scenes.",
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="Only download + extract PLYs (no rendering). Useful for running CPU-side "
             "while a GPU job holds the GRES cap.",
    )
    args = parser.parse_args()

    for scene_name in args.scene:
        cfg = SCENES[scene_name]
        print(f"\n========== prep {scene_name} ==========")
        download_raw_ply(cfg)

    if args.download_only:
        print("\n--download-only: skipping reference render step.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\ndevice: {device}")
    for scene_name in args.scene:
        cfg = SCENES[scene_name]
        render_full_reference(cfg, device)

    print("\nAll requested scenes prepared.")


if __name__ == "__main__":
    main()
