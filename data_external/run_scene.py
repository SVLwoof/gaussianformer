"""Scene-parameterized eval pipeline. Generalizes the original run_tomatoes.py
to arbitrary scenes defined in data_external/scene_configs.py.

For each N in cfg.n_variants:
  1. Prune the (already-scored) normalized Gaussians to top-N by LightGaussian score.
  2. Write pruned PLY and H5 (14-dim tokens + n_views orbit cameras).
  3. Render the pruned scene via gsplat (column 2: pruned-N reference).
  4. Render the same H5 through GaussianFormer (column 3: model output).
  5. Build a per-N 3-way overview: full-scene gsplat | pruned-N gsplat | GF.

The full-scene gsplat reference (column 1) must exist under cfg.full_gsplat_dir
before this script runs. For tomatoes it was pre-rendered by
render_tomatoes_raw_orbit.py; for new scenes use prep_objaverse_scene.py.

Outputs land under cfg.scene_dir / {pruned PLY, h5/, renders/}.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gsplat
import h5py
import imageio
import numpy as np
import torch
from plyfile import PlyData, PlyElement

from data_external.orbit import look_at_blender, make_orbit_views
from data_external.scene_configs import SCENES, SceneConfig
from gaussianformer.models.config import GaussianFormerConfig
from gaussianformer.models.gaussianformer import GaussianFormer
from gaussianformer.pipelines.rendering_pipeline import GaussianFormerRenderingPipeline


def normalize_raw(cfg: SceneConfig) -> dict[str, np.ndarray]:
    """Load raw.ply and apply scene-configured normalization."""
    print(f"Loading {cfg.raw_ply}...")
    plydata = PlyData.read(str(cfg.raw_ply))
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

    if cfg.median_center:
        median = np.median(means, axis=0)
        means = means - median
        print(f"  median {median.tolist()} -> centered")

    if cfg.flip_x_rotation:
        # 180-deg about X: (x,y,z) -> (x,-y,-z); quaternion composes via the same axis.
        means[:, 1] *= -1.0
        means[:, 2] *= -1.0
        qw, qx, qy, qz = quats[:, 0].copy(), quats[:, 1].copy(), quats[:, 2].copy(), quats[:, 3].copy()
        quats = np.stack([-qx, qw, -qz, qy], axis=-1).astype(np.float32)
        print("  applied 180-deg X rotation")

    if cfg.norm_scale is not None:
        scale = cfg.norm_scale
    else:
        aabb_max = float(np.abs(means).max())
        scale = cfg.norm_target_aabb / max(aabb_max, 1e-9)
    means *= scale
    scales = np.exp(scales_log) * scale
    print(f"  scaled by {scale:.4f}")

    print(f"  positions: [{means.min():+.3f}, {means.max():+.3f}]   "
          f"scales (linear): [{scales.min():.5f}, {scales.max():.5f}]")
    print(f"  opacities: [{opacities.min():.3f}, {opacities.max():.3f}]   "
          f"colors: [{colors.min():.3f}, {colors.max():.3f}]")

    return {
        "means": means, "scales": scales, "quats": quats,
        "colors": colors, "opacities": opacities,
        "raw_vertex_data": v.data,
    }


def lightgaussian_scores(arrays: dict, cfg: SceneConfig, device: torch.device) -> np.ndarray:
    """Return per-Gaussian LightGaussian scores: sum_v(opacity*radii_x*radii_y)*max_scale**gamma."""
    means = torch.from_numpy(arrays["means"]).to(device)
    scales = torch.from_numpy(arrays["scales"]).to(device)
    quats = torch.from_numpy(arrays["quats"]).to(device)
    opacities = torch.from_numpy(arrays["opacities"]).to(device)
    colors = torch.from_numpy(arrays["colors"]).to(device)
    n = means.shape[0]

    viewmats_np, Ks_np = make_orbit_views(
        cfg.score_views, cfg.orbit_radius, cfg.orbit_fov_deg, cfg.score_resolution, up_axis="y"
    )
    viewmats = torch.from_numpy(viewmats_np).to(device)
    Ks = torch.from_numpy(Ks_np).to(device)

    score = torch.zeros(n, device=device, dtype=torch.float64)
    print(f"\nScoring {cfg.score_views} orbit views @ {cfg.score_resolution}px (LightGaussian)...")
    for v in range(cfg.score_views):
        with torch.no_grad():
            _, _, info = gsplat.rasterization(
                means=means, quats=quats, scales=scales,
                opacities=opacities, colors=colors,
                viewmats=viewmats[v:v+1], Ks=Ks[v:v+1],
                width=cfg.score_resolution, height=cfg.score_resolution,
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
        if (v + 1) % 16 == 0 or v == cfg.score_views - 1:
            n_seen = int((score > 0).sum().item())
            print(f"  view {v + 1:3d}/{cfg.score_views}  cumulative seen: {n_seen:,} Gaussians")

    max_scale = scales.max(dim=1).values.double()
    return (score * (max_scale ** cfg.gamma)).cpu().numpy()


def write_pruned_ply(arrays: dict, keep_idx: np.ndarray, out_ply: Path) -> None:
    """Write a Kerbl-format PLY for the pruned subset (preserves raw f_rest_* SH)."""
    new_data = arrays["raw_vertex_data"][keep_idx].copy()
    new_data["x"] = arrays["means"][keep_idx, 0]
    new_data["y"] = arrays["means"][keep_idx, 1]
    new_data["z"] = arrays["means"][keep_idx, 2]
    for i in range(3):
        new_data[f"scale_{i}"] = np.log(arrays["scales"][keep_idx, i] + 1e-30)
    for i in range(4):
        new_data[f"rot_{i}"] = arrays["quats"][keep_idx, i]
    new_element = PlyElement.describe(new_data, "vertex")
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([new_element]).write(str(out_ply))


def write_h5(arrays: dict, keep_idx: np.ndarray, out_h5: Path,
             c2w_all: np.ndarray, fov_all: np.ndarray) -> None:
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("means", data=arrays["means"][keep_idx])
        f.create_dataset("scales", data=arrays["scales"][keep_idx])
        f.create_dataset("rotations", data=arrays["quats"][keep_idx])
        f.create_dataset("colors", data=arrays["colors"][keep_idx])
        f.create_dataset("opacities", data=arrays["opacities"][keep_idx][:, None])
        f.create_dataset("c2w", data=c2w_all)
        f.create_dataset("fov", data=fov_all)


def render_gsplat_subset(arrays: dict, keep_idx: np.ndarray,
                          viewmats: torch.Tensor, Ks: torch.Tensor,
                          resolution: int, device: torch.device) -> np.ndarray:
    """Render the pruned subset on the orbit. Returns [V,H,W,3] in [0,1]."""
    means = torch.from_numpy(arrays["means"][keep_idx]).to(device)
    scales = torch.from_numpy(arrays["scales"][keep_idx]).to(device)
    quats = torch.from_numpy(arrays["quats"][keep_idx]).to(device)
    opacities = torch.from_numpy(arrays["opacities"][keep_idx]).to(device)
    colors = torch.from_numpy(arrays["colors"][keep_idx]).to(device)
    out = []
    for v in range(viewmats.shape[0]):
        with torch.no_grad():
            img, _, _ = gsplat.rasterization(
                means=means, quats=quats, scales=scales,
                opacities=opacities, colors=colors,
                viewmats=viewmats[v:v+1], Ks=Ks[v:v+1],
                width=resolution, height=resolution,
                sh_degree=None, eps2d=0.3, render_mode="RGB",
                near_plane=0.01, packed=True,
            )
        out.append(img[0].clamp(0, 1).cpu().numpy())
    return np.stack(out)


def render_gaussianformer(h5_path: Path, pipeline, resolution: int,
                           device: torch.device) -> np.ndarray:
    """Render via GaussianFormer. Returns [V,H,W,3] in [0,1] (LDR clipped)."""
    from infer_gaussian import load_single_gaussian_h5_data
    data = load_single_gaussian_h5_data(h5_path)
    gaussians = data["gaussians"].unsqueeze(0).to(device)
    mask = data["mask"].unsqueeze(0).to(device)
    c2w_all = data["c2w"].to(device)
    fov_all = data["fov"].to(device)
    n_views = c2w_all.shape[0]
    out = []
    for v in range(n_views):
        c2w = c2w_all[v:v+1].unsqueeze(0)
        fov = fov_all[v:v+1].unsqueeze(0)
        with torch.no_grad():
            res = pipeline(
                gaussians=gaussians, mask=mask, c2w=c2w, fov=fov,
                resolution=resolution, torch_dtype=torch.bfloat16,
            )
        hdr = res[0, 0].cpu().float().numpy()
        out.append(np.clip(hdr, 0, 1).astype(np.float32))
    return np.stack(out)


def make_3way_overview(full_imgs: np.ndarray, pruned_imgs: np.ndarray,
                        gf_imgs: np.ndarray,
                        labels: tuple[str, str, str] = (
                            "gsplat-full", "gsplat-pruned", "GaussianFormer"),
                        title: str | None = None) -> np.ndarray:
    """Per-view triple stacked into 7 rows x 2 columns, with column labels on top."""
    from PIL import Image, ImageDraw, ImageFont
    n = full_imgs.shape[0]
    H, W = pruned_imgs.shape[1], pruned_imgs.shape[2]
    gutter = np.ones((H, 4, 3), dtype=np.float32)
    triples = [np.concatenate([full_imgs[i], gutter, pruned_imgs[i], gutter, gf_imgs[i]], axis=1)
               for i in range(n)]
    cols = 2
    rows_n = (n + cols - 1) // cols
    Hh, Ww, _ = triples[0].shape
    grid = np.ones(
        (rows_n * Hh + (rows_n - 1) * 8, cols * Ww + (cols - 1) * 8, 3),
        dtype=np.float32,
    )
    for i, img in enumerate(triples):
        r, c = divmod(i, cols)
        y, x = r * (Hh + 8), c * (Ww + 8)
        grid[y:y+Hh, x:x+Ww] = img

    title_h = 38 if title else 0
    col_h = 32
    header_h = title_h + col_h
    header = np.ones((header_h, grid.shape[1], 3), dtype=np.float32)
    pil = Image.fromarray((header * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        col_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except OSError:
        title_font = col_font = ImageFont.load_default()

    def _draw_centered(text: str, cx: int, cy: int, font) -> None:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((cx - tw // 2, cy - th // 2), text, fill=(0, 0, 0), font=font)

    if title:
        _draw_centered(title, grid.shape[1] // 2, title_h // 2, title_font)

    sub_centers = [W // 2, W + 4 + W // 2, 2 * W + 8 + W // 2]
    for triple_idx in range(cols):
        offset = triple_idx * (Ww + 8)
        for label, cx in zip(labels, sub_centers):
            _draw_centered(label, offset + cx, title_h + col_h // 2, col_font)

    header = np.asarray(pil).astype(np.float32) / 255.0
    return np.clip(np.concatenate([header, grid], axis=0), 0, 1)


def make_master_overview(per_n_imgs: dict[int, np.ndarray],
                          full_imgs: np.ndarray, view_idx: int) -> np.ndarray:
    """Master comparison for a single view: full | pruned-N1 | gf-N1 | pruned-N2 | gf-N2 | ..."""
    H = full_imgs.shape[1]
    gutter = np.ones((H, 4, 3), dtype=np.float32)
    parts = [full_imgs[view_idx]]
    for n in sorted(per_n_imgs.keys()):
        pruned, gf = per_n_imgs[n]
        parts.extend([gutter, pruned[view_idx], gutter, gf[view_idx]])
    return np.clip(np.concatenate(parts, axis=1), 0, 1)


def build_orbit_c2w(n_views: int, radius: float, fov_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """14-view orbit with alternating high/low elevation (matches the training-data rig)."""
    c2w_list = []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        elev = 0.4 * radius if i % 2 == 0 else -0.1 * radius
        eye = np.array(
            [radius * np.cos(theta), elev, radius * np.sin(theta)],
            dtype=np.float32,
        )
        c2w_list.append(look_at_blender(eye, np.zeros(3, dtype=np.float32),
                                          up=np.array([0, 1, 0], dtype=np.float32)))
    c2w_all = np.stack(c2w_list).astype(np.float32)
    fov_all = np.full(n_views, fov_deg, dtype=np.float32)
    return c2w_all, fov_all


def main(scene: str | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default=scene, required=(scene is None),
                        choices=sorted(SCENES.keys()),
                        help="Scene name from data_external/scene_configs.py SCENES registry.")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to GaussianFormer checkpoint .pt to evaluate.")
    parser.add_argument("--tag", type=str, default=None,
                        help="Suffix appended to render output dirs (e.g. 'v11_ep10'). "
                        "Render outputs land at renders/gaussianformer_n{N}_{tag}/ "
                        "and overviews at overview_3way_n{N}_{tag}.png.")
    parser.add_argument("--pe_type", type=str, default="rope",
                        choices=["rope", "nerf", "nerf_perfield"],
                        help="Must match the architecture of the checkpoint.")
    parser.add_argument("--scale_pe_num_freqs", type=int, default=6,
                        help="Only used when pe_type='nerf_perfield'.")
    parser.add_argument("--label", type=str, default=None,
                        help="Display label for the GaussianFormer column in overview captions "
                        "(e.g. 'V12 ep75'). Defaults to --tag.")
    args = parser.parse_args()
    cfg = SCENES[args.scene]
    checkpoint = args.checkpoint
    tag = f"_{args.tag}" if args.tag else ""
    gf_label = args.label or args.tag or "GaussianFormer"

    device = torch.device("cuda")

    arrays = normalize_raw(cfg)
    n_total = len(arrays["means"])

    normalized_ply = cfg.scene_dir / "normalized.ply"
    write_pruned_ply(arrays, np.arange(n_total), normalized_ply)
    print(f"Wrote {normalized_ply} (full normalized PLY, {normalized_ply.stat().st_size/1e6:.1f} MB)")

    scores = lightgaussian_scores(arrays, cfg, device)
    rank = np.argsort(-scores)

    viewmats_np, Ks_np = make_orbit_views(cfg.n_views, cfg.orbit_radius, cfg.orbit_fov_deg,
                                            cfg.resolution, up_axis="y")
    viewmats = torch.from_numpy(viewmats_np).to(device)
    Ks = torch.from_numpy(Ks_np).to(device)
    c2w_all, fov_all = build_orbit_c2w(cfg.n_views, cfg.orbit_radius, cfg.orbit_fov_deg)

    full_imgs = []
    for i in range(cfg.n_views):
        p = cfg.full_gsplat_dir / f"view_{i:02d}.png"
        full_imgs.append(imageio.v3.imread(p).astype(np.float32) / 255.0)
    full_imgs = np.stack(full_imgs)
    print(f"\nLoaded {cfg.n_views} full-scene gsplat references from {cfg.full_gsplat_dir}/")

    print(f"\nLoading checkpoint {checkpoint}...")
    config = GaussianFormerConfig(pe_type=args.pe_type, scale_pe_num_freqs=args.scale_pe_num_freqs)
    model = GaussianFormer(config)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "?")
    loss = ckpt.get("loss", 0)
    print(f"  epoch {epoch}, loss {loss:.6f}")
    pipeline = GaussianFormerRenderingPipeline(model)
    pipeline.to(device)

    per_n: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    psnr_table: dict[int, tuple[float, ...]] = {}

    for n_target in cfg.n_variants:
        print(f"\n========== {cfg.name} N = {n_target} ==========")
        keep_idx = np.sort(rank[:n_target])

        pruned_ply = cfg.scene_dir / f"pruned_n{n_target}.ply"
        h5_path = cfg.h5_dir / f"{cfg.name}_n{n_target}.h5"
        gsplat_dir = cfg.renders_dir / f"gsplat_n{n_target}"
        gf_dir = cfg.renders_dir / f"gaussianformer_n{n_target}{tag}"
        gsplat_dir.mkdir(parents=True, exist_ok=True)
        gf_dir.mkdir(parents=True, exist_ok=True)

        write_pruned_ply(arrays, keep_idx, pruned_ply)
        print(f"  wrote {pruned_ply.name} ({pruned_ply.stat().st_size/1e6:.2f} MB)")
        write_h5(arrays, keep_idx, h5_path, c2w_all, fov_all)
        print(f"  wrote {h5_path}")

        pruned_imgs = render_gsplat_subset(arrays, keep_idx, viewmats, Ks, cfg.resolution, device)
        for i in range(cfg.n_views):
            imageio.v3.imwrite(gsplat_dir / f"view_{i:02d}.png",
                               (pruned_imgs[i] * 255).astype(np.uint8))
        print(f"  rendered {cfg.n_views} pruned-gsplat views -> {gsplat_dir}/")

        gf_imgs = render_gaussianformer(h5_path, pipeline, cfg.resolution, device)
        for i in range(cfg.n_views):
            imageio.v3.imwrite(gf_dir / f"view_{i:02d}.png",
                               (gf_imgs[i] * 255).astype(np.uint8))
        print(f"  rendered {cfg.n_views} GaussianFormer views -> {gf_dir}/")

        psnr_vs_pruned = []
        psnr_vs_full = []
        for i in range(cfg.n_views):
            mse_p = ((gf_imgs[i] - pruned_imgs[i]) ** 2).mean()
            mse_f = ((gf_imgs[i] - full_imgs[i]) ** 2).mean()
            psnr_vs_pruned.append(10 * np.log10(1.0 / (mse_p + 1e-12)))
            psnr_vs_full.append(10 * np.log10(1.0 / (mse_f + 1e-12)))
        pp = np.array(psnr_vs_pruned)
        pf = np.array(psnr_vs_full)
        psnr_table[n_target] = (pp.mean(), pp.min(), pp.max(),
                                 pf.mean(), pf.min(), pf.max())
        print(f"  PSNR vs pruned-gsplat: mean {pp.mean():.2f} dB ({pp.min():.2f} – {pp.max():.2f})")
        print(f"  PSNR vs full-gsplat:   mean {pf.mean():.2f} dB ({pf.min():.2f} – {pf.max():.2f})")

        overview = make_3way_overview(
            full_imgs, pruned_imgs, gf_imgs,
            labels=("gsplat-full", f"gsplat-pruned (N={n_target:,})", gf_label),
            title=f"{cfg.name}  -  N={n_target:,}  -  {gf_label} vs gsplat",
        )
        overview_path = cfg.renders_dir / f"overview_3way_n{n_target}{tag}.png"
        imageio.v3.imwrite(overview_path, (overview * 255).astype(np.uint8))
        print(f"  3-way overview -> {overview_path}")

        per_n[n_target] = (pruned_imgs, gf_imgs)

    print("\n========== MASTER OVERVIEW ==========")
    chosen_views = [0, 1]
    rows = [make_master_overview(per_n, full_imgs, v) for v in chosen_views]
    W = rows[0].shape[1]
    gutter = np.ones((8, W, 3), dtype=np.float32)
    master = np.concatenate([rows[0], gutter, rows[1]], axis=0)
    master_path = cfg.renders_dir / f"overview_all_N{tag}.png"
    imageio.v3.imwrite(master_path, (np.clip(master, 0, 1) * 255).astype(np.uint8))
    print(f"Master overview (2 views x [full | (pruned|gf) per N]) -> {master_path}")

    print("\n========== PSNR SUMMARY ==========")
    print(f"  {'N':>6}  {'vs pruned (mean / min / max)':>35}  {'vs full (mean / min / max)':>35}")
    for n in sorted(psnr_table.keys()):
        pm, pmn, pmx, fm, fmn, fmx = psnr_table[n]
        print(f"  {n:>6}  "
              f"{pm:>10.2f} / {pmn:>5.2f} / {pmx:>5.2f}    "
              f"{fm:>10.2f} / {fmn:>5.2f} / {fmx:>5.2f}")


if __name__ == "__main__":
    main()
