"""Side-by-side render comparison across multiple checkpoints on val scenes.

Renders each (scene, view) with each model and composes a labeled strip
[GT | model_1 | model_2 | ...]. Also writes a grid summary (one row per scene,
first listed view, columns = GT + each model).

Usage:
  uv run --frozen python -m render_compare \\
    --models checkpoints_v10b/phase2_epoch_26.pt:V10b:rope \\
             checkpoints_v12/phase2_epoch_75.pt:V12:nerf \\
             checkpoints_v13/phase2_epoch_10.pt:V13:nerf \\
    --scenes 0,30,60,90,120,150,180 \\
    --views 0,7 \\
    --h5_dir data_v9_n20k/h5s_val \\
    --gt_dir data_v9/renders_val \\
    --out_dir compare_renders/v13_ep10_vs_baselines
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import gsplat
import h5py
import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from simple_ocio import ToneMapper

from data_external.orbit import make_orbit_views
from gaussianformer.models.config import GaussianFormerConfig
from gaussianformer.models.gaussianformer import GaussianFormer
from gaussianformer.pipelines.rendering_pipeline import GaussianFormerRenderingPipeline
from infer_gaussian import load_single_gaussian_h5_data

# Orbit camera params the val data was generated with (data_v9 process_objaverse
# defaults). The orbit is scene-independent, so viewmats/Ks are computed once.
ORBIT_N_VIEWS = 14
ORBIT_RADIUS = 1.7
ORBIT_FOV_DEG = 45.0
PRUNE_N = 20_000  # target_n the data_v9_n20k H5s were pruned to


def scene_n_gaussians(h5_path) -> int:
    with h5py.File(h5_path, "r") as f:
        return f["means"].shape[0]


def render_pruned_gt(h5_path, view_idx: int, viewmats: torch.Tensor, Ks: torch.Tensor,
                     resolution: int, device: torch.device) -> np.ndarray:
    """Rasterize the H5's (pruned) Gaussians at one orbit view, matching the exact
    gsplat call render_full used to make the full-GT PNGs (process_objaverse.py)."""
    with h5py.File(h5_path, "r") as f:
        means = torch.from_numpy(np.array(f["means"], dtype=np.float32)).to(device)
        scales = torch.from_numpy(np.array(f["scales"], dtype=np.float32)).to(device)
        quats = torch.from_numpy(np.array(f["rotations"], dtype=np.float32)).to(device)
        colors = torch.from_numpy(np.array(f["colors"], dtype=np.float32)).to(device)
        opacities = torch.from_numpy(np.array(f["opacities"], dtype=np.float32)).to(device)
    if opacities.ndim == 2:
        opacities = opacities.squeeze(-1)
    with torch.no_grad():
        img, _, _ = gsplat.rasterization(
            means=means, quats=quats, scales=scales,
            opacities=opacities, colors=colors,
            viewmats=viewmats[view_idx:view_idx + 1], Ks=Ks[view_idx:view_idx + 1],
            width=resolution, height=resolution,
            sh_degree=None, eps2d=0.3, render_mode="RGB",
            near_plane=0.01, packed=True,
        )
    return img[0].clamp(0, 1).cpu().numpy()


@dataclass
class ModelSpec:
    ckpt: Path
    label: str
    pe_type: str


def parse_model(s: str) -> ModelSpec:
    ckpt, label, pe = s.split(":")
    return ModelSpec(ckpt=Path(ckpt), label=label, pe_type=pe)


def load_model(spec: ModelSpec, device: torch.device) -> GaussianFormerRenderingPipeline:
    config = GaussianFormerConfig(pe_type=spec.pe_type)
    ckpt = torch.load(spec.ckpt, map_location="cpu", weights_only=True)
    if "lora" in ckpt:
        # Adapter-only checkpoint: base (with its recorded config overrides) from the recorded
        # seed, re-wrap, load A/B, fold in.
        from gaussianformer.layers.lora import load_lora
        config = config.with_overrides(ckpt["lora"].get("model_cfg") or None)
        model = GaussianFormer(config)
        base = torch.load(ckpt["lora"]["base_ckpt"], map_location="cpu", weights_only=True)
        own = model.state_dict()
        base_sd = {k: v for k, v in base["model_state_dict"].items()
                   if not (k.endswith(".freqs") and (k not in own or own[k].shape != v.shape))}
        missing, unexpected = model.load_state_dict(base_sd, strict=False)
        assert not unexpected and all(k.endswith(".freqs") for k in missing), (missing, unexpected)
        load_lora(model, ckpt, merge=True)
    else:
        model = GaussianFormer(config)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    pipe = GaussianFormerRenderingPipeline(model)
    pipe.to(device)
    return pipe


def render_view(pipe: GaussianFormerRenderingPipeline, data, view_idx: int,
                resolution: int, tone_mapper) -> np.ndarray:
    c2w = data["c2w"][view_idx:view_idx + 1].unsqueeze(0)
    fov = data["fov"][view_idx:view_idx + 1].unsqueeze(0)
    with torch.no_grad():
        out = pipe(
            gaussians=data["gaussians"].unsqueeze(0),
            mask=data["mask"].unsqueeze(0),
            c2w=c2w, fov=fov,
            resolution=resolution, torch_dtype=torch.bfloat16,
        )
    hdr = out[0, 0].cpu().float().numpy()
    ldr = tone_mapper.hdr_to_ldr(hdr) if tone_mapper is not None else np.clip(hdr, 0, 1)
    return np.clip(ldr, 0, 1).astype(np.float32)


def load_gt(gt_path: Path, resolution: int) -> np.ndarray:
    gt = iio.imread(gt_path).astype(np.float32) / 255.0
    if gt.shape[0] != resolution:
        gt = np.asarray(Image.fromarray((gt * 255).astype(np.uint8)).resize(
            (resolution, resolution), Image.LANCZOS)).astype(np.float32) / 255.0
    if gt.shape[-1] == 4:
        gt = gt[..., :3]
    return gt


def label_panel(img: np.ndarray, text: str, font: ImageFont.ImageFont) -> np.ndarray:
    pil = Image.fromarray((img * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    # bottom-left text on a translucent dark band
    band_h = 28
    overlay = Image.new("RGBA", (pil.width, band_h), (0, 0, 0, 160))
    pil.paste(overlay, (0, 0), overlay)
    draw.text((6, 4), text, fill=(255, 255, 255), font=font)
    return np.asarray(pil).astype(np.float32) / 255.0


def compose_row(panels: list[np.ndarray], gap: int = 6) -> np.ndarray:
    h = panels[0].shape[0]
    sep = np.ones((h, gap, 3), dtype=np.float32)
    out = panels[0]
    for p in panels[1:]:
        out = np.concatenate([out, sep, p], axis=1)
    return out


def compose_grid(rows: list[np.ndarray], gap: int = 8) -> np.ndarray:
    w = rows[0].shape[1]
    sep = np.ones((gap, w, 3), dtype=np.float32)
    out = rows[0]
    for r in rows[1:]:
        out = np.concatenate([out, sep, r], axis=0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="Each item: path:label:pe_type (e.g. ckpt.pt:V13:nerf)")
    ap.add_argument("--scenes", type=str, default="0,30,60,90,120,150,180",
                    help="Comma-separated scene indices")
    ap.add_argument("--views", type=str, default="0,7",
                    help="Comma-separated view indices to render per scene")
    ap.add_argument("--h5_dir", type=Path, default=Path("data_v9_n20k/h5s_val"))
    ap.add_argument("--gt_dir", type=Path, default=Path("data_v9/renders_val"))
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--tone_mapper", type=str, default="none",
                    help="MUST match the GT pipeline. data_v9 GT is written with NO "
                    "tone map, so 'none' (clip) is correct; AGX desaturates.")
    ap.add_argument("--pruned_gt", action="store_true",
                    help="Insert a 'pruned-GT' column: gsplat rasterization of the H5's "
                    "(pruned) Gaussians, via the same gsplat call that made the full-GT "
                    "PNGs. Auto-dropped per-scene when the H5 has < --prune_n Gaussians "
                    "(i.e. the scene wasn't pruned, so pruned-GT == GT).")
    ap.add_argument("--prune_n", type=int, default=PRUNE_N,
                    help="Gaussian count the H5s were pruned to; a scene counts as pruned "
                    "iff it has >= this many Gaussians.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scenes = [int(s) for s in args.scenes.split(",")]
    views = [int(v) for v in args.views.split(",")]

    tm_name = "Khronos PBR Neutral" if args.tone_mapper == "pbr_neutral" else args.tone_mapper
    tone_mapper = ToneMapper(tm_name) if args.tone_mapper != "none" else None

    specs = [parse_model(m) for m in args.models]
    print(f"Loading {len(specs)} models...")
    pipes: list[tuple[ModelSpec, GaussianFormerRenderingPipeline]] = []
    for s in specs:
        print(f"  {s.label} ({s.pe_type}): {s.ckpt}")
        pipes.append((s, load_model(s, device)))

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    pruned_viewmats = pruned_Ks = None
    if args.pruned_gt:
        vm, ks = make_orbit_views(ORBIT_N_VIEWS, ORBIT_RADIUS, ORBIT_FOV_DEG,
                                  args.resolution, up_axis="y")
        pruned_viewmats = torch.from_numpy(vm).to(device)
        pruned_Ks = torch.from_numpy(ks).to(device)

    grid_rows = []
    for si in scenes:
        h5_path = args.h5_dir / f"scene_{si:04d}.h5"
        if not h5_path.exists():
            print(f"  skip scene {si:04d}: no h5")
            continue
        data = load_single_gaussian_h5_data(h5_path)
        for k in ("gaussians", "mask", "c2w", "fov"):
            data[k] = data[k].to(device)
        with h5py.File(h5_path, "r") as f:
            n_views = f["c2w"].shape[0]

        for v_idx in views:
            if v_idx >= n_views:
                continue
            # GT
            gt_path = args.gt_dir / f"scene_{si:04d}_view_{v_idx}.png"
            if not gt_path.exists():
                print(f"  skip scene {si:04d} view {v_idx}: no gt")
                continue
            gt = load_gt(gt_path, args.resolution)
            panels = [label_panel(gt, f"GT  scene_{si:04d} v{v_idx}", font)]

            if args.pruned_gt:
                if scene_n_gaussians(h5_path) >= args.prune_n:
                    pgt = render_pruned_gt(h5_path, v_idx, pruned_viewmats, pruned_Ks,
                                           args.resolution, device)
                    ppsnr = 10.0 * np.log10(1.0 / (float(((pgt - gt) ** 2).mean()) + 1e-12))
                    panels.append(label_panel(pgt, f"pruned-GT N={args.prune_n}  PSNR {ppsnr:.2f}dB", font))
                else:
                    # Scene wasn't pruned -> pruned-GT == GT; show a placeholder to keep
                    # grid columns aligned (no need to render, per spec).
                    panels.append(label_panel(np.zeros_like(gt), "pruned-GT n/a (not pruned)", font))

            for spec, pipe in pipes:
                rendered = render_view(pipe, data, v_idx, args.resolution, tone_mapper)
                mse = float(((rendered - gt) ** 2).mean())
                psnr = 10.0 * np.log10(1.0 / (mse + 1e-12))
                panels.append(label_panel(rendered, f"{spec.label}  PSNR {psnr:.2f}dB", font))

            strip = compose_row(panels)
            strip_path = args.out_dir / f"scene_{si:04d}_view{v_idx}_compare.png"
            iio.imwrite(strip_path, (np.clip(strip, 0, 1) * 255).astype(np.uint8))
            print(f"  wrote {strip_path}")

            if v_idx == views[0]:
                grid_rows.append(strip)

    if grid_rows:
        grid = compose_grid(grid_rows)
        grid_path = args.out_dir / "grid_overview.png"
        iio.imwrite(grid_path, (np.clip(grid, 0, 1) * 255).astype(np.uint8))
        print(f"  wrote {grid_path}")


if __name__ == "__main__":
    main()
