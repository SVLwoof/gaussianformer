"""Held-out evaluation: PSNR and LPIPS of the model and of rasterizing its 20k-Gaussian input, both against
the full splat's ground truth, on each object's bounding box (square crop, 12 px margin).

  python evaluate.py --data data/val --objects data/splits/heldout300.json
"""
import argparse
import json
from pathlib import Path

import gsplat
import imageio.v3 as iio
import lpips
import numpy as np
import torch
from PIL import Image

from data.cameras import DISTANCES, to_gsplat
from gaussianformer.pipelines.rendering_pipeline import GaussianFormerRenderingPipeline
from gaussianformer.utils.checkpoint import load_checkpoint
from training.dataset import load_gaussians


def crop(images: list[np.ndarray], ref: np.ndarray, pad: int = 12) -> list[np.ndarray]:
    """Square crop around the object in `ref` (pixels brighter than 0.02), resized to the image size."""
    ys, xs = np.where(ref.mean(-1) > 0.02)
    if len(ys) < 10:
        return images
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) // 2 + pad
    size = ref.shape[0]
    y0, y1, x0, x1 = max(0, cy - half), min(size, cy + half), max(0, cx - half), min(size, cx + half)
    out = []
    for im in images:
        c = Image.fromarray((np.clip(im[y0:y1, x0:x1], 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
        out.append(np.asarray(c, dtype=np.float32) / 255.0)
    return out


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return 10.0 * np.log10(1.0 / (float(((a - b) ** 2).mean()) + 1e-12))


def load_model(model: str, device) -> GaussianFormerRenderingPipeline:
    if Path(model).suffix == ".pt":
        return GaussianFormerRenderingPipeline(load_checkpoint(Path(model))[0].eval()).to(device)
    return GaussianFormerRenderingPipeline.from_pretrained(model).to(device)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="shahafvl/gaussianformer", help="Hugging Face id, directory or .pt checkpoint")
    ap.add_argument("--data", type=Path, required=True, help="directory with h5s/ and renders/")
    ap.add_argument("--objects", type=Path, required=True, help="JSON list of object indices")
    ap.add_argument("--out", type=Path, default=None, help="optional JSONL of per-view results")
    ap.add_argument("--views", default="0,4,7,11", help="comma-separated view indices, or 'all'")
    a = ap.parse_args()
    device = torch.device("cuda")
    pipe = load_model(a.model, device)
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    group_of = [i for i, (_, n) in enumerate(DISTANCES) for _ in range(n)]
    rows = []
    for scene in json.loads(a.objects.read_text()):
        name = f"scene_{scene:04d}"
        gaussians, c2w, fov = load_gaussians(a.data / "h5s" / f"{name}.h5")
        wanted = range(len(c2w)) if a.views == "all" else [int(v) for v in a.views.split(",")]
        views = [v for v in wanted if (a.data / "renders" / f"{name}_view_{v}.png").exists()]
        g = {k: torch.from_numpy(x).to(device) for k, x in
             zip(("means", "scales", "rotations", "colors", "opacities"), np.split(gaussians, [3, 6, 10, 13], -1))}
        viewmats, Ks = (torch.from_numpy(x).to(device) for x in to_gsplat(c2w[views]))
        with torch.no_grad():
            raster, _, _ = gsplat.rasterization(
                means=g["means"], quats=g["rotations"], scales=g["scales"], opacities=g["opacities"][:, 0],
                colors=g["colors"], viewmats=viewmats, Ks=Ks, width=512, height=512, sh_degree=None, eps2d=0.3,
                render_mode="RGB", near_plane=0.01, packed=True)
        raster = raster.clamp(0, 1).cpu().numpy()
        mask = torch.ones(1, len(gaussians), dtype=torch.bool, device=device)
        model = pipe(torch.from_numpy(gaussians)[None].to(device), mask, torch.from_numpy(c2w[views])[None].to(device),
                     torch.from_numpy(fov[views])[None].to(device))[0].float().clamp(0, 1).cpu().numpy()
        for j, v in enumerate(views):
            gt = iio.imread(a.data / "renders" / f"{name}_view_{v}.png")[..., :3].astype(np.float32) / 255.0
            c_gt, c_raster, c_model = crop([gt, raster[j], model[j]], gt)
            with torch.no_grad():
                pair = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].to(device) * 2 - 1
                lp_model = lpips_fn(pair(c_model), pair(c_gt)).item()
                lp_raster = lpips_fn(pair(c_raster), pair(c_gt)).item()
            rows.append(dict(scene=scene, view=v, distance=DISTANCES[group_of[v]][0] if v < len(group_of) else None,
                             psnr=psnr(c_model, c_gt), lpips=lp_model, raster_psnr=psnr(c_raster, c_gt),
                             raster_lpips=lp_raster))
        print(f"{name}: {np.mean([r['psnr'] for r in rows if r['scene'] == scene]):.2f} dB", flush=True)
    if a.out:
        a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"\n{'distance':>8}  {'views':>6}  {'PSNR':>6}  {'LPIPS':>6}  {'raster PSNR':>11}  {'raster LPIPS':>12}")
    for distance, _ in DISTANCES:
        sel = [r for r in rows if r["distance"] == distance]
        if sel:
            m = {k: np.mean([r[k] for r in sel]) for k in ("psnr", "lpips", "raster_psnr", "raster_lpips")}
            print(f"{distance:>8}  {len(sel):>6}  {m['psnr']:6.2f}  {m['lpips']:6.4f}  {m['raster_psnr']:11.2f}  "
                  f"{m['raster_lpips']:12.4f}")


if __name__ == "__main__":
    main()
