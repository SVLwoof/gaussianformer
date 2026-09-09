"""GT | rec-GT | model columns for N=10 probe checkpoints, foreground-cropped like the metrics.

  uv run --no-sync python -m data_v10.probe_strips --out tmp/probe_strips.png \
     --models "seed:checkpoints_v18_256/phase2_epoch_30.pt:" \
              "p2_bias_feat:checkpoints_probe_p2_bias_feat/phase2_epoch_3000.pt:proj_bias=true proj_feat=true" \
     --scenes 387 4447 12092 --views 0 7
model spec = label:ckpt:model_cfg (space-separated key=val, may be empty).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, torch, imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont
from data_external.orbit import make_orbit_views
from render_compare import load_gt
from infer_gaussian import load_single_gaussian_h5_data
from data_v10.model_on_v10 import _fg_crop
from data_v10.ceiling_eval import SPLITS, RADIUS, FOV, RES, psnr, render_rec_all, render_model_all
from gaussianformer.models.config import GaussianFormerConfig
from gaussianformer.models.gaussianformer import GaussianFormer
from gaussianformer.pipelines.rendering_pipeline import GaussianFormerRenderingPipeline

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def load_pipe(ckpt: Path, cfg_pairs: list[str], device):
    cfg = GaussianFormerConfig(pe_type="rope").with_overrides(cfg_pairs)
    model = GaussianFormer(cfg)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["model_state_dict"]
    own = model.state_dict()
    sd = {k: v for k, v in sd.items() if not (k.endswith(".freqs") and (k not in own or own[k].shape != v.shape))}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    bad = [k for k in missing if not k.endswith(".freqs") and own[k].abs().sum() > 0]
    assert not bad, bad
    pipe = GaussianFormerRenderingPipeline(model.eval())
    pipe.to(device)  # returns None, not self
    return pipe


def band(img, text, font, colour=(255, 230, 0), size=384):
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
    d = ImageDraw.Draw(im); d.rectangle([0, 0, size, 24], fill=(0, 0, 0)); d.text((4, 3), text, fill=colour, font=font)
    return np.asarray(im).astype(np.float32) / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--scenes", nargs="+", type=int, required=True)
    ap.add_argument("--views", nargs="+", type=int, default=[0, 7])
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--panel", type=int, default=384)
    a = ap.parse_args()
    device = "cuda"
    vm_np, K_np = make_orbit_views(14, RADIUS, FOV, RES, up_axis="y")
    vm, K = torch.from_numpy(vm_np).to(device), torch.from_numpy(K_np).to(device)
    font = ImageFont.truetype(FONT, 17)
    specs = []
    for m in a.models:
        label, ckpt, cfg = m.split(":", 2)
        specs.append((label, Path(ckpt), cfg.split()))
    h5_dir, ren_dir = SPLITS[a.split]
    rows = []
    # one model in memory at a time (195 M params each); cache renders per (scene, view)
    renders = {lab: {} for lab, _, _ in specs}
    for label, ckpt, cfg in specs:
        pipe = load_pipe(ckpt, cfg, device)
        for s in a.scenes:
            data = load_single_gaussian_h5_data(h5_dir / f"scene_{s:04d}.h5")
            for k in ("gaussians", "mask", "c2w", "fov"):
                data[k] = data[k].to(device)
            out = render_model_all(pipe, data, a.views, RES, 7)
            for j, v in enumerate(a.views):
                renders[label][(s, v)] = out[j]
        del pipe; torch.cuda.empty_cache()
        print(f"rendered {label}", flush=True)
    for s in a.scenes:
        recs = render_rec_all(h5_dir / f"scene_{s:04d}.h5", a.views, vm, K, RES, device)
        for j, v in enumerate(a.views):
            gt = load_gt(ren_dir / f"scene_{s:04d}_view_{v}.png", RES)
            imgs = [gt, recs[j]] + [renders[lab][(s, v)] for lab, _, _ in specs]
            crops = _fg_crop(imgs, gt, pad=12, out=RES)
            p_rec = psnr(crops[1], crops[0])
            panels = [band(crops[0], f"GT  s{s} v{v}", font, (255, 255, 255), a.panel),
                      band(crops[1], f"rec-GT  {p_rec:.1f} dB", font, (255, 230, 0), a.panel)]
            for (lab, _, _), c in zip(specs, crops[2:]):
                p = psnr(c, crops[0])
                panels.append(band(c, f"{lab}  {p:.1f} dB ({p - p_rec:+.1f})", font,
                                   (120, 255, 160) if p >= p_rec else (255, 160, 120), a.panel))
                print(f"s{s} v{v} {lab}: {p:.2f} dB (rec-GT {p_rec:.2f})", flush=True)
            rows.append(np.concatenate(panels, 1))
    grid = (np.clip(np.concatenate(rows, 0), 0, 1) * 255).astype(np.uint8)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(a.out, grid)
    print(f"wrote {a.out} {grid.shape[1]}x{grid.shape[0]}  DONE_STRIPS", flush=True)


if __name__ == "__main__":
    main()
