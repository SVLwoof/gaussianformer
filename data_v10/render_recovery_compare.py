"""Render full-GT | naive-20k | recovered-20k strips from the SAVED recovered H5s (no
re-recovery), to visualise the prune-and-recovery gain on detailed objects.

  uv run --frozen python -m data_v10.render_recovery_compare --scenes_file <json> --out <dir>
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont

from data_external.orbit import make_orbit_views
from data_v10.prune_recovery import load_full, significance_topk, to_dev, rasterize, RADIUS, FOV, RES


def render_arr(a, vm, K, vi, device):
    p = to_dev(a, device)
    p = dict(means=p["means"], quats=p["rotations"], scales=p["scales"], opacities=p["opacities"], colors=p["colors"])
    with torch.no_grad():
        return rasterize(p, vm, K, slice(vi, vi + 1), RES)[0].clamp(0, 1).cpu().numpy()


def psnr(a, b): return 10.0 * np.log10(1.0 / (float(((a - b) ** 2).mean()) + 1e-12))


def lbl(img, t, font):
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im); d.rectangle([0, 0, im.width, 28], fill=(0, 0, 0)); d.text((5, 4), t, fill=(255, 230, 0), font=font)
    return np.asarray(im).astype(np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes_file", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--views", default="0,6")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    views = [int(v) for v in args.views.split(",")]
    vm_np, K_np = make_orbit_views(14, RADIUS, FOV, RES, up_axis="y")
    vm, K = torch.from_numpy(vm_np).to(device), torch.from_numpy(K_np).to(device)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)

    rows = []
    for s in [int(x) for x in json.loads(args.scenes_file.read_text())]:
        fp = Path("data_v10/full_h5s") / f"scene_{s:04d}.h5"
        rp = Path("data_v10/h5s_20k_rec") / f"scene_{s:04d}.h5"
        if not (fp.exists() and rp.exists()):
            print(f"skip {s}: missing"); continue
        full = load_full(fp)
        naive = significance_topk(full, vm, K, 20000, device)
        rec = load_full(rp)
        for vi in views:
            f = render_arr(full, vm, K, vi, device)
            n = render_arr(naive, vm, K, vi, device)
            r = render_arr(rec, vm, K, vi, device)
            rows.append(np.concatenate([
                lbl(f, f"full-GT s{s} v{vi}", font),
                lbl(n, f"naive-20k  {psnr(n, f):.1f}dB", font),
                lbl(r, f"recovered-20k  {psnr(r, f):.1f}dB", font)], axis=1))
        print(f"scene_{s:04d}: naive {psnr(render_arr(naive,vm,K,0,device), render_arr(full,vm,K,0,device)):.1f} -> "
              f"recovered {psnr(render_arr(rec,vm,K,0,device), render_arr(full,vm,K,0,device)):.1f} dB (v0)", flush=True)
    grid = (np.clip(np.concatenate(rows, axis=0), 0, 1) * 255).astype(np.uint8)
    iio.imwrite(args.out / "recovery_compare.png", grid)
    print(f"wrote {args.out}/recovery_compare.png", flush=True)
    print("DONE_RENDERCMP", flush=True)


if __name__ == "__main__":
    main()
