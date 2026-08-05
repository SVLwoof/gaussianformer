"""Run the current best model (V14best) on data_v10 objects (naive-pruned 20k) to see how it
performs on the new data -- including NEW objects (scene_idx>3000) never seen in training
(= a cross-object generalisation spot-check). Builds GT | pruned-GT | model strips + metrics.

  uv run --frozen python -m data_v10.model_on_v10 --scenes_file data_v10/scenes_show.json \
    --ckpt checkpoints_v14auglp10/phase2_epoch_10.pt --out data_v10/model_eval
"""
from __future__ import annotations
import argparse, json, tempfile
from pathlib import Path
import numpy as np, torch, gsplat, h5py
import imageio.v3 as iio
import lpips as lpips_lib
from PIL import Image, ImageDraw, ImageFont

from data_external.orbit import make_orbit_views
from render_compare import load_model, ModelSpec, render_view, load_gt, render_pruned_gt
from infer_gaussian import load_single_gaussian_h5_data
from data_v10.prune_recovery import load_full, significance_topk

RADIUS, FOV, RES = 1.7, 45.0, 512


def psnr(a, b): return 10.0 * np.log10(1.0 / (float(((a - b) ** 2).mean()) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes_file", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints_v14auglp10/phase2_epoch_10.pt"))
    ap.add_argument("--pe_type", default="rope")
    ap.add_argument("--keep_n", type=int, default=20000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--views", default="0,3,6")
    ap.add_argument("--label", default="V14best", help="model panel label + console tag")
    ap.add_argument("--out_name", default="v14best_on_v10", help="output png/json stem")
    ap.add_argument("--input_mode", choices=["naive", "recovered"], default="naive",
                    help="naive = significance_topk 50k->20k (V14best training match); "
                    "recovered = load data_v10/h5s_20k_rec/<scene>.h5 (V15 training match)")
    ap.add_argument("--crop_fg", action="store_true",
                    help="Object-only eval: crop to the GT object bbox + compute PSNR/LPIPS "
                    "there (background is ~95-98% black -> whole-image PSNR is misleading).")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    views = [int(v) for v in args.views.split(",")]

    vm_np, K_np = make_orbit_views(14, RADIUS, FOV, RES, up_axis="y")
    vm, K = torch.from_numpy(vm_np).to(device), torch.from_numpy(K_np).to(device)
    pipe = load_model(ModelSpec(ckpt=args.ckpt, label=args.label, pe_type=args.pe_type), device)
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)

    scenes = [int(x) for x in json.loads(args.scenes_file.read_text())]
    results, all_rows = [], []
    for s in scenes:
        if args.input_mode == "recovered":
            # V15 training match: feed the recovered-20k h5 directly (model input + pruned-GT).
            inp = Path("data_v10/h5s_20k_rec") / f"scene_{s:04d}.h5"
            tmp = None
            if not inp.exists():
                print(f"skip {s}: no recovered h5"); continue
        else:
            # naive significance_topk 50k -> 20k (V14best training match).
            full_h5 = Path("data_v10/full_h5s") / f"scene_{s:04d}.h5"
            if not full_h5.exists():
                print(f"skip {s}: missing"); continue
            full = load_full(full_h5)
            naive = significance_topk(full, vm, K, args.keep_n, device)
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tf:
                tmp = Path(tf.name)
            with h5py.File(tmp, "w") as f, h5py.File(full_h5, "r") as src:
                for k in ("means", "scales", "colors"):
                    f.create_dataset(k, data=naive[k])
                f.create_dataset("rotations", data=naive["rotations"])
                f.create_dataset("opacities", data=naive["opacities"].reshape(-1, 1))
                f.create_dataset("c2w", data=np.array(src["c2w"]))
                f.create_dataset("fov", data=np.array(src["fov"]))
            inp = tmp
        data = load_single_gaussian_h5_data(inp)
        for k in ("gaussians", "mask", "c2w", "fov"):
            data[k] = data[k].to(device)

        pm, plp, ppm = [], [], []
        for v in views:
            gt = load_gt(Path("data_v10/renders") / f"scene_{s:04d}_view_{v}.png", RES)
            pgt = render_pruned_gt(inp, v, vm, K, RES, device)
            mdl = render_view(pipe, data, v, RES, None)
            if args.crop_fg:
                # Object-only eval: crop all panels to the GT object bbox (the background is
                # ~95-98% black and inflates whole-image PSNR). Metrics computed on the crop;
                # crop upscaled to RES for a zoomed, honest visual.
                gt, pgt, mdl = _fg_crop([gt, pgt, mdl], gt, pad=12, out=RES)
            pm.append(psnr(mdl, gt)); ppm.append(psnr(pgt, gt))
            plp.append(float(lpips_fn(torch.from_numpy(mdl).permute(2,0,1)[None].to(device)*2-1,
                                      torch.from_numpy(gt).permute(2,0,1)[None].to(device)*2-1).item()))
            pgt_tag = "rec-GT" if args.input_mode == "recovered" else "pruned-GT"
            all_rows.append(np.concatenate([
                _label(gt, f"GT s{s} v{v}", font),
                _label(pgt, f"{pgt_tag} {psnr(pgt,gt):.1f}", font),
                _label(mdl, f"{args.label} {psnr(mdl,gt):.1f}dB", font)], axis=1))
        r = dict(scene=s, model_psnr=float(np.mean(pm)), model_lpips=float(np.mean(plp)),
                 pruned_psnr=float(np.mean(ppm)), new_object=(s > 3000))
        results.append(r)
        if tmp is not None:
            tmp.unlink(missing_ok=True)
        print(f"scene_{s:04d} ({'NEW' if s>3000 else 'seen'}): {args.label} vs GT {r['model_psnr']:.2f}dB / "
              f"LPIPS {r['model_lpips']:.4f}  (pruned-GT {r['pruned_psnr']:.2f})", flush=True)

    grid = (np.clip(np.concatenate(all_rows, axis=0), 0, 1) * 255).astype(np.uint8)
    iio.imwrite(args.out / f"{args.out_name}.png", grid)
    (args.out / f"{args.out_name}.json").write_text(json.dumps(results, indent=1))
    if results:
        print(f"\nMEAN {args.label}-vs-GT: {np.mean([r['model_psnr'] for r in results]):.2f}dB | "
              f"NEW objects only: {np.mean([r['model_psnr'] for r in results if r['new_object']] or [0]):.2f}dB", flush=True)
    print("DONE_MODELEVAL", flush=True)


def _fg_crop(imgs, ref, pad=12, out=512):
    """Crop every img to ref's object bounding box (non-black region), square + padded,
    then resize to out x out (nearest = honest, no smoothing). Foreground-only view."""
    lum = ref.mean(-1)
    ys, xs = np.where(lum > 0.02)
    if len(ys) < 10:
        return imgs
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    half = max(y1 - y0, x1 - x0) // 2 + pad
    H = ref.shape[0]
    y0, y1 = max(0, cy - half), min(H, cy + half)
    x0, x1 = max(0, cx - half), min(H, cx + half)
    cropped = []
    for im in imgs:
        c = im[y0:y1, x0:x1]
        pil = Image.fromarray((np.clip(c, 0, 1) * 255).astype(np.uint8)).resize((out, out), Image.NEAREST)
        cropped.append(np.array(pil, dtype=np.float32) / 255.0)
    return cropped


def _label(img, text, font):
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im); d.rectangle([0, 0, im.width, 26], fill=(0, 0, 0)); d.text((4, 3), text, fill=(255, 230, 0), font=font)
    return np.asarray(im).astype(np.float32) / 255.0


if __name__ == "__main__":
    main()
