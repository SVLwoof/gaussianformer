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
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    views = [int(v) for v in args.views.split(",")]

    vm_np, K_np = make_orbit_views(14, RADIUS, FOV, RES, up_axis="y")
    vm, K = torch.from_numpy(vm_np).to(device), torch.from_numpy(K_np).to(device)
    pipe = load_model(ModelSpec(ckpt=args.ckpt, label="V14best", pe_type=args.pe_type), device)
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)

    scenes = [int(x) for x in json.loads(args.scenes_file.read_text())]
    results, all_rows = [], []
    for s in scenes:
        h5 = Path("data_v10/full_h5s") / f"scene_{s:04d}.h5"
        if not h5.exists():
            print(f"skip {s}: missing"); continue
        full = load_full(h5)
        naive = significance_topk(full, vm, K, args.keep_n, device)  # 50k -> 20k (current method)
        # write a temp 20k h5 the model loader understands
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tf:
            tmp = Path(tf.name)
        with h5py.File(tmp, "w") as f, h5py.File(h5, "r") as src:
            for k in ("means", "scales", "colors"):
                f.create_dataset(k, data=naive[k])
            f.create_dataset("rotations", data=naive["rotations"])
            f.create_dataset("opacities", data=naive["opacities"].reshape(-1, 1))
            f.create_dataset("c2w", data=np.array(src["c2w"]))
            f.create_dataset("fov", data=np.array(src["fov"]))
        data = load_single_gaussian_h5_data(tmp)
        for k in ("gaussians", "mask", "c2w", "fov"):
            data[k] = data[k].to(device)

        pm, plp, ppm = [], [], []
        for v in views:
            gt = load_gt(Path("data_v10/renders") / f"scene_{s:04d}_view_{v}.png", RES)
            pgt = render_pruned_gt(tmp, v, vm, K, RES, device)
            mdl = render_view(pipe, data, v, RES, None)
            pm.append(psnr(mdl, gt)); ppm.append(psnr(pgt, gt))
            plp.append(float(lpips_fn(torch.from_numpy(mdl).permute(2,0,1)[None].to(device)*2-1,
                                      torch.from_numpy(gt).permute(2,0,1)[None].to(device)*2-1).item()))
            lab = lambda im, t: np.array(_label(im, t, font))
            all_rows.append(np.concatenate([
                _label(gt, f"GT s{s} v{v}", font),
                _label(pgt, f"pruned-GT {psnr(pgt,gt):.1f}", font),
                _label(mdl, f"V14best {psnr(mdl,gt):.1f}dB", font)], axis=1))
        r = dict(scene=s, model_psnr=float(np.mean(pm)), model_lpips=float(np.mean(plp)),
                 pruned_psnr=float(np.mean(ppm)), new_object=(s > 3000))
        results.append(r)
        tmp.unlink(missing_ok=True)
        print(f"scene_{s:04d} ({'NEW' if s>3000 else 'seen'}): V14best vs GT {r['model_psnr']:.2f}dB / "
              f"LPIPS {r['model_lpips']:.4f}  (pruned-GT {r['pruned_psnr']:.2f})", flush=True)

    grid = (np.clip(np.concatenate(all_rows, axis=0), 0, 1) * 255).astype(np.uint8)
    iio.imwrite(args.out / "v14best_on_v10.png", grid)
    (args.out / "v14best_on_v10.json").write_text(json.dumps(results, indent=1))
    if results:
        print(f"\nMEAN V14best-vs-GT: {np.mean([r['model_psnr'] for r in results]):.2f}dB | "
              f"NEW objects only: {np.mean([r['model_psnr'] for r in results if r['new_object']] or [0]):.2f}dB", flush=True)
    print("DONE_MODELEVAL", flush=True)


def _label(img, text, font):
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im); d.rectangle([0, 0, im.width, 26], fill=(0, 0, 0)); d.text((4, 3), text, fill=(255, 230, 0), font=font)
    return np.asarray(im).astype(np.float32) / 255.0


if __name__ == "__main__":
    main()
