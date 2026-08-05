"""Paper showcase: render random UNSEEN objects through each GaussianFormer generation, saving
ONE clean PNG per (scene, view, model). No grids, no labels, no side-by-side -- those are
composed later from these raw renders. Filenames encode everything:

    data_v10/showcase/s{scene:05d}_v{view:02d}_{model}.png

Each model is fed the input distribution it was TRAINED on (V14best on naive-pruned 20k, V15+ on
recovered 20k) -- feeding a model the wrong distribution scores it out-of-distribution and would
mis-state the comparison (project memory: V14best gets WORSE on recovered input). The matching GT
already lives at data_v10/renders/scene_{scene:04d}_view_{view}.png -- not re-saved here.

A lightweight metrics manifest (PSNR/LPIPS per render vs GT) is written for later selection; it is
metadata, not a comparison image.

  uv run --frozen python -m data_v10.showcase_versions \\
    --scenes_file <json list of ints> --views 0,3,7,10 --out_name showcase_shardK
"""
from __future__ import annotations
import argparse, json, tempfile
from pathlib import Path
import numpy as np, torch, h5py
import imageio.v3 as iio
import lpips as lpips_lib

from data_external.orbit import make_orbit_views
from render_compare import load_model, ModelSpec, render_view, load_gt
from infer_gaussian import load_single_gaussian_h5_data
from data_v10.prune_recovery import load_full, significance_topk

RADIUS, FOV, RES = 1.7, 45.0, 512

# Best-of-generation lineup. (filename_tag, ckpt, input_mode). Clean tags -> clean filenames.
MODELS = [
    ("V14best",   "checkpoints_v14auglp10/phase2_epoch_10.pt", "naive"),
    ("V16-LPIPS", "checkpoints_v16_lpips_mid/phase2_epoch_12.pt", "recovered"),
    ("V17-ep36",  "checkpoints_v17_512lp/phase2_epoch_36.pt", "recovered"),
]


def psnr(a, b):
    return 10.0 * np.log10(1.0 / (float(((a - b) ** 2).mean()) + 1e-12))


def _build_input(scene, mode, vm, K, keep_n, device):
    """Return (h5_path, is_temp) for one scene under `mode`, or (None, False) if unavailable."""
    if mode == "recovered":
        inp = Path("data_v10/h5s_20k_rec") / f"scene_{scene:04d}.h5"
        return (inp, False) if inp.exists() else (None, False)
    # naive: significance_topk 50k -> keep_n (V14best training match)
    full_h5 = Path("data_v10/full_h5s") / f"scene_{scene:04d}.h5"
    if not full_h5.exists():
        return None, False
    full = load_full(full_h5)
    naive = significance_topk(full, vm, K, keep_n, device)
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tf:
        tmp = Path(tf.name)
    with h5py.File(tmp, "w") as f, h5py.File(full_h5, "r") as src:
        for k in ("means", "scales", "colors"):
            f.create_dataset(k, data=naive[k])
        f.create_dataset("rotations", data=naive["rotations"])
        f.create_dataset("opacities", data=naive["opacities"].reshape(-1, 1))
        f.create_dataset("c2w", data=np.array(src["c2w"]))
        f.create_dataset("fov", data=np.array(src["fov"]))
    return tmp, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes_file", type=Path)
    ap.add_argument("--scenes", type=str, help="comma-separated (overrides --scenes_file)")
    ap.add_argument("--views", default="0,3,7,10")
    ap.add_argument("--keep_n", type=int, default=20000)
    ap.add_argument("--out", type=Path, default=Path("data_v10/showcase"))
    ap.add_argument("--out_name", default="showcase")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    views = [int(v) for v in args.views.split(",")]
    if args.scenes:
        scenes = [int(x) for x in args.scenes.split(",")]
    else:
        scenes = [int(x) for x in json.loads(args.scenes_file.read_text())]

    vm_np, K_np = make_orbit_views(14, RADIUS, FOV, RES, up_axis="y")
    vm, K = torch.from_numpy(vm_np).to(device), torch.from_numpy(K_np).to(device)
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()

    pipes = []
    for label, ckpt, mode in MODELS:
        print(f"loading {label} ({mode}): {ckpt}", flush=True)
        pipes.append((label, mode, load_model(ModelSpec(ckpt=Path(ckpt), label=label, pe_type="rope"), device)))
    modes_needed = {m for _, _, m in MODELS}

    manifest, n_png = [], 0
    for s in scenes:
        inputs, ok = {}, True
        for mode in modes_needed:
            h5, is_tmp = _build_input(s, mode, vm, K, args.keep_n, device)
            if h5 is None:
                print(f"skip scene_{s:05d}: missing {mode} input", flush=True); ok = False; break
            data = load_single_gaussian_h5_data(h5)
            for k in ("gaussians", "mask", "c2w", "fov"):
                data[k] = data[k].to(device)
            inputs[mode] = (data, h5 if is_tmp else None)
        if not ok:
            for _, tmp in inputs.values():
                if tmp is not None:
                    tmp.unlink(missing_ok=True)
            continue

        srow = dict(scene=s, new_object=(s > 3000), views={})
        for v in views:
            gt_path = Path("data_v10/renders") / f"scene_{s:04d}_view_{v}.png"
            gt = load_gt(gt_path, RES) if gt_path.exists() else None
            vmetrics = {}
            for label, mode, pipe in pipes:
                data, _ = inputs[mode]
                mdl = render_view(pipe, data, v, RES, None)
                iio.imwrite(args.out / f"s{s:05d}_v{v:02d}_{label}.png",
                            (np.clip(mdl, 0, 1) * 255).astype(np.uint8))
                n_png += 1
                if gt is not None:
                    p = psnr(mdl, gt)
                    lp = float(lpips_fn(
                        torch.from_numpy(mdl).permute(2, 0, 1)[None].to(device) * 2 - 1,
                        torch.from_numpy(gt).permute(2, 0, 1)[None].to(device) * 2 - 1).item())
                    vmetrics[label] = dict(psnr=round(float(p), 3), lpips=round(lp, 4))
            if vmetrics:
                srow["views"][v] = vmetrics
                d = vmetrics.get("V17-ep36", {}).get("psnr", 0) - vmetrics.get("V14best", {}).get("psnr", 0)
                print(f"scene_{s:05d} v{v}: V14 {vmetrics['V14best']['psnr']:.1f} -> "
                      f"V17 {vmetrics['V17-ep36']['psnr']:.1f} ({d:+.1f}dB)", flush=True)
            else:
                print(f"scene_{s:05d} v{v}: rendered (no GT for metrics)", flush=True)

        manifest.append(srow)
        for _, tmp in inputs.values():
            if tmp is not None:
                tmp.unlink(missing_ok=True)

    (args.out / f"{args.out_name}_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"DONE_SHOWCASE {args.out_name}: {len(manifest)} scenes, {n_png} renders", flush=True)


if __name__ == "__main__":
    main()
