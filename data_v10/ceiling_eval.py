"""Characterise the recovered-input CEILING (rec-GT) at scale, and V17's margin below it.

Every headline number so far came from a handful of hand-picked scenes -- and those scenes live in
data_v10/h5s_20k_rec, which is exactly what V16/V17 trained on (training/dataset.py globs the whole
directory; V17's log confirms "13405 scenes"). So the ~10 dB rec-GT vs V17 gap we keep quoting is a
TRAINING-set gap measured on 3 objects. This script measures it properly:

  * rec-GT vs GT  -- the ceiling itself, never characterised beyond anecdote
  * V17   vs GT   -- the model
  * margin = rec-GT - V17, as a DISTRIBUTION rather than a single number

over three slices (see SPLITS / the driver script): true held-out val, memorised train, and the
2x-expansion objects (idx >= 15000) that V17 never saw.

Metrics use the SAME convention as our reported numbers -- `_fg_crop` is imported from
model_on_v10 rather than reimplemented, so the crop cannot silently drift. Whole-image metrics are
recorded alongside purely as the foil (objects are 2-5% of pixels).

Rows stream to JSONL and completed (scene, view) pairs are skipped on restart, because this runs
--killable and WILL be preempted mid-shard.

  uv run --frozen python -m data_v10.ceiling_eval --split val \
    --scenes_file data_v10/ceiling_scenes/val_00.json --out data_v10/ceiling/val_00.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gsplat
import h5py
import numpy as np
import torch
import lpips as lpips_lib

from data_external.orbit import make_orbit_views
from render_compare import load_model, ModelSpec, load_gt
from infer_gaussian import load_single_gaussian_h5_data
from data_v10.model_on_v10 import _fg_crop

RADIUS, FOV, RES = 1.7, 45.0, 512

# (h5 dir, GT renders dir). 'train' and 'unseen2x' share a directory -- they are distinguished by
# scene index, not by location: V17 trained on every h5 with idx < 15000 that existed on Jul 23,
# and the 2x expansion appended idx >= 15000 afterwards.
SPLITS = {
    "train": (Path("data_v10/h5s_20k_rec"), Path("data_v10/renders")),
    "unseen2x": (Path("data_v10/h5s_20k_rec"), Path("data_v10/renders")),
    "val": (Path("data_v10/h5s_20k_rec_val"), Path("data_v10/renders_val")),
}


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return 10.0 * np.log10(1.0 / (float(((a - b) ** 2).mean()) + 1e-12))


def render_rec_all(h5, views, vm, K, res, device) -> np.ndarray:
    """Rasterize the recovered Gaussians for ALL views in one gsplat call.

    render_compare.render_pruned_gt reopens the h5 and issues one call per view; at 14 views that
    is 14 NFS reads of the same 1.1 MB payload. Same gsplat arguments as process_full.render_full,
    so the output is bit-comparable to how the GT PNGs were made.
    """
    with h5py.File(h5, "r") as f:
        g = {k: torch.from_numpy(np.array(f[k], dtype=np.float32)).to(device)
             for k in ("means", "scales", "rotations", "colors", "opacities")}
    op = g["opacities"].squeeze(-1) if g["opacities"].ndim == 2 else g["opacities"]
    idx = torch.as_tensor(views, device=device)
    with torch.no_grad():
        img, _, _ = gsplat.rasterization(
            means=g["means"], quats=g["rotations"], scales=g["scales"],
            opacities=op, colors=g["colors"],
            viewmats=vm[idx], Ks=K[idx], width=res, height=res,
            sh_degree=None, eps2d=0.3, render_mode="RGB", near_plane=0.01, packed=True,
        )
    return img.clamp(0, 1).cpu().numpy()


def render_model_all(pipe, data, views, res, chunk) -> np.ndarray:
    """Model renders for ALL views, batching views into single pipeline calls.

    The pipeline encodes the 20k-Gaussian sequence ONCE per call and only the ray/decoder stages
    scale with nv (rendering_pipeline.py:52-93), so feeding views one at a time re-runs the
    quadratic view-independent encoder 14x per object. Chunked rather than all-14 because the DPT
    decoder allocates per view at 512^2 -- --view_chunk is the knob if a node OOMs.
    """
    out = []
    for i in range(0, len(views), chunk):
        idx = torch.as_tensor(views[i:i + chunk], device=data["c2w"].device)
        with torch.no_grad():
            r = pipe(
                gaussians=data["gaussians"].unsqueeze(0), mask=data["mask"].unsqueeze(0),
                c2w=data["c2w"][idx].unsqueeze(0), fov=data["fov"][idx].unsqueeze(0),
                resolution=res, torch_dtype=torch.bfloat16,
            )
        out.append(r[0].cpu().float().numpy())
    return np.clip(np.concatenate(out, axis=0), 0, 1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=sorted(SPLITS), required=True)
    ap.add_argument("--scenes_file", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0, help="this task's index within --nshards")
    ap.add_argument("--nshards", type=int, default=1,
                    help="stride-shard the scene list: scenes[shard::nshards]. Stride rather than "
                    "contiguous ranges -- scene indices are grouped by source chunk, so contiguous "
                    "blocks differ in object complexity and the array waits on the slowest task.")
    ap.add_argument("--limit", type=int, default=0, help="cap scenes (rate-measurement runs)")
    ap.add_argument("--view_chunk", type=int, default=7,
                    help="views per pipeline call; lower it if the DPT decoder OOMs on a 45G node")
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints_v17_512lp/phase2_epoch_36.pt"))
    ap.add_argument("--label", default="V17-ep36")
    ap.add_argument("--model_tag", default="v17",
                    help="stamped into every row and the output filename, so a later V18 sweep "
                    "lands alongside V17's rows instead of colliding with them")
    ap.add_argument("--pe_type", default="rope")
    ap.add_argument("--encoder_layers", type=int, default=12)
    ap.add_argument("--view_layers", type=int, default=6)
    ap.add_argument("--ffn_mult", type=int, default=4)
    ap.add_argument("--input_mlp_hidden", type=int, default=0)
    ap.add_argument("--log_scale_input", action="store_true",
                    help="must match the checkpoint's training-time input transform")
    ap.add_argument("--views", default="all", help="'all' (0..13) or comma-separated indices")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    h5_dir, ren_dir = SPLITS[args.split]
    views = list(range(14)) if args.views == "all" else [int(v) for v in args.views.split(",")]
    scenes = [int(x) for x in json.loads(args.scenes_file.read_text())][args.shard::args.nshards]
    if args.limit:
        scenes = scenes[:args.limit]
    print(f"shard {args.shard}/{args.nshards}: {len(scenes)} scenes x {len(views)} views "
          f"= {len(scenes) * len(views)} rows", flush=True)

    # Resume: a preempted shard leaves a partial JSONL. Skip what it already holds.
    done: set[tuple[int, int]] = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["scene"], r["view"]))
        print(f"resume: {len(done)} rows already in {args.out}", flush=True)

    vm_np, K_np = make_orbit_views(14, RADIUS, FOV, RES, up_axis="y")
    vm, K = torch.from_numpy(vm_np).to(device), torch.from_numpy(K_np).to(device)
    if (args.encoder_layers != 12 or args.view_layers != 6
            or args.ffn_mult != 4 or args.input_mlp_hidden):
        # depth-pruned checkpoints need a matching config; render_compare.load_model hardcodes
        # the default depth, so build the pipeline inline for this case
        from gaussianformer.models.config import GaussianFormerConfig
        from gaussianformer.models.gaussianformer import GaussianFormer
        from gaussianformer.pipelines.rendering_pipeline import GaussianFormerRenderingPipeline
        cfg = GaussianFormerConfig(pe_type=args.pe_type, num_layers=args.encoder_layers,
                                   view_transformer_n_layers=args.view_layers,
                                   dim_feedforward=768 * args.ffn_mult,
                                   view_transformer_ffn_hidden_dim=768 * args.ffn_mult,
                                   input_mlp_hidden=args.input_mlp_hidden)
        model = GaussianFormer(cfg)
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        pipe = GaussianFormerRenderingPipeline(model)
        pipe.to(device)
    else:
        pipe = load_model(ModelSpec(ckpt=args.ckpt, label=args.label, pe_type=args.pe_type), device)
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()

    def lp_many(pairs: list[tuple[np.ndarray, np.ndarray]]) -> list[float]:
        """All four comparisons for a row in one batched LPIPS forward."""
        def stack(i: int) -> torch.Tensor:
            t = torch.from_numpy(np.stack([p[i] for p in pairs])).permute(0, 3, 1, 2)
            return t.to(device) * 2 - 1
        with torch.no_grad():
            return lpips_fn(stack(0), stack(1)).flatten().tolist()

    n_rows, n_skip = 0, 0
    t0 = time.monotonic()
    with args.out.open("a") as fh:
        for s in scenes:
            todo = [v for v in views if (s, v) not in done]
            if not todo:
                continue
            h5 = h5_dir / f"scene_{s:04d}.h5"
            if not h5.exists():
                print(f"skip scene_{s:04d}: no recovered h5", flush=True)
                n_skip += 1
                continue

            data = load_single_gaussian_h5_data(h5)
            if args.log_scale_input:
                data["gaussians"][:, 3:6] = torch.log10(data["gaussians"][:, 3:6].clamp(min=1e-8)) + 3.0
            for k in ("gaussians", "mask", "c2w", "fov"):
                data[k] = data[k].to(device)

            recs = render_rec_all(h5, todo, vm, K, RES, device)
            mdls = render_model_all(pipe, data, todo, RES, args.view_chunk)

            for j, v in enumerate(todo):
                gt_path = ren_dir / f"scene_{s:04d}_view_{v}.png"
                if not gt_path.exists():
                    n_skip += 1
                    continue
                gt = load_gt(gt_path, RES)
                rec, mdl = recs[j], mdls[j]

                # Object footprint: the covariate that decides whether whole-image PSNR is
                # measuring the object or the black background.
                fg_frac = float((gt.mean(-1) > 0.02).mean())
                c_gt, c_rec, c_mdl = _fg_crop([gt, rec, mdl], gt, pad=12, out=RES)
                l_rec_fg, l_mdl_fg, l_rec_full, l_mdl_full = lp_many(
                    [(c_rec, c_gt), (c_mdl, c_gt), (rec, gt), (mdl, gt)])

                row = dict(
                    scene=s, view=v, split=args.split, model=args.model_tag,
                    fg_frac=round(fg_frac, 5),
                    rec_psnr_fg=round(psnr(c_rec, c_gt), 3),
                    rec_lpips_fg=round(l_rec_fg, 5),
                    mdl_psnr_fg=round(psnr(c_mdl, c_gt), 3),
                    mdl_lpips_fg=round(l_mdl_fg, 5),
                    rec_psnr_full=round(psnr(rec, gt), 3),
                    rec_lpips_full=round(l_rec_full, 5),
                    mdl_psnr_full=round(psnr(mdl, gt), 3),
                    mdl_lpips_full=round(l_mdl_full, 5),
                )
                row["margin_psnr_fg"] = round(row["rec_psnr_fg"] - row["mdl_psnr_fg"], 3)
                row["margin_lpips_fg"] = round(row["mdl_lpips_fg"] - row["rec_lpips_fg"], 5)
                fh.write(json.dumps(row) + "\n")
                n_rows += 1
            fh.flush()  # survive preemption at scene granularity
            print(f"scene_{s:04d} [{args.split}] done ({len(todo)} views)", flush=True)

    dt = time.monotonic() - t0
    rate = dt / n_rows if n_rows else float("nan")
    print(f"DONE_CEILING {args.split} shard {args.shard}/{args.nshards}: {n_rows} rows, "
          f"{n_skip} skipped, {dt:.0f}s ({rate:.3f} s/row)", flush=True)


if __name__ == "__main__":
    main()
