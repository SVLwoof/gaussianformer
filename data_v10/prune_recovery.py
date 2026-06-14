"""Prune-and-recovery for N=20k (LightGaussian step 3, which our pipeline skipped).

For each object: score the full 50k -> keep top-N -> RECOVER (fine-tune the kept gaussians
with L1+SSIM against multi-view full-splat renders, densification off) so they co-adapt to
compensate for the removed 60%. Evaluates naive-topk vs recovered against the full splat on
held-out views (strict generalisation) + the 14 canonical model views.

  uv run --frozen python -m data_v10.prune_recovery --scenes 99,1441 --split train \
     --iters 1500 --out data_v10/recovery_eval --save_h5_dir data_v10/h5s_20k_rec
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import gsplat
import h5py
import lpips as lpips_lib
from data_external.orbit import make_orbit_views, look_at_blender, c2w_to_viewmat

RADIUS, FOV, RES = 1.7, 45.0, 512
GAMMA = 0.1   # scale exponent in the significance score (matches current pipeline use)


def load_full(h5: Path) -> dict:
    with h5py.File(h5, "r") as f:
        a = {k: np.array(f[k], np.float32) for k in ("means", "scales", "rotations", "colors", "opacities")}
    a["opacities"] = a["opacities"].reshape(-1)
    return a


def to_dev(a, device):
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in a.items()}


def rasterize(p, viewmats, Ks, vi, res):
    img, _, _ = gsplat.rasterization(
        means=p["means"], quats=p["quats"] / p["quats"].norm(dim=-1, keepdim=True),
        scales=p["scales"], opacities=p["opacities"], colors=p["colors"],
        viewmats=viewmats[vi], Ks=Ks[vi], width=res, height=res,
        sh_degree=None, eps2d=0.3, render_mode="RGB", near_plane=0.01, packed=True)
    return img  # [len(vi),H,W,3]


def significance_topk(a, viewmats, Ks, keep_n, device):
    t = to_dev(a, device)
    n = t["means"].shape[0]
    score = torch.zeros(n, device=device, dtype=torch.float64)
    for v in range(viewmats.shape[0]):
        with torch.no_grad():
            _, _, info = gsplat.rasterization(
                means=t["means"], quats=t["rotations"], scales=t["scales"],
                opacities=t["opacities"], colors=t["colors"],
                viewmats=viewmats[v:v+1], Ks=Ks[v:v+1], width=RES, height=RES,
                sh_degree=None, eps2d=0.3, render_mode="RGB", near_plane=0.01, packed=True)
        gids = info["gaussian_ids"]
        contrib = info["opacities"].double() * info["radii"][:, 0].double() * info["radii"][:, 1].double()
        score.scatter_add_(0, gids, contrib)
    score = score * (t["scales"].max(1).values.double() ** GAMMA)
    idx = torch.topk(score, keep_n).indices.cpu().numpy()
    return {k: a[k][idx] for k in a}


def _gauss_window(ch, ks=11, sigma=1.5, device="cpu"):
    g = torch.arange(ks, dtype=torch.float32, device=device) - ks // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2)); g = (g / g.sum())
    w = (g[:, None] * g[None, :])
    return w.expand(ch, 1, ks, ks).contiguous()


def ssim(x, y, win):  # x,y: [B,3,H,W]
    mu_x = F.conv2d(x, win, padding=5, groups=3); mu_y = F.conv2d(y, win, padding=5, groups=3)
    mx2, my2, mxy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sx = F.conv2d(x * x, win, padding=5, groups=3) - mx2
    sy = F.conv2d(y * y, win, padding=5, groups=3) - my2
    sxy = F.conv2d(x * y, win, padding=5, groups=3) - mxy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mxy + c1) * (2 * sxy + c2)) / ((mx2 + my2 + c1) * (sx + sy + c2))
    return s.mean()


def recover(a20k, target, viewmats, Ks, iters, device):
    """Fine-tune the kept gaussians (no densification) against precomputed target views."""
    t = to_dev(a20k, device)
    means = torch.nn.Parameter(t["means"])
    log_scales = torch.nn.Parameter(t["scales"].clamp_min(1e-8).log())
    quats = torch.nn.Parameter(t["rotations"])
    op_logit = torch.nn.Parameter(torch.logit(t["opacities"].clamp(1e-4, 1 - 1e-4)))
    col_logit = torch.nn.Parameter(torch.logit(t["colors"].clamp(1e-4, 1 - 1e-4)))
    opt = torch.optim.Adam([
        {"params": [means], "lr": 1.6e-4}, {"params": [log_scales], "lr": 5e-3},
        {"params": [quats], "lr": 1e-3}, {"params": [op_logit], "lr": 5e-2},
        {"params": [col_logit], "lr": 2.5e-3},
    ])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95 ** (1 / 400))
    win = _gauss_window(3, device=device)
    M = viewmats.shape[0]
    for it in range(iters):
        vi = torch.randint(0, M, (4,), device=device)
        p = dict(means=means, quats=quats, scales=log_scales.exp(),
                 opacities=op_logit.sigmoid(), colors=col_logit.sigmoid())
        img = rasterize(p, viewmats, Ks, vi, RES).clamp(0, 1)         # [4,H,W,3]
        tgt = target[vi]
        l1 = (img - tgt).abs().mean()
        ss = 1 - ssim(img.permute(0, 3, 1, 2), tgt.permute(0, 3, 1, 2), win)
        (0.8 * l1 + 0.2 * ss).backward()
        opt.step(); opt.zero_grad(); sched.step()
    with torch.no_grad():
        return dict(
            means=means.detach().cpu().numpy(),
            scales=log_scales.exp().detach().cpu().numpy(),
            rotations=(quats / quats.norm(dim=-1, keepdim=True)).detach().cpu().numpy(),
            colors=col_logit.sigmoid().detach().cpu().numpy(),
            opacities=op_logit.sigmoid().detach().cpu().numpy(),
        )


def psnr(a, b): return 10.0 * np.log10(1.0 / (float(((a - b) ** 2).mean()) + 1e-12))


def eval_set(a, full_t, viewmats, Ks, device, lpips_fn):
    p = {k: v for k, v in to_dev(a, device).items()}
    p = dict(means=p["means"], quats=p["rotations"], scales=p["scales"],
             opacities=p["opacities"], colors=p["colors"])
    ps, lp = [], []
    for v in range(viewmats.shape[0]):
        with torch.no_grad():
            pred = rasterize(p, viewmats, Ks, slice(v, v+1), RES)[0].clamp(0, 1)
            gt = rasterize(full_t, viewmats, Ks, slice(v, v+1), RES)[0].clamp(0, 1)
            ps.append(psnr(pred.cpu().numpy(), gt.cpu().numpy()))
            lp.append(float(lpips_fn((pred.permute(2,0,1)[None]*2-1), (gt.permute(2,0,1)[None]*2-1)).item()))
    return float(np.mean(ps)), float(np.mean(lp))


def heldout_views(n, device):
    vm = []
    for j in range(n):
        th = 2 * np.pi * (j + 0.5) / n
        eye = np.array([RADIUS*np.cos(th), 0.15*RADIUS, RADIUS*np.sin(th)], np.float32)
        vm.append(c2w_to_viewmat(look_at_blender(eye, np.zeros(3, np.float32), up=np.array([0,1,0], np.float32))))
    f = RES / (2*np.tan(0.5*FOV*np.pi/180)); K = np.array([[f,0,RES/2],[0,f,RES/2],[0,0,1]], np.float32)
    return torch.from_numpy(np.stack(vm)).to(device), torch.from_numpy(np.tile(K[None], (n,1,1))).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=None, help="comma-separated scene indices")
    ap.add_argument("--scenes_file", type=Path, default=None, help="json list of scene indices (avoids --export comma issues)")
    ap.add_argument("--compare_dir", type=Path, default=None, help="if set, save a full|naive|rec strip per scene")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--keep_n", type=int, default=20000)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--rec_views", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save_h5_dir", type=Path, default=None)
    ap.add_argument("--save_only", action="store_true",
                    help="Production: prune+recover+save recovered H5 only; skip the naive/held-out eval rendering.")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    h5dir = Path("data_v10") / ("full_h5s" if args.split == "train" else "full_h5s_val")

    rv_np, rk_np = make_orbit_views(args.rec_views, RADIUS, FOV, RES, up_axis="y")   # recovery views
    rv, rk = torch.from_numpy(rv_np).to(device), torch.from_numpy(rk_np).to(device)
    cv_np, ck_np = make_orbit_views(14, RADIUS, FOV, RES, up_axis="y")               # canonical (model) views
    cv, ck = torch.from_numpy(cv_np).to(device), torch.from_numpy(ck_np).to(device)
    hv, hk = heldout_views(16, device)                                              # strict held-out
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()

    if args.scenes_file:
        scene_ids = [int(x) for x in json.loads(args.scenes_file.read_text())]
    else:
        scene_ids = [int(x) for x in args.scenes.split(",")]
    if args.compare_dir:
        args.compare_dir.mkdir(parents=True, exist_ok=True)
        import imageio.v3 as iio

    results = []
    for s in scene_ids:
        h5 = h5dir / f"scene_{s:04d}.h5"
        if not h5.exists():
            print(f"skip scene_{s:04d}: missing"); continue
        full = load_full(h5)
        full_t = {k: v for k, v in to_dev(full, device).items()}
        full_t = dict(means=full_t["means"], quats=full_t["rotations"], scales=full_t["scales"],
                      opacities=full_t["opacities"], colors=full_t["colors"])
        naive = significance_topk(full, rv, rk, args.keep_n, device)
        # precompute recovery targets (full splat at recovery views)
        with torch.no_grad():
            target = torch.stack([rasterize(full_t, rv, rk, slice(v, v+1), RES)[0].clamp(0, 1)
                                  for v in range(args.rec_views)])
        rec = recover(naive, target, rv, rk, args.iters, device)

        if args.save_only:
            print(f"scene_{s:04d}: recovered (save-only)", flush=True)
        else:
            r = {"scene": s}
            for name, vm, K in [("canon", cv, ck), ("heldout", hv, hk)]:
                r[f"naive_{name}_psnr"], r[f"naive_{name}_lpips"] = eval_set(naive, full_t, vm, K, device, lpips_fn)
                r[f"rec_{name}_psnr"], r[f"rec_{name}_lpips"] = eval_set(rec, full_t, vm, K, device, lpips_fn)
            r["d_canon_psnr"] = r["rec_canon_psnr"] - r["naive_canon_psnr"]
            r["d_heldout_psnr"] = r["rec_heldout_psnr"] - r["naive_heldout_psnr"]
            results.append(r)
            print(f"scene_{s:04d}: canon naive {r['naive_canon_psnr']:.2f} -> rec {r['rec_canon_psnr']:.2f} "
                  f"(+{r['d_canon_psnr']:.2f}) | heldout naive {r['naive_heldout_psnr']:.2f} -> rec "
                  f"{r['rec_heldout_psnr']:.2f} (+{r['d_heldout_psnr']:.2f})", flush=True)
        if args.compare_dir:
            def _ren(a, vm, K, vi):
                p = to_dev(a, device); p = dict(means=p["means"], quats=p["rotations"], scales=p["scales"], opacities=p["opacities"], colors=p["colors"])
                with torch.no_grad():
                    return rasterize(p, vm, K, slice(vi, vi+1), RES)[0].clamp(0, 1).cpu().numpy()
            rows = []
            for vm, K, vi in [(cv, ck, 0), (cv, ck, 3), (hv, hk, 0)]:
                with torch.no_grad():
                    fimg = rasterize(full_t, vm, K, slice(vi, vi+1), RES)[0].clamp(0, 1).cpu().numpy()
                rows.append(np.concatenate([fimg, _ren(naive, vm, K, vi), _ren(rec, vm, K, vi)], axis=1))
            strip = (np.clip(np.concatenate(rows, axis=0), 0, 1) * 255).astype(np.uint8)
            iio.imwrite(args.compare_dir / f"scene_{s:04d}_full_naive_rec.png", strip)

        if args.save_h5_dir:
            args.save_h5_dir.mkdir(parents=True, exist_ok=True)
            with h5py.File(args.save_h5_dir / f"scene_{s:04d}.h5", "w") as f:
                for k in ("means", "scales", "colors"):
                    f.create_dataset(k, data=rec[k])
                f.create_dataset("rotations", data=rec["rotations"])
                f.create_dataset("opacities", data=rec["opacities"][:, None])
                with h5py.File(h5, "r") as src:
                    f.create_dataset("c2w", data=np.array(src["c2w"])); f.create_dataset("fov", data=np.array(src["fov"]))

        # Free GPU memory between objects -- otherwise the caching allocator creeps over
        # hundreds of objects and OOMs hard (exit 7, no traceback) ~object 400.
        del full, full_t, naive, target, rec
        torch.cuda.empty_cache()

    tag = (args.scenes_file.stem if args.scenes_file else args.scenes.replace(",", "_"))[:40]
    (args.out / f"recovery_{args.split}_{tag}.json").write_text(json.dumps(results, indent=1))
    if results:
        print(f"\nMEAN d_psnr: canon +{np.mean([r['d_canon_psnr'] for r in results]):.2f} | "
              f"heldout +{np.mean([r['d_heldout_psnr'] for r in results]):.2f}", flush=True)
    print("DONE_RECOVERY", flush=True)


if __name__ == "__main__":
    main()
