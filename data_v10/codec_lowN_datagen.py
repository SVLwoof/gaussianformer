"""Low-N codec inputs: re-prune an object's FULL splat to N Gaussians with the fleet recipe
(significance top-K on a 64-view orbit + 1500-iter recovery) and lay it out as a sibling
codec_scaleout object `<scene>_n<N/1000>k` so train_codec_lora.sh / codec_scaleout_eval.sh
work unchanged. Supervision (train renders, eval GT, eval poses) is the same full-splat data
as the 20k object, linked per file (renders are keyed by the h5 stem).

  python data_v10/codec_lowN_datagen.py --scene gopro --n 5000
"""
from __future__ import annotations
import argparse, os
import numpy as np, torch, h5py, imageio.v3 as iio
from pathlib import Path
from data_external.orbit import make_orbit_views, c2w_to_viewmat
from data_v10.prune_recovery import significance_topk, recover, rasterize, to_dev

RES, FOV, RADIUS = 512, 45.0, 1.7
device = "cuda"
KEYS = ("means", "scales", "rotations", "colors", "opacities")
CS = Path("experiments/overfit/data/codec_scaleout")


def load_full(scene: str) -> dict[str, np.ndarray]:
    if scene.startswith("scene_"):
        with h5py.File(f"data_v10/full_h5s_val/{scene}.h5") as f:
            full = {k: np.array(f[k], np.float32) for k in KEYS}
    elif scene == "tomatoes":
        # already normalised/oriented for the tomato codec lineage; read raw fields, no recentre
        from plyfile import PlyData
        v = PlyData.read("data_external/tomatoes/normalized.ply")["vertex"]
        C0 = 0.28209479177387814
        q = np.stack([v[f"rot_{i}"] for i in range(4)], -1).astype(np.float32)
        full = dict(
            means=np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float32),
            rotations=q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-9),
            scales=np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], -1).astype(np.float32)),
            colors=np.clip(0.5 + C0 * np.stack([v[f"f_dc_{i}"] for i in range(3)], -1), 0, 1).astype(np.float32),
            opacities=1 / (1 + np.exp(-np.asarray(v["opacity"], np.float32))),
        )
    else:
        from data_external.prep_external_codec import load_normalized
        full = load_normalized(scene, flip_x=bool(int(os.environ.get("FLIP_X", "1"))))
    full["opacities"] = full["opacities"].reshape(-1)
    return full


def source_layout(scene: str) -> tuple[Path, Path, str, Path]:
    """(train h5 with c2w/fov, train renders dir, render stem, eval dir with npys + renders/codec_eval_*)."""
    if scene == "tomatoes":
        r = Path("experiments/overfit/data/tomatoes_codec5")
        return r / "h5s/tomatoes_codec5.h5", r / "renders", "tomatoes_codec5", Path("data_external/tomatoes")
    r = CS / scene
    return r / "h5s" / f"{scene}.h5", r / "renders", scene, r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--n", type=int, required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    tag = f"{a.scene}_n{a.n // 1000}k"
    root = CS / tag
    src_h5, src_renders, stem, src_eval = source_layout(a.scene)
    if (root / "h5s" / f"{tag}.h5").exists():
        print(f"{tag}: exists, skipping"); return

    full = load_full(a.scene)
    full_t = to_dev(dict(means=full["means"], scales=full["scales"], quats=full["rotations"],
                         colors=full["colors"], opacities=full["opacities"]), device)
    vm64, K64 = make_orbit_views(64, RADIUS, FOV, RES, up_axis="y")
    vm64, K64 = torch.from_numpy(vm64).to(device), torch.from_numpy(K64).to(device)
    targets = torch.cat([rasterize(full_t, vm64, K64, list(range(i, min(i + 8, 64))), RES)
                         for i in range(0, 64, 8)])
    print(f"{a.scene}: full N={len(full['means'])} -> prune to {a.n}", flush=True)
    pruned = significance_topk(full, vm64, K64, a.n, device)
    pruned = {k: (v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)) for k, v in pruned.items()}
    pruned["opacities"] = pruned["opacities"].reshape(-1)
    rec = recover(pruned, targets, vm64, K64, 1500, device)
    rec = {k: (v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)) for k, v in rec.items()}

    # --- orientation / frame sanity: recovered splat vs the object's GT eval render ---
    c2w0 = np.load(src_eval / "codec_eval_novel_rand_c2w.npy")[0]
    focal = 0.5 * RES / np.tan(0.5 * np.radians(FOV))
    K = torch.as_tensor(np.array([[focal, 0, RES / 2], [0, focal, RES / 2], [0, 0, 1]], np.float32)[None], device=device)
    vm = torch.as_tensor(c2w_to_viewmat(c2w0)[None], device=device)
    rec_t = to_dev(dict(means=rec["means"], scales=rec["scales"], quats=rec["rotations"],
                        colors=rec["colors"], opacities=np.asarray(rec["opacities"]).reshape(-1)), device)
    img = rasterize(rec_t, vm, K, [0], RES)[0].clamp(0, 1).cpu().numpy()
    gt = iio.imread(src_eval / "renders/codec_eval_novel_rand/gt_view_0.png").astype(np.float32) / 255
    psnr = 10 * np.log10(1 / max(((img - gt) ** 2).mean(), 1e-12))
    print(f"{tag}: rec-GT vs GT on eval view 0 = {psnr:.2f} dB", flush=True)
    assert psnr > 15, f"orientation/frame mismatch suspected ({psnr:.1f} dB)"

    # --- layout ---
    (root / "h5s").mkdir(parents=True, exist_ok=True)
    with h5py.File(src_h5) as f:
        c2w, fov = np.array(f["c2w"]), np.array(f["fov"])
    with h5py.File(root / "h5s" / f"{tag}.h5", "w") as f:
        for k in ("means", "scales", "rotations", "colors"):
            f.create_dataset(k, data=rec[k].astype(np.float32))
        f.create_dataset("opacities", data=np.asarray(rec["opacities"], np.float32).reshape(-1, 1))
        f.create_dataset("c2w", data=c2w); f.create_dataset("fov", data=fov)
    rd = root / "renders"; rd.mkdir(exist_ok=True)
    n_link = 0
    for j in range(len(c2w)):
        dst = rd / f"{tag}_view_{j}.png"
        if not dst.exists():
            os.symlink(os.path.relpath(src_renders / f"{stem}_view_{j}.png", rd), dst); n_link += 1
    for name in ("novel_rand", "novel_close", "novel_far"):
        np.save(root / f"codec_eval_{name}_c2w.npy", np.load(src_eval / f"codec_eval_{name}_c2w.npy"))
        dst = rd / f"codec_eval_{name}"
        if not dst.exists():
            os.symlink(os.path.relpath(src_eval / "renders" / f"codec_eval_{name}", rd), dst)
    iio.imwrite(root / "recgt_view0.png", (np.concatenate([img, gt], 1) * 255).astype(np.uint8))
    print(f"{tag}: h5 {os.path.getsize(root / 'h5s' / f'{tag}.h5') / 1e3:.0f} KB, {n_link} render links -> {root}", flush=True)


if __name__ == "__main__":
    main()
