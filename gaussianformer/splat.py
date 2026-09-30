"""3D Gaussian Splatting PLY files -> model inputs.

A 3DGS PLY stores the optimiser's raw parameters; GaussianFormer takes activated values in a normalised object
frame, [N, 14] = pos(3) | scale(3) | quat wxyz(4) | rgb(3) | opacity(1):

    scale    exp(scale_0..2)
    quat     rot_0..3 (w, x, y, z), normalised
    rgb      clip(0.5 + SH_C0 * f_dc_0..2, 0, 1)   (higher spherical-harmonic orders are dropped)
    opacity  sigmoid(opacity)

The object is centred on its median position, turned Y-up and scaled so its largest coordinate is 0.45, and
pruned to about 20k Gaussians (the size the model is trained on) by LightGaussian importance, followed by a
short fine-tune of the kept Gaussians against the full splat.

  from gaussianformer.splat import load_ply
  gaussians = load_ply("object.ply", up="z")   # [N, 14] tensor on the GPU
"""
import io
from pathlib import Path

import numpy as np
import roma
import torch
import torch.nn.functional as F
from plyfile import PlyData

from gaussianformer.utils.cameras import RESOLUTION, orbit, to_gsplat

SH_C0 = 0.28209479177387814
TO_Y_UP = {
    "y": np.eye(3, dtype=np.float32),
    "z": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32),   # Z-up (e.g. Objaverse_Splats)
    "-y": np.diag([1.0, -1.0, -1.0]).astype(np.float32),                    # Y-down (e.g. COLMAP captures)
}
KEEP = 20_000
RECOVERY_VIEWS = 64
RECOVERY_ITERS = 1500
FIELDS = ("means", "scales", "rotations", "colors", "opacities")


def read_ply(src: str | Path | bytes) -> dict[str, np.ndarray]:
    """3DGS PLY (path or bytes) -> activated Gaussians: means, scales, rotations (wxyz), colors, opacities."""
    v = PlyData.read(io.BytesIO(src) if isinstance(src, bytes) else str(src))["vertex"]
    quats = np.stack([v[f"rot_{i}"] for i in range(4)], -1).astype(np.float32)
    return dict(
        means=np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float32),
        scales=np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], -1)).astype(np.float32),
        rotations=quats / (np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-9),
        colors=np.clip(0.5 + SH_C0 * np.stack([v[f"f_dc_{i}"] for i in range(3)], -1), 0, 1).astype(np.float32),
        opacities=(1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], np.float32)))).astype(np.float32),
    )


def normalize(g: dict[str, np.ndarray], up: str = "y") -> dict[str, np.ndarray] | None:
    """Centre on the median position, rotate `up` to +Y and scale into [-0.45, 0.45]^3; None if degenerate."""
    R = TO_Y_UP[up]
    means = ((g["means"] - np.median(g["means"], axis=0)) @ R.T).astype(np.float32)
    rot = roma.unitquat_to_rotmat(torch.from_numpy(g["rotations"])[:, [1, 2, 3, 0]])
    rotations = roma.rotmat_to_unitquat(torch.from_numpy(R) @ rot)[:, [3, 0, 1, 2]].numpy()
    extent = float(np.abs(means).max())
    if extent < 1e-4:
        return None
    s = 0.45 / extent
    return dict(means=means * s, scales=g["scales"] * s, rotations=rotations, colors=g["colors"],
                opacities=g["opacities"])


def rasterize(g: dict[str, torch.Tensor], viewmats: torch.Tensor, Ks: torch.Tensor):
    """gsplat render of Gaussians `g` (activated values, opacities [N]) -> images [V, H, W, 3], render info."""
    import gsplat

    return gsplat.rasterization(
        means=g["means"], quats=g["rotations"] / g["rotations"].norm(dim=-1, keepdim=True), scales=g["scales"],
        opacities=g["opacities"], colors=g["colors"], viewmats=viewmats, Ks=Ks, width=RESOLUTION,
        height=RESOLUTION, sh_degree=None, eps2d=0.3, render_mode="RGB", near_plane=0.01, packed=True)[::2]


def importance_topk(g: dict[str, torch.Tensor], viewmats, Ks, keep: int) -> torch.Tensor:
    """Indices of the `keep` Gaussians with the highest LightGaussian importance: opacity x projected area
    summed over the views, times the largest scale^0.1."""
    score = torch.zeros(len(g["means"]), device=g["means"].device, dtype=torch.float64)
    for v in range(len(viewmats)):
        with torch.no_grad():
            _, info = rasterize(g, viewmats[v:v + 1], Ks[v:v + 1])
        area = info["radii"][:, 0].double() * info["radii"][:, 1].double()
        score.scatter_add_(0, info["gaussian_ids"], info["opacities"].double() * area)
    score = score * g["scales"].max(1).values.double() ** 0.1
    return torch.topk(score, min(keep, len(score))).indices


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    g = torch.exp(-(torch.arange(11, dtype=torch.float32, device=x.device) - 5) ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :]).expand(3, 1, 11, 11).contiguous()
    conv = lambda t: F.conv2d(t, w, padding=5, groups=3)
    mx, my = conv(x), conv(y)
    sx, sy, sxy = conv(x * x) - mx * mx, conv(y * y) - my * my, conv(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sx + sy + c2))).mean()


def recover(g: dict[str, torch.Tensor], target: torch.Tensor, viewmats, Ks) -> dict[str, torch.Tensor]:
    """Fine-tune pruned Gaussians (L1 + SSIM, no densification) against the full splat's renders `target`."""
    p = {"means": g["means"], "log_scales": g["scales"].clamp_min(1e-8).log(), "rotations": g["rotations"],
         "opacity_logits": torch.logit(g["opacities"].clamp(1e-4, 1 - 1e-4)),
         "color_logits": torch.logit(g["colors"].clamp(1e-4, 1 - 1e-4))}
    p = {k: torch.nn.Parameter(v.clone()) for k, v in p.items()}
    lrs = {"means": 1.6e-4, "log_scales": 5e-3, "rotations": 1e-3, "opacity_logits": 5e-2, "color_logits": 2.5e-3}
    opt = torch.optim.Adam([{"params": [p[k]], "lr": lr} for k, lr in lrs.items()])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95 ** (1 / 400))
    current = lambda: dict(means=p["means"], scales=p["log_scales"].exp(), rotations=p["rotations"],
                           opacities=p["opacity_logits"].sigmoid(), colors=p["color_logits"].sigmoid())
    for _ in range(RECOVERY_ITERS):
        v = torch.randint(0, len(viewmats), (4,), device=target.device)
        img = rasterize(current(), viewmats[v], Ks[v])[0].clamp(0, 1)
        loss = 0.8 * (img - target[v]).abs().mean() + 0.2 * (1 - ssim(img.permute(0, 3, 1, 2), target[v].permute(0, 3, 1, 2)))
        loss.backward()
        opt.step()
        opt.zero_grad()
        sched.step()
    with torch.no_grad():
        out = current()
        out["rotations"] = out["rotations"] / out["rotations"].norm(dim=-1, keepdim=True)
        return {k: v.detach() for k, v in out.items()}


def prune(full: dict[str, torch.Tensor], keep: int = KEEP, recovery: bool = True) -> dict[str, torch.Tensor]:
    """Keep the `keep` most important Gaussians (scored from a 64-view orbit), then optionally fine-tune them."""
    views = tuple(torch.from_numpy(x).to(full["means"].device) for x in to_gsplat(orbit(RECOVERY_VIEWS, 1.7)))
    with torch.no_grad():
        target = torch.cat([rasterize(full, views[0][i:i + 1], views[1][i:i + 1])[0].clamp(0, 1)
                            for i in range(RECOVERY_VIEWS)]) if recovery else None
    kept = {k: v[importance_topk(full, *views, keep)] for k, v in full.items()}
    return recover(kept, target, *views) if recovery else kept


def to_tensor(g: dict[str, torch.Tensor | np.ndarray]) -> torch.Tensor:
    """Gaussians as a dict -> the model's [N, 14] layout."""
    parts = [torch.as_tensor(g[k]).float().reshape(len(g["means"]), -1) for k in FIELDS]
    return torch.cat(parts, dim=-1)


def load_ply(path: str | Path, up: str = "y", keep: int = KEEP, recovery: bool = True,
             device: str | torch.device = "cuda") -> torch.Tensor:
    """3DGS PLY -> model input [N, 14] on `device`: activated, normalised and pruned to at most `keep` Gaussians.
    `up` is the file's up axis ('y', 'z' or '-y'). `recovery` fine-tunes the kept Gaussians against the full
    splat (about a minute on a GPU); without it a large splat pruned to 20k loses coverage and turns dark."""
    g = normalize(read_ply(path), up)
    assert g is not None, f"{path}: all Gaussians at one point"
    g = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in g.items()}
    if len(g["means"]) > keep:
        g = prune(g, keep, recovery)
    return to_tensor(g)
