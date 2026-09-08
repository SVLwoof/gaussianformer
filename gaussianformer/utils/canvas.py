"""P1 canvas conditioning: rasterize the input splat with gsplat for the same cameras the model
renders, so the view stage can start from exact projection + occlusion and learn to refine.

Output is in the model's log10(x+1) space. Detached: the canvas is an input like the rays.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def render_canvas(gaussians: torch.Tensor, mask: torch.Tensor, c2w: torch.Tensor, fov_deg: torch.Tensor,
                  resolution: int) -> torch.Tensor:
    """gaussians [B,N,14] (pos, scale, quat wxyz, rgb, opacity), mask [B,N], c2w [B,V,4,4] (Blender
    convention, same as the pipeline), fov_deg [B,V] -> canvas [B*V, H, W, 3] in log10(x+1)."""
    import gsplat  # optional dependency; only imported when canvas_cond is on

    B, V = c2w.shape[:2]
    dev = gaussians.device
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], device=dev, dtype=torch.float32))
    out = []
    for b in range(B):
        g = gaussians[b, mask[b]].float()
        p = dict(means=g[:, :3], scales=g[:, 3:6], quats=g[:, 6:10] / g[:, 6:10].norm(dim=-1, keepdim=True),
                 colors=g[:, 10:13], opacities=g[:, 13])
        viewmats = flip @ torch.linalg.inv(c2w[b].float())                       # [V,4,4] OpenCV w2c
        focal = 0.5 * resolution / torch.tan(0.5 * fov_deg[b].float() * torch.pi / 180)  # [V]
        Ks = torch.zeros(V, 3, 3, device=dev)
        Ks[:, 0, 0] = focal; Ks[:, 1, 1] = focal
        Ks[:, 0, 2] = resolution / 2; Ks[:, 1, 2] = resolution / 2; Ks[:, 2, 2] = 1.0
        img, _, _ = gsplat.rasterization(
            means=p["means"], quats=p["quats"], scales=p["scales"], opacities=p["opacities"],
            colors=p["colors"], viewmats=viewmats, Ks=Ks, width=resolution, height=resolution,
            sh_degree=None, eps2d=0.3, render_mode="RGB", near_plane=0.01, packed=True)
        out.append(torch.log10(img.clamp(0, 1) + 1))                              # [V,H,W,3]
    return torch.cat(out, 0)
