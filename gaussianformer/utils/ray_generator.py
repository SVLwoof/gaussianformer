import torch
import torch.nn as nn
import torch.nn.functional as F


class RayGenerator(nn.Module):
    """Pinhole camera rays through pixel centres (Blender convention: -Z forward, +Y up)."""

    def forward(self, c2w: torch.Tensor, fov: torch.Tensor, img_res: int = 256):
        """c2w [..., 4, 4], fov [..., 1] in radians -> origins [..., 3], unit directions [..., H, W, 3]."""
        batch_shape = c2w.shape[:-2]
        x, y = torch.meshgrid(
            torch.linspace(0.5, img_res - 0.5, img_res, device=c2w.device, dtype=c2w.dtype),
            torch.linspace(0.5, img_res - 0.5, img_res, device=c2w.device, dtype=c2w.dtype),
            indexing="xy")
        c = img_res / 2
        f = img_res / 2 / torch.tan(0.5 * fov[..., 0, None, None])
        x = x[None].repeat(*batch_shape, 1, 1)
        y = y[None].repeat(*batch_shape, 1, 1)
        dirs = torch.stack([(x - c) / f, -(y - c) / f, -torch.ones_like(x)], dim=-1)
        rays_d = torch.sum(dirs[..., None, :] * c2w[..., None, None, :3, :3], dim=-1)
        return c2w[..., :3, 3], F.normalize(rays_d, dim=-1, p=2)
