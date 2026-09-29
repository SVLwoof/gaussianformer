import torch

from gaussianformer.models.gaussianformer import GaussianFormer
from gaussianformer.utils.ray_generator import RayGenerator
from gaussianformer.utils.transform import transform_gaussians_to_cam_coord


def model_forward(model: GaussianFormer, gaussians, mask, c2w, fov, resolution: int, tf32_view: bool = False):
    """Render log-HDR images [B, V, H, W, 3]; gaussians [B, N, 14], mask [B, N], c2w [B, V, 4, 4],
    fov [B, V] in degrees. The decoder works in each camera's frame, so its rays start at the origin."""
    B, V = c2w.shape[:2]
    with torch.no_grad():
        gaussians_cam = transform_gaussians_to_cam_coord(
            c2w.reshape(-1, 4, 4), gaussians.repeat_interleave(V, dim=0)).reshape(B, V, *gaussians.shape[1:])
        eye = torch.eye(4, device=c2w.device, dtype=c2w.dtype).expand(B, V, 4, 4)
        fov = fov / 180.0 * torch.pi
        rays_o, rays_d = RayGenerator()(eye, fov[..., None], resolution)
    out = model(gaussians, mask, rays_o, rays_d, gaussians_cam, fov, tf32_view=tf32_view)
    return out.permute(0, 1, 3, 4, 2)


class GaussianFormerRenderingPipeline:
    def __init__(self, model: GaussianFormer):
        self.model = model

    @classmethod
    def from_pretrained(cls, model_id: str):
        return cls(GaussianFormer.from_pretrained(model_id).eval())

    @property
    def device(self):
        return self.model.device

    def to(self, device: torch.device):
        self.model.to(device)
        return self

    @torch.no_grad()
    def __call__(self, gaussians, mask, c2w, fov, resolution: int = 512, torch_dtype: torch.dtype = torch.bfloat16):
        """
        gaussians [B, N, 14] = pos(3) | scale(3) | quat wxyz(4) | rgb(3) | opacity(1); mask [B, N] bool;
        c2w [B, V, 4, 4] camera-to-world (Blender convention); fov [B, V] in degrees.
        Returns images [B, V, H, W, 3], display-referred (clip to [0, 1] for 8-bit output).
        """
        with torch.autocast(device_type=self.device.type, dtype=torch_dtype):
            log_img = model_forward(self.model, gaussians, mask, c2w, fov.reshape(c2w.shape[:2]), resolution,
                                    tf32_view=torch_dtype != torch.float32)
        return torch.pow(10.0, log_img) - 1.0
