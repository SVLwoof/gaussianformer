import torch

from gaussianformer.models.gaussianformer import GaussianFormer
from gaussianformer.utils.ray_generator import RayGenerator
from gaussianformer.utils.transform import transform_gaussians_to_cam_coord


class GaussianFormerRenderingPipeline:
    def __init__(self, model: GaussianFormer):
        self.model = model
        self.config = model.config
        self.ray_generator = RayGenerator().to(model.device)

    @classmethod
    def from_pretrained(cls, model_id: str):
        model = GaussianFormer.from_pretrained(model_id)
        model.eval()
        return cls(model)

    @property
    def device(self):
        return self.model.device

    def to(self, device: torch.device):
        self.model.to(device)
        self.ray_generator.to(device)

    def render(
            self,
            gaussians,
            mask,
            c2w,
            fov,
            resolution: int = 512,
            torch_dtype: torch.dtype = torch.float16
    ):
        """
        Render images using the GaussianFormer model

        Args:
            gaussians: Gaussian data tensor [bs, num_gaussians, gaussian_dim]
            mask: Mask data tensor [bs, num_gaussians] - boolean mask indicating valid gaussians
            c2w: Camera-to-world matrix tensor [bs, num_views, 4, 4]
            fov: Field of view tensor [bs, num_views, 1] - in degrees
            resolution: Render resolution (default: 512)
            torch_dtype: PyTorch dtype for inference, default is torch.float16

        Returns:
            torch.Tensor: Rendered HDR image tensor [bs, num_views, H, W, 3]
        """

        bs, nv = c2w.shape[0], c2w.shape[1]

        # Handle view transformation
        if self.config.turn_to_cam_coord:
            # Reshape for transformation
            c2w_reshaped = c2w.reshape(-1, 4, 4)
            gaussians_repeated = torch.repeat_interleave(gaussians, nv, dim=0)

            gaussians_for_view_tf, c2w_for_view_tf = transform_gaussians_to_cam_coord(
                c2w_reshaped,
                gaussians_repeated
            )
            # Reshape back
            c2w_for_view_tf = c2w_for_view_tf.reshape(bs, nv, 4, 4)
            gaussians_for_view_tf = gaussians_for_view_tf.reshape(bs, nv, -1, self.config.gaussian_dim)
        else:
            # Expand gaussians for each view
            gaussians_for_view_tf = gaussians.unsqueeze(1).expand(-1, nv, -1, -1)
            c2w_for_view_tf = c2w

        # Generate rays
        # rays_o, rays_d: [bs, nv, H, W, 3]
        rays_o, rays_d = self.ray_generator(c2w_for_view_tf, fov / 180. * torch.pi, resolution)

        # Set precision
        assert torch_dtype in [torch.bfloat16, torch.float16, torch.float32], f"Invalid precision: {torch_dtype}\nChoose from: torch.bfloat16, torch.float16, torch.float32"
        tf32_view_tf = torch_dtype == torch.bfloat16 or torch_dtype == torch.float16

        # Perform rendering
        # Flatten triangles: [bs, num_tris, 3, 3] -> [bs, num_tris*9]
        # Flatten vn: [bs, num_tris, 3, 3] -> [bs, num_tris*9]
        # Flatten tri_vpos_view_tf: [bs, nv, num_tris, 3, 3] -> [bs, nv, num_tris*9]
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch_dtype):
            rendered_imgs = self.model(
                gaussians=gaussians,
                valid_mask=mask,
                rays_o=rays_o,
                rays_d=rays_d,
                # Pass only the positional part for view transformation
                gaussians_view_tf=gaussians_for_view_tf[..., :self.config.pos_dim],
                tf32_view_tf=tf32_view_tf,
            )

        # Process output
        # rendered_imgs: [bs, nv, C, H, W] -> [bs, nv, H, W, C]
        rendered_imgs = rendered_imgs.permute(0, 1, 3, 4, 2)

        # Log decode lighting if needed
        if not self.config.use_ldr:
            rendered_imgs = torch.pow(10., rendered_imgs) - 1.

        return rendered_imgs

    def __call__(self, *args, **kwargs):
        return self.render(*args, **kwargs)
