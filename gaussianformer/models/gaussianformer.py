import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from gaussianformer.layers.attention import TransformerEncoder
from gaussianformer.models.config import GaussianFormerConfig
from gaussianformer.models.view_transformer import ViewTransformer

VIEW_CHUNK = 7  # views decoded per pass; the DPT upsampling of more 512 px views exceeds int32 indexing


class GaussianFormer(nn.Module, PyTorchModelHubMixin):
    """Scene encoder over one token per Gaussian [pos(3), scale(3), quat wxyz(4), rgb(3), opacity(1)],
    followed by the view decoder. Output: log10(HDR + 1) images."""

    def __init__(self, config: GaussianFormerConfig | dict):
        super().__init__()
        if isinstance(config, dict):
            config = GaussianFormerConfig(**config)
        self.config = config
        self.gaussian_encoder = nn.Linear(14, config.latent_dim)
        self.gaussian_encoder_norm = nn.RMSNorm(config.latent_dim)
        self.gaussian_token = nn.Parameter(torch.randn(1, 1, config.latent_dim))
        self.reg_tokens = nn.Parameter(torch.randn(1, config.num_register_tokens, config.latent_dim))
        self.transformer = TransformerEncoder(config.num_layers, config.num_heads, config.latent_dim,
                                              config.dim_feedforward, config.pos_pe_num_freqs)
        self.view_transformer = ViewTransformer(config)

    @property
    def device(self):
        return next(self.parameters()).device

    def _with_registers(self, pos, mask):
        """Prepend register tokens positioned at the centre of the valid Gaussians."""
        n = self.config.num_register_tokens
        w = (mask.float() / (mask.sum(dim=1, keepdim=True) + 1e-5))[..., None]
        center = (w * pos).sum(dim=1, keepdim=True)
        pos = torch.cat([center.repeat(1, n, 1), pos], dim=1)
        mask = torch.cat([torch.ones((mask.size(0), n), dtype=torch.bool, device=mask.device), mask], dim=1)
        return pos, mask

    def forward(self, gaussians, valid_mask, rays_o, rays_d, gaussians_cam, fov, tf32_view=False):
        """
        gaussians [B, N, 14] padded, valid_mask [B, N]; rays_o [B, V, 3], rays_d [B, V, H, W, 3];
        gaussians_cam [B, V, N, 14] the Gaussians in each view's camera frame; fov [B, V] in radians.
        Returns [B, V, 3, H, W].
        """
        B, V = rays_o.shape[:2]
        tokens = self.gaussian_token + self.gaussian_encoder_norm(self.gaussian_encoder(gaussians))
        seq = torch.cat([self.reg_tokens.expand(B, -1, -1), tokens], dim=1)
        pos, mask = self._with_registers(gaussians[..., :3], valid_mask)
        seq = self.transformer(seq, mask, pos)

        seq = seq.repeat_interleave(V, dim=0)
        gaussians_cam = gaussians_cam.reshape(B * V, *gaussians_cam.shape[2:])
        pos_cam, mask_cam = self._with_registers(gaussians_cam[..., :3], valid_mask.repeat_interleave(V, dim=0))
        scale = gaussians_cam[..., 3:6]
        scale_cam = torch.cat([scale.new_zeros(B * V, self.config.num_register_tokens, 3), scale], dim=1)
        rays_o, rays_d, fov = rays_o.reshape(B * V, 3), rays_d.reshape(B * V, *rays_d.shape[2:]), fov.reshape(-1)
        out = torch.cat([
            self.view_transformer(rays_o[i:i + VIEW_CHUNK], rays_d[i:i + VIEW_CHUNK], seq[i:i + VIEW_CHUNK],
                                  pos_cam[i:i + VIEW_CHUNK], scale_cam[i:i + VIEW_CHUNK], mask_cam[i:i + VIEW_CHUNK],
                                  fov[i:i + VIEW_CHUNK], tf32_mode=tf32_view)
            for i in range(0, B * V, VIEW_CHUNK)])
        return out.view(B, V, *out.shape[1:])
