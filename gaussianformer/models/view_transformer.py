import torch
import torch.nn as nn
from einops import rearrange

from gaussianformer.layers.attention import TransformerDecoder
from gaussianformer.layers.dpt import DPTHead
from gaussianformer.layers.window import build_window
from gaussianformer.models.config import GaussianFormerConfig


class ViewTransformer(nn.Module):
    """View decoder: ray-direction patch tokens attend to the scene tokens; a DPT head decodes pixels."""

    def __init__(self, config: GaussianFormerConfig):
        super().__init__()
        self.config = config
        assert not config.ray_embed_patch or config.ray_embed_patch > config.patch_size, \
            "ray_embed_patch must exceed patch_size"
        dim = config.view_transformer_latent_dim
        embed_patch = config.ray_embed_patch or config.patch_size
        self.ray_map_patch_token = nn.Parameter(torch.randn(1, 1, dim))
        self.ray_map_encoder = nn.Linear(3 * embed_patch ** 2, dim)
        self.ray_map_encoder_norm = nn.RMSNorm(dim)
        self.transformer = TransformerDecoder(
            num_layers=config.view_transformer_n_layers,
            num_heads=config.view_transformer_n_heads,
            dim=dim,
            ctx_dim=config.latent_dim,
            ffn_hidden_dim=config.view_transformer_ffn_hidden_dim,
            rope_dim=config.pos_pe_num_freqs,
            proj_rope_2d=config.proj_rope_2d,
            ray_rope_2d_dim=config.ray_rope_2d_dim,
            ray_rope_2d_scale=config.ray_rope_2d_scale,
            grad_checkpoint=config.view_grad_checkpoint,
        )
        self.out_dpt = DPTHead(dim, features=config.dpt_features, out_channels=config.dpt_out_channels)
        self.out_layers = list(range(config.view_transformer_n_layers - 4, config.view_transformer_n_layers))
        self.out_proj_act = nn.ELU(alpha=1e-3)

    def project(self, pos, fov, H, W):
        """Camera-frame positions -> image position in patch units, depth, behind-camera flag, focal length.
        Camera at the origin looking down -Z (as in RayGenerator)."""
        focal = 0.5 * W / torch.tan(0.5 * fov.to(pos.dtype)).view(-1, 1)
        X, Y, Z = pos.unbind(-1)
        depth = (-Z).clamp_min(1e-3)
        u = (W / 2 + focal * X / depth) / self.config.patch_size
        v = (H / 2 - focal * Y / depth) / self.config.patch_size
        return torch.stack([u, v], -1), depth, Z > -1e-3, focal

    def forward(self, camera_o, ray_map, ctx_tokens, ctx_pos, ctx_scale, key_mask, fov, tf32_mode=False):
        """
        camera_o [B, 3]; ray_map [B, H, W, 3] ray directions; ctx_tokens [B, N, D] scene tokens;
        ctx_pos [B, N, 3] and ctx_scale [B, N, 3] camera-frame positions and scales (registers first);
        key_mask [B, N]; fov [B] in radians. Returns log-HDR images [B, 3, H, W].
        """
        cfg = self.config
        B, H, W, _ = ray_map.shape
        patch_h, patch_w = H // cfg.patch_size, W // cfg.patch_size
        embed_patch = cfg.ray_embed_patch or cfg.patch_size
        rays = ray_map
        if embed_patch != cfg.patch_size:  # nearest-upsample each patch to the embedding's patch size
            r = embed_patch // cfg.patch_size
            rays = rays.permute(0, 3, 1, 2).repeat_interleave(r, 2).repeat_interleave(r, 3).permute(0, 2, 3, 1)
        rays = rearrange(rays, "b (h1 p1) (w1 p2) c -> b (h1 w1) (c p1 p2)", p1=embed_patch, p2=embed_patch)
        ray_tokens = self.ray_map_patch_token + self.ray_map_encoder_norm(self.ray_map_encoder(rays))
        ray_pos = camera_o[:, None].repeat(1, ray_tokens.size(1), 1)

        uv_q = uv_k = window = None
        if cfg.proj_rope_2d or cfg.xattn_window:
            uv_k, depth, behind, focal = self.project(ctx_pos, fov, H, W)
            ii, jj = torch.meshgrid(torch.arange(patch_h, device=ray_map.device),
                                    torch.arange(patch_w, device=ray_map.device), indexing="ij")
            uv_q = (torch.stack([jj, ii], -1).reshape(1, -1, 2).float() + 0.5).expand(B, -1, -1)
        if cfg.xattn_window:
            radius = cfg.xattn_sigmas * ctx_scale.amax(-1) * focal / depth / cfg.patch_size
            key_ok = key_mask & ~behind
            key_ok[:, :cfg.num_register_tokens] = False
            window = build_window(uv_k, radius, key_ok, cfg.num_register_tokens, patch_h, patch_w,
                                  cfg.xattn_window, cfg.xattn_margin)
        if not cfg.proj_rope_2d:
            uv_q = uv_k = None

        fp32 = tf32_mode and not cfg.view_bf16
        with torch.autocast(device_type="cuda", dtype=torch.float32 if fp32 else torch.bfloat16):
            features = self.transformer(ray_tokens, ctx_tokens, key_mask, ctx_pos, ray_pos, self.out_layers,
                                        force_sdpa=fp32, uv_q=uv_q, uv_k=uv_k, window=window)
        return self.out_proj_act(self.out_dpt(features, patch_h, patch_w, patch_size=cfg.patch_size))
