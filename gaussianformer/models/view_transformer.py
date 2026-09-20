import torch
import torch.nn as nn
import torch.nn.functional as F
import roma

from gaussianformer.models.config import GaussianFormerConfig
from gaussianformer.encodings.nerf_encoding import NeRFEncoding
from gaussianformer.layers.attention import TransformerDecoder
from gaussianformer.layers.window import build_window
from gaussianformer.layers.dpt import DPTHead

from einops import rearrange


class ViewTransformer(nn.Module):
    def __init__(self, config: GaussianFormerConfig):
        super().__init__()
        self.config = config

        # --- Positional Encoding Setup ---
        # The ray decoder is pure RoPE for both pe_types ('nerf' differs from 'rope' only
        # in the scene encoder; its ray decoder is identical).
        if config.pe_type in ('rope', 'nerf'):
            self.rope_dim = config.rope_dim or config.pos_pe_num_freqs
        else:
            raise ValueError(f"Unsupported positional encoding type: {config.pe_type}")

        # --- Ray Token Generation ---
        self.ray_map_patch_token = nn.Parameter(torch.randn(1, 1, config.view_transformer_latent_dim))
        if config.vdir_pe_type == 'nerf':
            self.vdir_pe = NeRFEncoding(
                in_dim=3,
                num_frequencies=config.vdir_num_freqs,
                include_input=True
            )
            self.ray_map_encoder = nn.Linear(
                self.vdir_pe.get_out_dim() * (config.ray_embed_patch or config.patch_size) ** 2,
                config.view_transformer_latent_dim
            )
            if config.norm_type == 'layer_norm':
                self.ray_map_encoder_norm = nn.LayerNorm(config.view_transformer_latent_dim)
            elif config.norm_type == 'rms_norm':
                self.ray_map_encoder_norm = nn.RMSNorm(config.view_transformer_latent_dim)
            else:
                raise ValueError(f"Unsupported normalization type: {config.norm_type}")
        else:
            raise ValueError(f"Unsupported view direction positional encoding type: {config.vdir_pe_type}")

        # --- Core View-Dependent Transformer ---
        self.transformer = TransformerDecoder(
            num_layers=self.config.view_transformer_n_layers,
            num_heads=self.config.view_transformer_n_heads,
            hidden_dim=self.config.view_transformer_latent_dim,
            ctx_dim=self.config.latent_dim,
            ffn_hidden_dim=self.config.view_transformer_ffn_hidden_dim,
            dropout=self.config.dropout,
            activation=self.config.activation,
            norm_type=self.config.norm_type,
            norm_first=self.config.norm_first,
            rope_dim=self.rope_dim,
            pos_dim=config.pos_dim,
            rope_double_max_freq=self.config.rope_double_max_freq,
            qk_norm=self.config.qk_norm,
            bias=self.config.bias,
            include_self_attn=self.config.view_transformer_include_self_attn,
            use_swin_attn=self.config.view_transformer_use_swin_attn,
            ray_rope_2d_dim=self.config.ray_rope_2d_dim,
            ray_rope_2d_scale=self.config.ray_rope_2d_scale,
            proj_rope_2d=self.config.proj_rope_2d,
            ray_self_rope_2d=self.config.ray_self_rope_2d,
            value_rope_2d=self.config.value_rope_2d,
            value_rope_2d_dim=self.config.value_rope_2d_dim,
            value_rope_2d_scale=self.config.value_rope_2d_scale,
            shape_rope_2d=self.config.shape_rope_2d,
            shape_rope_2d_dim=self.config.shape_rope_2d_dim,
            shape_rope_2d_scale=self.config.shape_rope_2d_scale,
        )
        assert not self.config.ray_embed_patch or self.config.ray_embed_patch > self.config.patch_size, \
            "ray_embed_patch must exceed patch_size (it is the pretrained embedding's patch)"
        if self.config.canvas_cond:
            # P1: rasterized canvas patches -> ray tokens, zero-init (exactly the baseline at init)
            self.canvas_encoder = nn.Linear(3 * self.config.patch_size ** 2, self.config.view_transformer_latent_dim)
            nn.init.zeros_(self.canvas_encoder.weight)
            nn.init.zeros_(self.canvas_encoder.bias)

        # --- Output Head ---
        if not config.use_dpt_decoder:
            self.out_proj = nn.Linear(self.config.view_transformer_latent_dim, self.config.patch_size * self.config.patch_size * (4 if config.include_alpha else 3))
        else:
            self.out_dpt = DPTHead(
                in_channels=self.config.view_transformer_latent_dim,
                features=self.config.dpt_features,
                out_channels=self.config.dpt_out_channels,
                out_dim=4 if config.include_alpha else 3
            )
            self.out_layers = list(range(self.config.view_transformer_n_layers - 4,
                                         self.config.view_transformer_n_layers)) if self.config.dpt_out_layers is None else self.config.dpt_out_layers
        self.out_proj_act = nn.ELU(alpha=1e-3)

    def project(self, spatial_pos, fov, H, W):
        """Camera-frame means -> (u, v) in PATCH units, positive depth, behind-camera flag.
        Matches RayGenerator: dirs = [(x-cx)/fx, -(y-cy)/fy, -1], camera at the origin looking down -Z."""
        ps = self.config.patch_size
        focal = 0.5 * W / torch.tan(0.5 * fov.to(spatial_pos.dtype)).view(-1, 1)  # [B, 1]
        X, Y, Z = spatial_pos.unbind(-1)
        depth = (-Z).clamp_min(1e-3)
        u = (W / 2 + focal * X / depth) / ps
        v = (H / 2 - focal * Y / depth) / ps
        return torch.stack([u, v], -1), depth, (Z > -1e-3), focal

    def axis_endpoints(self, spatial_pos, shape, fov, H, W):
        """Projected endpoints of each Gaussian's two largest principal axes: [B, N, 4] = (u1, v1, u2, v2)
        in patch units. `shape` = camera-frame [scale(3), quat(w,x,y,z)(4)]. Each axis' sign is
        fixed so its projected offset from the centre points to v > 0 (v == 0: u > 0)."""
        scale, quat = shape[..., :3], shape[..., 3:7]
        R = roma.unitquat_to_rotmat(quat[..., [1, 2, 3, 0]])  # [B, N, 3, 3], columns = principal directions
        top2 = scale.argsort(-1, descending=True)[..., :2]  # [B, N, 2]
        axes = torch.gather(R * scale[..., None, :], -1, top2[..., None, :].expand(-1, -1, 3, -1))  # [B, N, 3, 2]
        B, N = spatial_pos.shape[:2]
        plus = (spatial_pos[..., None] + axes).transpose(-1, -2).reshape(B, N * 2, 3)
        minus = (spatial_pos[..., None] - axes).transpose(-1, -2).reshape(B, N * 2, 3)
        uv_p, _, _, _ = self.project(plus, fov, H, W)
        uv_m, _, _, _ = self.project(minus, fov, H, W)
        uv_c = self.project(spatial_pos, fov, H, W)[0].repeat_interleave(2, 1)
        d = uv_p - uv_c
        flip = (d[..., 1] < 0) | ((d[..., 1] == 0) & (d[..., 0] < 0))
        return torch.where(flip[..., None], uv_m, uv_p).reshape(B, N, 4)

    def forward(self, camera_o, ray_map, ctx_tokens, spatial_pos, valid_mask, tf32_mode=False,
                fov=None, canvas=None, shape=None):
        """
        Cross attention between ray map and context tokens (gaussians).

        Args:
            camera_o (torch.Tensor): (B, 3) camera origin
            ray_map (torch.Tensor): (B, H, W, 3) ray directions
            ctx_tokens (torch.Tensor): (B, N_ITEMS, D) context tokens from the encoder
            spatial_pos (torch.Tensor): (B, N_ITEMS, pos_dim) spatial positions of context items
            valid_mask (torch.Tensor): (B, N_ITEMS) boolean mask for valid context items
            tf32_mode (bool): whether to use tf32 mode
        Returns:
            decoded_img: (B, 3, H, W)
        """

        # --- Prepare Query Sequence (Ray Tokens) ---
        ray_map_pe = self.vdir_pe(ray_map)
        pe = self.config.ray_embed_patch or self.config.patch_size
        if pe != self.config.patch_size:
            # finer grid on the pretrained embedding: each patch_size x patch_size ray patch is
            # upsampled (nearest) to pe x pe, so the Linear sees its native patch layout
            r = pe // self.config.patch_size
            ray_map_pe = ray_map_pe.permute(0, 3, 1, 2).repeat_interleave(r, 2).repeat_interleave(r, 3).permute(0, 2, 3, 1)
        ray_tokens = rearrange(ray_map_pe, 'b (h1 p1) (w1 p2) c -> b (h1 w1) (c p1 p2)', p1=pe, p2=pe)
        patch_h = ray_map.size(1) // self.config.patch_size
        patch_w = ray_map.size(2) // self.config.patch_size
        ray_tokens = self.ray_map_patch_token + self.ray_map_encoder_norm(self.ray_map_encoder(ray_tokens))  # [B, N_PATCHES, D]
        if self.config.canvas_cond:
            assert canvas is not None, "canvas_cond needs the rasterized canvas [B, H, W, 3]"
            canvas_tokens = rearrange(canvas.to(ray_tokens.dtype), 'b (h1 p1) (w1 p2) c -> b (h1 w1) (c p1 p2)',
                                      p1=self.config.patch_size, p2=self.config.patch_size)
            ray_tokens = ray_tokens + self.canvas_encoder(canvas_tokens)
        n_patches = ray_tokens.size(1)
        ray_pos = camera_o[:, None].repeat(1, n_patches, 1)  # [B, N_PATCHES, 3]

        # --- P2: explicit perspective projection of every Gaussian (what rasterization uses) ---
        uv_q = uv_k = None
        if self.config.proj_rope_2d or self.config.value_rope_2d:
            assert fov is not None, "projected 2-D RoPE needs fov (radians) per view"
            uv_k, _, _, _ = self.project(spatial_pos, fov, ray_map.size(1), ray_map.size(2))
            ii, jj = torch.meshgrid(torch.arange(patch_h, device=ray_map.device),
                                    torch.arange(patch_w, device=ray_map.device), indexing="ij")
            uv_q = (torch.stack([jj, ii], -1).reshape(1, -1, 2).float() + 0.5).expand(ray_map.size(0), -1, -1)
        window = None
        if self.config.xattn_window:
            assert fov is not None and shape is not None, "windowed cross-attention needs fov and the camera-frame scales"
            uv_k_all, depth, behind, focal = self.project(spatial_pos, fov, ray_map.size(1), ray_map.size(2))
            n_reg = self.config.num_register_tokens
            radius = self.config.xattn_sigmas * shape[..., :3].amax(-1) * focal / depth / self.config.patch_size
            key_ok = valid_mask & ~behind
            key_ok[:, :n_reg] = False
            window = build_window(uv_k_all, radius, key_ok, n_reg, patch_h, patch_w,
                                  self.config.xattn_window, self.config.xattn_margin)
        ends_q = ends_k = None
        if self.config.shape_rope_2d:
            assert shape is not None, "shape_rope_2d needs the camera-frame scale + rotation per Gaussian"
            ends_k = self.axis_endpoints(spatial_pos, shape, fov, ray_map.size(1), ray_map.size(2))
            ends_q = uv_q.repeat(1, 1, 2)

        # --- Decode with Transformer ---
        # The TransformerDecoder internally handles RoPE based on `ray_pos` and `spatial_pos`.
        if self.config.use_dpt_decoder:
            with torch.autocast(device_type="cuda", dtype=torch.float32 if tf32_mode else torch.bfloat16):
                out_features = self.transformer(
                    ray_tokens,
                    ctx_tokens,
                    src_key_padding_mask=valid_mask,
                    spatial_pos=spatial_pos,  # For context RoPE
                    ray_pos=ray_pos,  # For query RoPE
                    out_layers=self.out_layers,
                    tf32_mode=tf32_mode,
                    patch_h=patch_h,
                    patch_w=patch_w,
                    uv_q=uv_q, uv_k=uv_k, ends_q=ends_q, ends_k=ends_k, window=window,
                )
            decoded_img = self.out_dpt(out_features, patch_h, patch_w, patch_size=self.config.patch_size)
            return self.out_proj_act(decoded_img)
        else:
            seq = self.transformer(
                ray_tokens,
                ctx_tokens,
                src_key_padding_mask=valid_mask,
                spatial_pos=spatial_pos,  # For context RoPE
                ray_pos=ray_pos,  # For query RoPE
                tf32_mode=tf32_mode,
                patch_h=patch_h,
                patch_w=patch_w,
                uv_q=uv_q, uv_k=uv_k, ends_q=ends_q, ends_k=ends_k, window=window,
            )  # [B, N_PATCHES, D]
            decoded_patches = self.out_proj_act(self.out_proj(seq))  # [B, N_PATCHES, P*P*3]
            decoded_img = rearrange(
                decoded_patches,
                'b (h1 w1) (c p1 p2) -> b c (h1 p1) (w1 p2)',
                p1=self.config.patch_size,
                p2=self.config.patch_size,
                h1=patch_h,
                w1=patch_w,
            )

            return decoded_img
