import torch
import torch.nn as nn
import torch.nn.functional as F

from gaussianformer.models.config import GaussianFormerConfig
from gaussianformer.encodings.nerf_encoding import NeRFEncoding
from gaussianformer.layers.attention import TransformerDecoder
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
            self.rope_dim = config.pos_pe_num_freqs
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
                self.vdir_pe.get_out_dim() * config.patch_size * config.patch_size,
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
            geom_bias=self.config.geom_bias,
        )

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

    def forward(self, camera_o, ray_map, ctx_tokens, spatial_pos, valid_mask, tf32_mode=False):
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
        ray_tokens = rearrange(
            ray_map_pe,
            'b (h1 p1) (w1 p2) c -> b (h1 w1) (c p1 p2)',
            p1=self.config.patch_size,
            p2=self.config.patch_size,
        )
        patch_h = ray_map.size(1) // self.config.patch_size
        patch_w = ray_map.size(2) // self.config.patch_size
        ray_tokens = self.ray_map_patch_token + self.ray_map_encoder_norm(self.ray_map_encoder(ray_tokens))  # [B, N_PATCHES, D]
        n_patches = ray_tokens.size(1)
        ray_pos = camera_o[:, None].repeat(1, n_patches, 1)  # [B, N_PATCHES, 3]

        # --- Geometric cross-attention bias ---
        # ray_pos above is the SAME camera origin for every patch, so RoPE contributes no
        # per-patch geometry to the cross-attention logits. This alignment map does:
        # cos(angle) between each patch's mean ray direction and the direction from the
        # camera to each context item. Register tokens (first num_register_tokens slots
        # of spatial_pos) sit at the scene center; their bias is zeroed.
        geom_align = None
        if self.config.geom_bias:
            ps = self.config.patch_size
            patch_dirs = ray_map.view(ray_map.size(0), patch_h, ps, patch_w, ps, 3).mean(dim=(2, 4))
            patch_dirs = F.normalize(patch_dirs, dim=-1).view(ray_map.size(0), -1, 3)
            ctx_dirs = F.normalize(spatial_pos - camera_o[:, None], dim=-1)  # [B, N_CTX, 3]
            geom_align = patch_dirs @ ctx_dirs.transpose(1, 2)  # [B, N_PATCHES, N_CTX]
            geom_align[:, :, :self.config.num_register_tokens] = 0.0

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
                    geom_align=geom_align
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
                geom_align=geom_align
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
