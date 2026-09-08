import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from gaussianformer.encodings.nerf_encoding import NeRFEncoding
from gaussianformer.layers.attention import TransformerEncoder
from gaussianformer.models.config import GaussianFormerConfig
from gaussianformer.models.view_transformer import ViewTransformer


class GaussianFormer(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: GaussianFormerConfig):
        super(GaussianFormer, self).__init__()
        self.config = config

        # --- Input Encoders ---
        norm_class = nn.LayerNorm if self.config.gaussian_encoder_norm_type == 'layer_norm' else nn.RMSNorm
        self.gaussian_token = nn.Parameter(torch.randn(1, 1, self.config.latent_dim))
        self.rope_dim = None

        # RoPE stays ON for both pe_types: the RenderFormer backbone is RoPE-pretrained.
        self.gaussian_encoder_norm = norm_class(self.config.latent_dim)
        if self.config.input_mlp_hidden:
            # Residual input-head MLP, zero-init output -> exactly the baseline at init.
            self.gaussian_encoder_mlp = nn.Sequential(
                nn.Linear(self.config.latent_dim, self.config.input_mlp_hidden),
                nn.GELU(),
                nn.Linear(self.config.input_mlp_hidden, self.config.latent_dim),
            )
            nn.init.zeros_(self.gaussian_encoder_mlp[-1].weight)
            nn.init.zeros_(self.gaussian_encoder_mlp[-1].bias)
        self.rope_dim = self.config.rope_dim or self.config.pos_pe_num_freqs

        if self.config.pe_type == 'nerf':
            # Concat encoder: position lifted into a NeRF basis, concatenated with the
            # remaining raw fields (scale as log-scale), projected by a single Linear --
            # one free weighting over all fields, like rope's Linear(14,768) but with
            # NeRF position.
            self.gaussian_pos_pe = NeRFEncoding(in_dim=3, num_frequencies=self.config.pos_pe_num_freqs,
                                                include_input=True)
            encoder_in_dim = self.gaussian_pos_pe.get_out_dim() + 11  # scale(3)+quat(4)+color(3)+opacity(1)
            self.gaussian_encoder = nn.Linear(encoder_in_dim, self.config.latent_dim)
        else:  # rope
            self.gaussian_encoder = nn.Linear(self.config.gaussian_dim, self.config.latent_dim)

        # --- Common Components ---
        self.reg_tokens = nn.Parameter(torch.randn(1, self.config.num_register_tokens, self.config.latent_dim))
        self.skip_token_num = self.config.num_register_tokens

        self.transformer = TransformerEncoder(
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            hidden_dim=self.config.latent_dim,
            ffn_hidden_dim=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation=self.config.activation,
            norm_type=self.config.norm_type,
            norm_first=self.config.norm_first,
            rope_dim=self.rope_dim,
            pos_dim=self.config.pos_dim,
            bias=self.config.bias,
            qk_norm=self.config.view_indep_qk_norm,
            rope_double_max_freq=self.config.rope_double_max_freq,
            rope_pos_scale=self.config.rope_pos_scale,
            rope_hf_scale=self.config.rope_hf_scale,
        )

        self.view_transformer = ViewTransformer(config)

    @property
    def device(self):
        return next(self.parameters()).device

    def _prepare_padded_positions(self, pos_list, valid_mask):
        """Prepares positions for RoPE by prepending positions for register tokens."""
        # Calculate weighted center of valid items
        mask_weight = (valid_mask.float() / (valid_mask.sum(dim=1, keepdim=True) + 1e-5))[..., None]
        weighted_pos = mask_weight * pos_list
        center_pos = weighted_pos.sum(dim=1, keepdim=True)

        # Repeat the center position for each register token
        center_pos_padded = center_pos.repeat(1, self.skip_token_num, 1)

        # Concatenate with item positions
        pos_list_padded = torch.cat([center_pos_padded, pos_list], dim=1)

        # Create a corresponding padded mask
        valid_mask_padded = torch.cat([
            torch.ones((pos_list.size(0), self.skip_token_num), dtype=torch.bool, device=valid_mask.device),
            valid_mask
        ], dim=1)

        return pos_list_padded, valid_mask_padded

    def construct_sequence(self, gaussians, valid_mask):
        """Constructs sequence from Gaussian data."""
        batch_size = gaussians.size(0)

        tokens = [self.reg_tokens.expand(batch_size, -1, -1)]

        if self.config.pe_type == 'nerf':
            # Concat encoder: NeRF-lifted position + log-scale + raw quaternion /
            # color / opacity, projected by a single Linear. The shared projection's
            # bias absorbs per-field offset, and its weights absorb per-field
            # magnitude -- no per-field normalization.
            pos = gaussians[..., 0:3]
            scale = gaussians[..., 3:6]
            rest = gaussians[..., 6:14]  # quat(4) + color(3) + opacity(1)
            log_scale = torch.log(scale.clamp(min=1e-6))
            feat = torch.cat([self.gaussian_pos_pe(pos), log_scale, rest], dim=-1)
            e = self.gaussian_encoder(feat)
            if self.config.input_mlp_hidden:
                e = e + self.gaussian_encoder_mlp(e)
            gaussian_emb = self.gaussian_encoder_norm(e)
            tokens.append(self.gaussian_token + gaussian_emb)
        else:  # rope
            e = self.gaussian_encoder(gaussians)
            if self.config.input_mlp_hidden:
                e = e + self.gaussian_encoder_mlp(e)
            gaussian_emb = self.gaussian_encoder_norm(e)
            tokens.append(self.gaussian_token + gaussian_emb)

        seq = torch.cat(tokens, dim=1)

        # Extract positions for RoPE
        pos_list = gaussians[..., :self.config.pos_dim]
        pos_list_padded, valid_mask_padded = self._prepare_padded_positions(pos_list, valid_mask)

        return seq, valid_mask_padded, pos_list_padded

    def forward(self, gaussians, valid_mask, rays_o, rays_d, gaussians_view_tf, tf32_view_tf=False, fov=None,
                canvas=None):
        """
        Forward pass of the transformer.

        Args:
            gaussians: [batch_size, max_num_items, gaussian_dim], padded
            valid_mask: [batch_size, max_num_items], boolean mask for valid items.
            rays_o: [batch_size, num_views, 3]
            rays_d: [batch_size, num_views, img_h, img_w, 3]
            gaussians_view_tf: [batch_size, num_views, max_num_items, pos_dim], padded view-transformed positions
            tf32_view_tf: bool, whether to use tf32 for view transformer.
        """
        seq, valid_mask_padded, pos_list_padded = self.construct_sequence(
            gaussians, valid_mask
        )

        seq = self.transformer(seq, src_key_padding_mask=valid_mask_padded, spatial_pos=pos_list_padded)

        batch_size, num_views = rays_o.size(0), rays_o.size(1)
        seq = seq.repeat_interleave(num_views, dim=0)
        rays_o = rays_o.view(-1, *rays_o.shape[2:])
        rays_d = rays_d.view(-1, *rays_d.shape[2:])

        view_tf = gaussians_view_tf.reshape(-1, *gaussians_view_tf.shape[2:])
        pos_view_tf = view_tf[..., :self.config.pos_dim]
        # Optional camera-frame extras (scale 3 + quaternion 4) for the projection features.
        view_extra = view_tf[..., self.config.pos_dim:] if view_tf.size(-1) > self.config.pos_dim else None
        valid_mask_repeated = valid_mask.repeat_interleave(num_views, dim=0)

        pos_seq_view, valid_mask_padded_view = self._prepare_padded_positions(pos_view_tf, valid_mask_repeated)
        if view_extra is not None:
            view_extra = torch.cat([view_extra.new_zeros(view_extra.size(0), self.skip_token_num, view_extra.size(-1)),
                                    view_extra], dim=1)

        res = self.view_transformer(
            rays_o,
            rays_d,
            seq,
            pos_seq_view,
            valid_mask_padded_view,
            tf32_mode=tf32_view_tf,
            fov=None if fov is None else fov.reshape(-1),
            view_extra=view_extra,
            canvas=canvas,  # [B*V, H, W, 3] or None
        )

        res = res.view(
            batch_size,
            num_views,
            *res.size()[1:],
        )  # [batch_size * num_views, ...] -> [batch_size, num_views, ...]
        return res
