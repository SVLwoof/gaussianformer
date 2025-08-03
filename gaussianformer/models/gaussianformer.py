import torch
from torch import nn
from torch.amp import autocast

from huggingface_hub import PyTorchModelHubMixin

from gaussianformer.encodings.nerf_encoding import NeRFEncoding
from gaussianformer.layers.attention import TransformerEncoder
from gaussianformer.models.view_transformer import ViewTransformer
from gaussianformer.models.config import GaussianFormerConfig


class GaussianFormer(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: GaussianFormerConfig):
        super(GaussianFormer, self).__init__()
        self.config = config

        # --- Input Encoders ---
        self.gaussian_encoder = nn.Linear(self.config.gaussian_dim, self.config.latent_dim)
        norm_class = nn.LayerNorm if self.config.gaussian_encoder_norm_type == 'layer_norm' else nn.RMSNorm
        self.gaussian_encoder_norm = norm_class(self.config.latent_dim)
        self.gaussian_token = nn.Parameter(torch.randn(1, 1, self.config.latent_dim))

        if self.config.pe_type == 'nerf':
            self.gaussian_pos_pe = NeRFEncoding(in_dim=3, num_frequencies=self.config.pos_pe_num_freqs,
                                                include_input=True)
            self.gaussian_encoding_proj = nn.Linear(self.gaussian_pos_pe.get_out_dim(), self.config.latent_dim)

        # --- Positional Encoding ---
        self.rope_dim = None
        if self.config.pe_type == 'rope':
            self.rope_dim = self.config.pos_pe_num_freqs

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
            rope_double_max_freq=self.config.rope_double_max_freq
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

        # Encode gaussian parameters into an embedding
        gaussian_emb = self.gaussian_encoder_norm(self.gaussian_encoder(gaussians))

        tokens = [self.reg_tokens.expand(batch_size, -1, -1)]

        # Add positional encoding if using NeRF-style PE
        if self.config.pe_type == 'nerf':
            # Assuming the first 3 dimensions of gaussians are position
            gaussian_pos = gaussians[..., :self.config.pos_dim]
            pos_pe = self.gaussian_pos_pe(gaussian_pos)
            pos_emb = self.gaussian_encoding_proj(pos_pe)
            tokens.append(self.gaussian_token + gaussian_emb + pos_emb)
        elif self.config.pe_type == 'rope':
            tokens.append(self.gaussian_token + gaussian_emb)

        seq = torch.cat(tokens, dim=1)

        # Extract positions for RoPE
        pos_list = gaussians[..., :self.config.pos_dim]
        pos_list_padded, valid_mask_padded = self._prepare_padded_positions(pos_list, valid_mask)

        return seq, valid_mask_padded, pos_list_padded

    def forward(self, gaussians, valid_mask, rays_o, rays_d, gaussians_view_tf, tf32_view_tf=False):
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

        pos_view_tf = gaussians_view_tf.reshape(-1, *gaussians_view_tf.shape[2:])
        valid_mask_repeated = valid_mask.repeat_interleave(num_views, dim=0)

        pos_seq_view, valid_mask_padded_view = self._prepare_padded_positions(pos_view_tf, valid_mask_repeated)

        res = self.view_transformer(
            rays_o,
            rays_d,
            seq,
            pos_seq_view,
            valid_mask_padded_view,
            tf32_mode=tf32_view_tf
        )

        res = res.view(
            batch_size,
            num_views,
            *res.size()[1:],
        )  # [batch_size * num_views, ...] -> [batch_size, num_views, ...]
        return res
