from dataclasses import dataclass, field, fields, replace
from typing import Literal, List, Optional


@dataclass(frozen=True)
class GaussianFormerConfig:
    # --- Gaussian Config ---
    gaussian_dim: int = 14
    """The dimension of the Gaussian parameters (e.g., pos, scale, rot, color, opacity)."""
    pos_dim: int = 3
    """The dimension of the spatial position data (3 for Gaussians)."""

    # --- Core Transformer Config ---
    latent_dim: int = 768
    """The latent dimension of the transformer."""
    num_layers: int = 12
    """The number of layers in the transformer."""
    num_heads: int = 6
    """The number of heads in the transformer."""
    input_mlp_hidden: int = 0
    """Hidden dim of the residual input-head MLP (0 = single-linear baseline)."""
    dim_feedforward: int = 768 * 4
    """The dimension of the feedforward network in the transformer."""
    num_register_tokens: int = 16
    """The number of register/class tokens to use."""
    dropout: float = 0.0
    """The dropout rate in the transformer."""
    activation: Literal['gelu', 'swiglu'] = 'swiglu'
    """The activation function in the transformer."""
    norm_type: Literal['layer_norm', 'rms_norm'] = 'rms_norm'
    """The type of normalization to use in the transformer."""
    norm_first: bool = True
    """Whether to normalize the input before the transformer."""
    view_indep_qk_norm: bool = True
    """Whether to apply normalization to query and key in the view-independent transformer."""
    qk_norm: bool = True
    """Whether to apply normalization to query and key in the view-dependent transformer."""
    bias: bool = False
    """Whether to use bias in the transformer linear layers."""

    # --- Positional Encoding Config ---
    pe_type: Literal['nerf', 'rope'] = 'rope'
    """The type of positional encoding to use. 'rope': raw Gaussian -> Linear, position
    via RoPE only. 'nerf': NeRF-lifted position concatenated with the raw remaining
    fields -> single Linear, RoPE still on."""
    rope_double_max_freq: bool = False
    """Whether to double the max frequency for RoPE."""
    rope_dim: Optional[int] = None
    """Rotary dim per stage (channel pairs rotated = pos_dim * rope_dim/2). None = pos_pe_num_freqs
    (12: 18 of 64 pairs rotated, 1..5 rad/unit). Raising it rotates previously position-blind
    pairs, so a warm init needs a 256px recovery stage."""
    rope_pos_scale: float = 1.0
    """Positions are multiplied by this before RoPE in BOTH stages: k shifts the frequency band
    to k..5k rad per world unit (P3: spatial bandwidth of position-dependent attention)."""
    rope_hf_scale: float = 0.0
    """P3 (warm-safe variant): ADD a second 3-D RoPE band at this position scale in the channel
    pairs right after the pretrained ones (which stay untouched), in both stages. 0 = off.
    e.g. 8 -> extra band 8..40 rad/unit; rotates 18 previously position-blind pairs."""
    ray_rope_2d: bool = False
    """2-D RoPE on the patch-grid position for ray-token SELF-attention in the view transformer
    (today it is permutation-invariant: every patch carries the camera origin as its position)."""
    ray_rope_2d_dim: int = 16
    """Rotary dim of the 2-D ray RoPE (8 log-spaced freqs per axis), placed after the 3-D pairs."""
    proj_bias: bool = False
    """P2b: zero-init-gated cross-attention bias -|u_patch - u_gaussian|^2 / sigma^2 in PATCH units,
    from the explicit perspective projection of each Gaussian (replaces the low-contrast cos-angle
    geom_bias). Gaussians behind the camera get a large distance; register slots get 0."""
    proj_sigma_patches: float = 2.0
    """Width of the proximity bias in patches (2 = 16 px at patch 8)."""
    proj_feat: bool = False
    """P2c: inject per-view [log depth, log projected radius (px), camera-frame quaternion] into the
    context tokens through a zero-init linear, so the view stage sees depth and footprint."""
    proj_rope_2d: bool = False
    """P2a: 2-D RoPE on the projected patch coordinates for cross-attention KEYS and on patch
    centres for QUERIES (uses ray_rope_2d_dim / ray_rope_2d_scale). Alters pretrained function:
    needs a 256px recovery stage."""
    canvas_cond: bool = False
    """P1: add a gsplat rasterization of the input splat (same camera, log10(x+1) space) to the ray
    tokens through a zero-init linear. Identity at init; the model starts at rec-GT quality once
    the linear learns to pass the canvas through, and the transformer's job becomes refinement."""
    ray_rope_2d_scale: float = 0.25
    """Patch coordinates are multiplied by this before the 2-D RoPE: 0.25 -> 0.25..1.75 rad/patch
    (wavelengths 3.6..25 patches on the 64x64 grid at 512px)."""
    pos_pe_num_freqs: int = 12
    """The number of frequencies in the positional encoding for gaussian positions."""
    gaussian_encoder_norm_type: Literal['layer_norm', 'rms_norm'] = 'rms_norm'
    """The type of normalization to use in the gaussian encoder."""

    # --- View Transformer Config ---
    view_transformer_latent_dim: int = 768
    """The latent dimension of the view transformer."""
    view_transformer_ffn_hidden_dim: int = 768 * 4
    """The hidden dimension of the feedforward network in the view transformer."""
    view_transformer_n_heads: int = 6
    """The number of heads in the view transformer."""
    view_transformer_n_layers: int = 6
    """The number of layers in the view transformer."""
    view_transformer_include_self_attn: bool = True
    """Whether to include self-attention between ray tokens in the view transformer."""
    view_transformer_use_swin_attn: bool = False
    """Whether to use swin self-attention in the view transformer."""
    vdir_pe_type: Literal['nerf'] = 'nerf'
    """The type of positional encoding to use for view direction."""
    vdir_num_freqs: int = 0
    """The number of frequencies in the positional encoding for view direction."""
    patch_size: int = 8
    """The size of the image patch in the view transformer."""
    include_alpha: bool = False
    """Whether to include the alpha channel in the output."""
    use_dpt_decoder: bool = True
    """Whether to use DPT decoder for rendering."""
    geom_bias: bool = False
    """Add a zero-init-gated ray/Gaussian alignment bias to the view transformer's
    cross-attention logits. RoPE gives every patch token the same position (the camera
    origin), so without this the logits carry no per-patch geometry."""
    dpt_features: int = 128
    """The dim of internal features in the DPT decoder."""
    dpt_out_channels: List[int] = field(default_factory=lambda: [96, 192, 384, 768])
    """The number of output channels per layer in the DPT decoder."""
    dpt_out_layers: Optional[List[int]] = None
    """The layers to use for the DPT decoder."""
    turn_to_cam_coord: bool = True
    """Whether to transform the scene to camera coordinates before rendering."""
    use_ldr: bool = False
    """Whether to run in LDR mode (as opposed to HDR)."""

    def get(self, key, default=None):
        return getattr(self, key, default)

    def with_overrides(self, pairs: list[str] | None) -> "GaussianFormerConfig":
        """Apply `key=value` overrides (CLI `--model_cfg`), coercing to the field's type.

        One generic knob for architecture probes instead of a flag per feature. Unknown keys
        raise; bools accept true/false/1/0; None-defaulted fields try int, then float, then str.
        """
        if not pairs:
            return self
        types = {f.name: f for f in fields(self)}
        out = {}
        for p in pairs:
            key, _, raw = p.partition("=")
            if key not in types:
                raise KeyError(f"unknown GaussianFormerConfig field {key!r}")
            cur = getattr(self, key)
            if isinstance(cur, bool):
                val = raw.lower() in ("1", "true", "yes")
            elif isinstance(cur, int):
                val = int(raw)
            elif isinstance(cur, float):
                val = float(raw)
            elif cur is None:
                val = raw
                for cast in (int, float):
                    try:
                        val = cast(raw)
                        break
                    except ValueError:
                        pass
            else:
                val = raw
            out[key] = val
        return replace(self, **out)
