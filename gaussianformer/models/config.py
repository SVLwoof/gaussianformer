from dataclasses import dataclass, field, fields, replace
import types
import typing
from typing import Literal


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
    rope_dim: int | None = None
    """Rotary dim per stage (channel pairs rotated = pos_dim * rope_dim/2). None = pos_pe_num_freqs
    (12: 18 of 64 pairs rotated, 1..5 rad/unit). Raising it rotates previously position-blind
    pairs, so a warm init needs a 256px recovery stage."""
    proj_rope_2d: bool = False
    """P2 (2026-09-09, positive): every Gaussian is perspective-projected into the view; its
    projected patch coordinate (u, v) gets a 2-D RoPE on the cross-attention KEYS and each ray
    token its patch centre on the QUERIES, in the channel pairs right after the pretrained 3-D
    ones. Alters pretrained function: needs a 256px recovery stage."""
    ray_rope_2d_dim: int = 16
    """Rotary dim of that 2-D RoPE (8 log-spaced freqs per axis)."""
    ray_rope_2d_scale: float = 0.25
    """Patch coordinates are multiplied by this before the 2-D RoPE: 0.25 -> 0.25..1.75 rad/patch
    (wavelengths 3.6..25 patches on the 64x64 grid at 512px)."""
    canvas_cond: bool = False
    """P1 (2026-09-10; line dropped 2026-09-17, kept to load its checkpoints): add a gsplat
    rasterization of the input splat (same camera, log10(x+1) space) to the ray tokens through a
    zero-init linear. Identity at init."""
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
    dpt_features: int = 128
    """The dim of internal features in the DPT decoder."""
    dpt_out_channels: list[int] = field(default_factory=lambda: [96, 192, 384, 768])
    """The number of output channels per layer in the DPT decoder."""
    dpt_out_layers: list[int] | None = None
    """The layers to use for the DPT decoder."""
    turn_to_cam_coord: bool = True
    """Whether to transform the scene to camera coordinates before rendering."""
    use_ldr: bool = False
    """Whether to run in LDR mode (as opposed to HDR)."""

    def get(self, key, default=None):
        return getattr(self, key, default)

    def with_overrides(self, pairs: list[str] | None) -> "GaussianFormerConfig":
        """Apply `key=value` overrides (CLI `--model_cfg`), coerced by the field's annotation.

        One generic knob for architecture changes instead of a flag per feature. Unknown keys
        raise; bools accept true/false/1/0; `X | None` coerces to X; list[int] is comma-separated.
        """
        if not pairs:
            return self
        hints = typing.get_type_hints(type(self))
        out = {}
        for p in pairs:
            key, _, raw = p.partition("=")
            if key not in hints:
                raise KeyError(f"unknown GaussianFormerConfig field {key!r}")
            out[key] = _coerce(hints[key], raw)
        return replace(self, **out)


def _coerce(tp, raw: str):
    if isinstance(tp, types.UnionType):  # X | None
        inner = [a for a in typing.get_args(tp) if a is not type(None)]
        return None if raw.lower() in ("none", "null") else _coerce(inner[0], raw)
    origin = typing.get_origin(tp)
    if origin is list:
        (item,) = typing.get_args(tp)
        return [_coerce(item, x) for x in raw.split(",") if x]
    if origin is Literal:
        assert raw in typing.get_args(tp), f"{raw!r} not in {typing.get_args(tp)}"
        return raw
    if tp is bool:
        return raw.lower() in ("1", "true", "yes")
    return tp(raw)
