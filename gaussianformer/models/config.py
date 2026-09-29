import types
import typing
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class GaussianFormerConfig:
    # Scene encoder: one token per Gaussian, 3-D RoPE on its position.
    latent_dim: int = 768
    num_layers: int = 12
    num_heads: int = 6
    dim_feedforward: int = 768 * 4
    num_register_tokens: int = 16
    pos_pe_num_freqs: int = 12
    """Rotary dimension of the 3-D position RoPE (shared with the view decoder)."""

    # View decoder: one token per image patch, cross-attention to the scene tokens, DPT head.
    view_transformer_latent_dim: int = 768
    view_transformer_ffn_hidden_dim: int = 768 * 4
    view_transformer_n_heads: int = 6
    view_transformer_n_layers: int = 6
    patch_size: int = 8
    ray_embed_patch: int = 0
    """If larger than patch_size, each patch's ray directions are upsampled to this size before the
    ray embedding, so a finer patch grid reuses an embedding pretrained at this patch size."""
    dpt_features: int = 128
    dpt_out_channels: list[int] = field(default_factory=lambda: [96, 192, 384, 768])

    proj_rope_2d: bool = False
    """2-D RoPE in the cross-attention: keys carry each Gaussian's projected image position, queries
    their patch centre; patch self-attention uses the same 2-D band."""
    ray_rope_2d_dim: int = 16
    ray_rope_2d_scale: float = 0.25
    """Patch coordinates are multiplied by this before the 2-D RoPE."""

    xattn_window: int = 0
    """Tile size in patches for windowed cross-attention: each tile attends only to the Gaussians
    whose projected footprint reaches it, plus the register tokens. 0 = dense."""
    xattn_margin: float = 1.0
    """Tile expansion in patches on top of the footprint."""
    xattn_sigmas: float = 3.0
    """Footprint radius in units of the Gaussian's largest scale."""

    view_bf16: bool = False
    """Run the view decoder in bf16 (Flash Attention for the patch self-attention)."""
    view_grad_checkpoint: bool = False
    """Recompute view-decoder layers in the backward pass to save activation memory."""

    def with_overrides(self, pairs: list[str] | None) -> "GaussianFormerConfig":
        """Apply `key=value` overrides (the `--model_cfg` option)."""
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
    if isinstance(tp, types.UnionType):
        inner = [a for a in typing.get_args(tp) if a is not type(None)]
        return None if raw.lower() in ("none", "null") else _coerce(inner[0], raw)
    if typing.get_origin(tp) is list:
        (item,) = typing.get_args(tp)
        return [_coerce(item, x) for x in raw.split(",") if x]
    if tp is bool:
        return raw.lower() in ("1", "true", "yes")
    return tp(raw)
