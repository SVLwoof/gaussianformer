import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange

from gaussianformer.encodings.rope import SpatialRotaryEmbedding, apply_rotary_emb, freqs_to_cos_sin
from gaussianformer.layers.window import windowed_attention

EPS = 1e-6

ATTN = os.environ.get("ATTN_IMPL", "flash_attn")
assert ATTN in ("flash_attn", "sdpa"), "ATTN_IMPL must be 'flash_attn' or 'sdpa'"
if ATTN == "flash_attn":
    try:
        from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_kvpacked_func, flash_attn_varlen_qkvpacked_func
        from flash_attn.bert_padding import pad_input, unpad_input
    except ImportError:
        print("flash_attn is not installed, falling back to PyTorch SDPA.")
        ATTN = "sdpa"


class FeedForwardSwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MultiHeadAttention(nn.Module):
    """Self-attention (kv_dim None) or cross-attention with QK-norm and RoPE."""

    def __init__(self, query_dim: int, num_heads: int, kv_dim: int | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.is_self_attn = kv_dim is None
        if self.is_self_attn:
            self.in_proj = nn.Linear(query_dim, 3 * query_dim, bias=False)
        else:
            self.q_proj = nn.Linear(query_dim, query_dim, bias=False)
            self.k_proj = nn.Linear(kv_dim, query_dim, bias=False)
            self.v_proj = nn.Linear(kv_dim, query_dim, bias=False)
        self.out_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.q_norm = nn.RMSNorm(query_dim, eps=EPS)
        self.k_norm = nn.RMSNorm(query_dim, eps=EPS)

    def forward(self, q, kv, key_mask=None, rope_q=None, rope_k=None, force_sdpa=False, window=None):
        """
        q [B, S, D], kv [B, N, D_kv]; key_mask [B, N] (True = attend); rope_q / rope_k (cos, sin) tables
        [B, 1, S|N, head_dim] (rope_k defaults to rope_q); window from layers.window.build_window.
        """
        bs, src_len, ctx_len = q.shape[0], q.shape[1], kv.shape[1]
        if self.is_self_attn:
            q, k, v = self.in_proj(q).chunk(3, dim=-1)
        else:
            q, k, v = self.q_proj(q), self.k_proj(kv), self.v_proj(kv)
        q = self.q_norm(q).type(v.dtype)
        k = self.k_norm(k).type(v.dtype)

        q = q.view(bs, src_len, self.num_heads, -1).transpose(1, 2)  # [B, H, S, d]
        k = k.view(bs, ctx_len, self.num_heads, -1).transpose(1, 2)
        v = v.view(bs, ctx_len, self.num_heads, -1).transpose(1, 2)
        if rope_q is not None:
            rope_k = rope_q if rope_k is None else rope_k
            q = apply_rotary_emb(q, *rope_q)
            k = apply_rotary_emb(k, *rope_k)
        q, k = q.type(v.dtype), k.type(v.dtype)

        if window is not None:
            out = windowed_attention(q, k, v, window, flash_available=ATTN == "flash_attn")
        elif ATTN == "sdpa" or force_sdpa:
            mask = None
            if key_mask is not None:
                mask = key_mask.view(bs, 1, 1, ctx_len).expand(-1, self.num_heads, -1, -1).reshape(
                    bs, self.num_heads, 1, ctx_len)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            out = out.transpose(1, 2).reshape(bs, src_len, -1)
        elif self.is_self_attn and key_mask is None:
            qkv = torch.stack([q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)], dim=2)
            out = flash_attn_qkvpacked_func(qkv).reshape(bs, src_len, -1)
        elif self.is_self_attn:
            q_u, idx, cu, max_len, _ = unpad_input(q.transpose(1, 2), key_mask)
            k_u = unpad_input(k.transpose(1, 2), key_mask)[0]
            v_u = unpad_input(v.transpose(1, 2), key_mask)[0]
            out = flash_attn_varlen_qkvpacked_func(torch.stack([q_u, k_u, v_u], dim=1), cu, max_len)
            out = pad_input(out, idx, bs, src_len).reshape(bs, src_len, -1)
        else:
            q_u = rearrange(q, "b h s d -> (b s) h d")
            cu_q = torch.arange(0, (bs + 1) * src_len, step=src_len, dtype=torch.int32, device=q.device)
            k_u, _, cu_k, max_k, _ = unpad_input(k.transpose(1, 2), key_mask)
            v_u = unpad_input(v.transpose(1, 2), key_mask)[0]
            out = flash_attn_varlen_kvpacked_func(q_u, torch.stack([k_u, v_u], dim=1), cu_q, cu_k, src_len, max_k)
            out = rearrange(out, "(b s) h d -> b s (h d)", b=bs)
        return self.out_proj(out)


class AttentionLayer(nn.Module):
    """Pre-norm block: attention (self, or cross + patch self-attention), then a SwiGLU FFN."""

    def __init__(self, dim: int, num_heads: int, ffn_hidden_dim: int, kv_dim: int | None = None):
        super().__init__()
        self.query_norm = nn.RMSNorm(dim, eps=EPS)
        self.multihead_attn = MultiHeadAttention(dim, num_heads, kv_dim)
        if kv_dim is not None:
            self.kv_norm = nn.RMSNorm(kv_dim, eps=EPS)
            self.self_attn_norm = nn.RMSNorm(dim, eps=EPS)
            self.self_attn = MultiHeadAttention(dim, num_heads)
        self.ffn_norm = nn.RMSNorm(dim, eps=EPS)
        self.ffn = FeedForwardSwiGLU(dim, ffn_hidden_dim)

    def forward(self, x, ctx=None, key_mask=None, rope_q=None, rope_k=None, force_sdpa=False, window=None):
        q = self.query_norm(x)
        if ctx is None:
            x = x + self.multihead_attn(q, q, key_mask, rope_q, force_sdpa=force_sdpa)
        else:
            x = x + self.multihead_attn(q, self.kv_norm(ctx), key_mask, rope_q, rope_k, force_sdpa, window)
            q = self.self_attn_norm(x)
            x = x + self.self_attn(q, q, None, rope_q, force_sdpa=force_sdpa)
        return x + self.ffn(self.ffn_norm(x))


class TransformerEncoder(nn.Module):
    """Scene encoder: self-attention over the Gaussian and register tokens with 3-D RoPE."""

    def __init__(self, num_layers: int, num_heads: int, dim: int, ffn_hidden_dim: int, rope_dim: int):
        super().__init__()
        self.head_dim = dim // num_heads
        self.layers = nn.ModuleList([AttentionLayer(dim, num_heads, ffn_hidden_dim) for _ in range(num_layers)])
        self.rope_emb = SpatialRotaryEmbedding(rope_dim)

    def forward(self, x, key_mask, pos):
        rope = freqs_to_cos_sin(self.rope_emb.get_spatial_freqs(pos), self.head_dim)
        for layer in self.layers:
            x = layer(x, key_mask=key_mask, rope_q=rope)
        return x


class TransformerDecoder(nn.Module):
    """View decoder: patch tokens cross-attend to the scene tokens, then self-attend.

    RoPE per head: the pretrained 3-D band (camera origin for patches, Gaussian position for
    Gaussians), then with proj_rope_2d a 2-D band in the channel pairs right after it
    (patch centre for patches, projected position for Gaussians).
    """

    def __init__(self, num_layers: int, num_heads: int, dim: int, ctx_dim: int, ffn_hidden_dim: int,
                 rope_dim: int, proj_rope_2d: bool, ray_rope_2d_dim: int, ray_rope_2d_scale: float,
                 grad_checkpoint: bool):
        super().__init__()
        self.head_dim = dim // num_heads
        self.layers = nn.ModuleList([AttentionLayer(dim, num_heads, ffn_hidden_dim, ctx_dim)
                                     for _ in range(num_layers)])
        self.rope_emb = SpatialRotaryEmbedding(rope_dim)
        if proj_rope_2d:
            assert 3 * rope_dim // 2 + ray_rope_2d_dim <= self.head_dim // 2, "not enough head channels for 2-D RoPE"
            self.rope_uv = SpatialRotaryEmbedding(ray_rope_2d_dim, pos_dim=2, pos_scale=ray_rope_2d_scale)
        self.grad_checkpoint = grad_checkpoint

    def _rope(self, pos, uv=None):
        bands = [self.rope_emb.get_spatial_freqs(pos)]
        if uv is not None:
            bands.append(self.rope_uv.get_spatial_freqs(uv))
        f = torch.cat([b[..., : b.shape[-1] // 2] for b in bands], -1)
        return freqs_to_cos_sin(torch.cat([f, f], -1), self.head_dim)

    def forward(self, x, ctx, key_mask, ctx_pos, ray_pos, out_layers, force_sdpa=False,
                uv_q=None, uv_k=None, window=None):
        kw = dict(key_mask=key_mask, rope_q=self._rope(ray_pos, uv_q), rope_k=self._rope(ctx_pos, uv_k),
                  force_sdpa=force_sdpa, window=window)
        out = []
        for i, layer in enumerate(self.layers):
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(lambda x_, layer_=layer: layer_(x_, ctx, **kw), x,
                                                      use_reentrant=False)
            else:
                x = layer(x, ctx, **kw)
            if i in out_layers:
                out.append(x)
        return out
