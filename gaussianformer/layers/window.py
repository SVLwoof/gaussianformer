"""Projection-windowed cross-attention: each ray token attends only to the Gaussians whose
projected footprint reaches its image tile (plus the register tokens), the way a rasterizer
bins splats by tile. Cost goes from queries x all Gaussians to queries x local Gaussians.

`build_window` turns the per-view projection into a flat (tile -> key indices) structure that
`windowed_attention` consumes with flash-attn varlen (one "sequence" per tile) or, on CPU /
SDPA, with a padded gather. Built once per forward and shared by every decoder layer.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def build_window(uv: Tensor, radius: Tensor, key_ok: Tensor, n_reg: int, patch_h: int, patch_w: int,
                 tile: int, margin: float) -> dict:
    """
    uv      [B, N, 2]  projected key centres in patch units (registers included, ignored)
    radius  [B, N]     projected footprint radius in patch units
    key_ok  [B, N]     valid Gaussian keys (False for padding, behind-camera, registers)
    n_reg               register tokens (keys 0..n_reg-1), attended by every tile
    Returns a dict: q_perm/q_inv [S] query permutation to tile-major order, kv_index [total_k]
    flat (b*N + n) key indices in tile order, cu_q/cu_k [B*T+1] int32 segment offsets,
    max_k, plus tiles/tile_q for reshaping.
    """
    B, N = uv.shape[:2]
    assert patch_h % tile == 0 and patch_w % tile == 0, "the ray grid must be a whole number of tiles"
    th, tw = patch_h // tile, patch_w // tile
    T = th * tw
    dev = uv.device
    # queries (raster order h1 w1) -> tile-major (tile, y, x)
    ii, jj = torch.meshgrid(torch.arange(patch_h, device=dev), torch.arange(patch_w, device=dev), indexing="ij")
    tile_of_q = (ii // tile) * tw + (jj // tile)
    q_perm = torch.argsort(tile_of_q.reshape(-1), stable=True)
    q_inv = torch.empty_like(q_perm)
    q_inv[q_perm] = torch.arange(q_perm.numel(), device=dev)
    # per-Gaussian tile range from the footprint bbox (+ margin), clipped to the grid
    lo = (uv - (radius + margin)[..., None]).div(tile).floor().clamp_min(0).long()
    hi = (uv + (radius + margin)[..., None]).div(tile).floor().long()
    hi = torch.minimum(hi, torch.tensor([tw - 1, th - 1], device=dev))
    ok = key_ok & (hi >= lo).all(-1)
    lo = lo.masked_fill(~ok[..., None], 0)
    hi = hi.masked_fill(~ok[..., None], -1)
    w = (hi[..., 0] - lo[..., 0] + 1).clamp_min(0)
    h = (hi[..., 1] - lo[..., 1] + 1).clamp_min(0)
    cnt = (w * h).reshape(-1)  # [B*N] tiles touched per key
    key = torch.arange(B * N, device=dev).repeat_interleave(cnt)
    local = torch.arange(key.numel(), device=dev) - torch.repeat_interleave(cnt.cumsum(0) - cnt, cnt)
    wk = w.reshape(-1)[key]
    dx, dy = local % wk, local // wk
    tx = lo[..., 0].reshape(-1)[key] + dx
    ty = lo[..., 1].reshape(-1)[key] + dy
    seg = (key // N) * T + ty * tw + tx  # (b, tile) segment
    # register tokens in every tile
    reg_seg = torch.arange(B * T, device=dev).repeat_interleave(n_reg)
    reg_key = (reg_seg // T) * N + torch.arange(n_reg, device=dev).repeat(B * T)
    seg = torch.cat([reg_seg, seg])
    key = torch.cat([reg_key, key])
    order = torch.argsort(seg, stable=True)
    seg, key = seg[order], key[order]
    counts = torch.bincount(seg, minlength=B * T)
    cu_k = F.pad(counts.cumsum(0), (1, 0)).to(torch.int32)
    cu_q = torch.arange(0, B * T + 1, device=dev, dtype=torch.int32) * (tile * tile)
    return dict(q_perm=q_perm, q_inv=q_inv, kv_index=key, cu_q=cu_q, cu_k=cu_k,
                max_k=int(counts.max()), tiles=T, tile_q=tile * tile, B=B, N=N)


def windowed_attention(q: Tensor, k: Tensor, v: Tensor, win: dict, flash_available: bool) -> Tensor:
    """q [B, H, S, d] (raster order), k/v [B, H, N, d] (RoPE already applied) -> [B, S, H*d].
    flash-attn varlen (one sequence per tile, no padding) whenever it is installed and the tensors
    are on the GPU -- in bf16, since flash has no fp32 kernel and the view stage runs under tf32;
    otherwise a padded gather + SDPA (CPU). The padded path costs tiles x max_k, so it is only
    for tests: a single tile can hold most of a small object's Gaussians."""
    B, H, S, d = q.shape
    N, T, tq = win["N"], win["tiles"], win["tile_q"]
    q_t = q[:, :, win["q_perm"]]  # tile-major queries
    if flash_available and q.is_cuda:
        from flash_attn import flash_attn_varlen_kvpacked_func
        ht = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        kv = torch.stack([k, v], 2).permute(0, 3, 2, 1, 4).reshape(B * N, 2, H, d)[win["kv_index"]].to(ht)  # [total_k, 2, H, d]
        q_flat = q_t.permute(0, 2, 1, 3).reshape(B * S, H, d).to(ht)
        out = flash_attn_varlen_kvpacked_func(q_flat, kv, win["cu_q"], win["cu_k"], tq, win["max_k"])
        out = out.reshape(B, S, H * d).to(q.dtype)
    else:
        k_flat = k.transpose(1, 2).reshape(B * N, H, d)[win["kv_index"]]  # [total_k, H, d]
        v_flat = v.transpose(1, 2).reshape(B * N, H, d)[win["kv_index"]]
        # padded gather: one row per (b, tile), keys padded to max_k and masked
        Kmax = win["max_k"]
        counts = (win["cu_k"][1:] - win["cu_k"][:-1]).long()
        pos = torch.arange(win["kv_index"].numel(), device=q.device) - torch.repeat_interleave(win["cu_k"][:-1].long(), counts)
        seg = torch.repeat_interleave(torch.arange(B * T, device=q.device), counts)
        kp = k_flat.new_zeros(B * T, Kmax, H, d)
        vp = v_flat.new_zeros(B * T, Kmax, H, d)
        mask = torch.zeros(B * T, Kmax, dtype=torch.bool, device=q.device)
        kp[seg, pos], vp[seg, pos], mask[seg, pos] = k_flat, v_flat, True
        qt = q_t.reshape(B, H, T, tq, d).permute(0, 2, 1, 3, 4).reshape(B * T, H, tq, d)
        out = F.scaled_dot_product_attention(qt, kp.transpose(1, 2), vp.transpose(1, 2),
                                             attn_mask=mask[:, None, None, :])  # [B*T, H, tq, d]
        out = out.reshape(B, T, H, tq, d).permute(0, 1, 3, 2, 4).reshape(B, S, H * d)
    return out[:, win["q_inv"]]
