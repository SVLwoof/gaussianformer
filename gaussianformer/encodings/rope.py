# Modified from https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py

# Copyright (c) 2021 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from math import log

import torch
from einops import rearrange
from torch import Tensor, einsum, nn
from torch.amp import autocast


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def freqs_to_cos_sin(freqs: Tensor, head_dim: int) -> tuple[Tensor, Tensor]:
    """[..., 2F] frequencies (duplicated halves) -> cos, sin padded with zero angles to [..., head_dim]."""
    freqs = freqs[..., : freqs.shape[-1] // 2]
    pad = torch.zeros((*freqs.shape[:-1], head_dim // 2 - freqs.shape[-1]), device=freqs.device)
    freqs = torch.cat((freqs, pad), dim=-1)
    freqs = torch.cat([freqs, freqs], dim=-1)
    return freqs.cos(), freqs.sin()


@autocast("cuda", enabled=False)
def apply_rotary_emb(t: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """t [B, H, N, head_dim], cos/sin [B, 1, N, head_dim]."""
    return ((t * cos) + (rotate_half(t) * sin)).type(t.dtype)


class SpatialRotaryEmbedding(nn.Module):
    """Rotary embedding of `pos_dim`-D positions: dim // 2 log-spaced frequencies per axis."""

    def __init__(self, dim: int, pos_dim: int = 3, pos_scale: float = 1.0):
        super().__init__()
        self.pos_dim = pos_dim
        self.pos_scale = pos_scale
        self.register_buffer("freqs", 2 ** torch.linspace(0, log(dim // 2 - 1, 2), dim // 2))

    @autocast("cuda", enabled=False)
    def get_spatial_freqs(self, pos: Tensor) -> Tensor:
        """pos [B, N, pos_dim] -> angles [B, 1, N, 2 * pos_dim * dim // 2] (halves duplicated)."""
        freqs = einsum("... i, f -> ... i f", (pos * self.pos_scale).type(self.freqs.dtype), self.freqs)
        freqs = rearrange(freqs, "b n i f -> b 1 n (i f)")
        return torch.cat([freqs, freqs], dim=-1)
