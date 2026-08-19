"""Warm DEPTH-EXPANSION of a GaussianFormer checkpoint (V19 direction).

Inserts new transformer blocks initialized to EXACT identity: in these pre-norm blocks every
residual branch ends in a single output matrix (attention `out_proj`, SwiGLU `ffn.w2`), so zeroing
those matrices makes the block compute x + 0. The expanded model therefore starts at exactly the
source model's function -- which matters here specifically because every damaged init we tried
this week (scratch at two widths, layer-dropped warm) fell into the LPIPS ~0.092 attractor and
never recovered. Identity expansion cannot fall in: epoch 0 IS the source model.

Placement:
  * encoder 12 -> 12+k: new layers interleaved evenly (output = last layer, so any placement
    preserves the function).
  * view transformer 6 -> 6+m: new layers only WITHIN THE FIRST (depth-4) slots -- the DPT taps
    the outputs of the LAST FOUR view layers (view_transformer.py: out_layers = last 4), so the
    old final four must stay in the final four slots for the taps to see identical features.

  uv run --no-sync python -m data_v10.make_expanded_ckpt \
    --src checkpoints_v18_256/phase2_epoch_30.pt --enc_add 6 --view_add 3 \
    --out checkpoints_depthexpand/init_e18v9.pt
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from gaussianformer.models.config import GaussianFormerConfig
from gaussianformer.models.gaussianformer import GaussianFormer

ZERO_SUFFIXES = ("multihead_attn.out_proj.weight", "self_attn.out_proj.weight", "ffn.w2.weight")


def placement(n_old: int, n_add: int, protect_last: int = 0) -> list[int | None]:
    """New-index list: old layer index or None (= fresh identity block).

    Old layers keep their relative order; None slots are spread evenly through the
    insertable prefix. With protect_last=P, the final P slots are forced to be the
    final P old layers (DPT tap protection)."""
    head_old = n_old - protect_last
    head_len = head_old + n_add
    slots: list[int | None] = []
    # even spread: place the n_add Nones at rounded positions in the head
    none_at = {round((i + 1) * head_len / (n_add + 1)) - 1 for i in range(n_add)}
    # collisions resolved by shifting -- guarantee exactly n_add Nones
    while len(none_at) < n_add:
        none_at.add(max(none_at) + 1)
    old_iter = iter(range(head_old))
    for pos in range(head_len):
        slots.append(None if pos in none_at else next(old_iter))
    slots += list(range(head_old, n_old))
    return slots


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--enc_add", type=int, default=6)
    ap.add_argument("--view_add", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    src = torch.load(args.src, map_location="cpu", weights_only=True)["model_state_dict"]
    n_enc, n_view = 12 + args.enc_add, 6 + args.view_add
    enc_map = placement(12, args.enc_add, protect_last=0)
    view_map = placement(6, args.view_add, protect_last=4)
    print(f"encoder map (new idx -> old): {enc_map}")
    print(f"view map    (new idx -> old): {view_map}")

    # Fresh model provides the random init for the new blocks (norms etc. keep their init;
    # only the residual-output matrices are zeroed).
    cfg = GaussianFormerConfig(pe_type="rope", num_layers=n_enc, view_transformer_n_layers=n_view)
    fresh = GaussianFormer(cfg).state_dict()

    out = {}
    for k, v in fresh.items():
        m = re.match(r"^(transformer\.layers|view_transformer\.transformer\.layers)\.(\d+)\.(.*)$", k)
        if m is None:
            out[k] = src[k] if k in src else v          # non-layer weights come from src verbatim
            continue
        root, idx, rest = m.group(1), int(m.group(2)), m.group(3)
        lmap = enc_map if root == "transformer.layers" else view_map
        old = lmap[idx]
        if old is not None:
            out[k] = src[f"{root}.{old}.{rest}"]
        elif rest in ZERO_SUFFIXES:
            out[k] = torch.zeros_like(v)                 # identity: residual branch outputs 0
        else:
            out[k] = v
    missing = [k for k in fresh if k not in out]
    assert not missing, missing

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": out}, args.out)
    print(f"{sum(t.numel() for t in src.values())/1e6:.1f}M -> "
          f"{sum(t.numel() for t in out.values())/1e6:.1f}M params; wrote {args.out}")


if __name__ == "__main__":
    main()
