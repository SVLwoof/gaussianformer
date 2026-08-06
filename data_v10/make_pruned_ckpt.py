"""Layer-prune a GaussianFormer checkpoint for the depth-capacity probe.

Keeps EVEN-indexed layers (0,2,4,...) and renumbers them densely, for both the
view-independent encoder and the view transformer. Everything else (input encoder,
register tokens, DPT, norms) is copied verbatim -- d=768 is unchanged, so all
non-layer weights remain shape-compatible. Emits a normal {'model_state_dict': ...}
checkpoint that --init_from loads into a model built with the reduced
--encoder_layers/--view_layers.

  uv run --no-sync python -m data_v10.make_pruned_ckpt \
    --src checkpoints_v18_256/phase2_epoch_30.pt --enc 6 --view 3 \
    --out checkpoints_depthprune/init_e6v3.pt
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--enc", type=int, default=6, help="encoder layers to keep (from 12)")
    ap.add_argument("--view", type=int, default=3, help="view-transformer layers to keep (from 6)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sd = torch.load(args.src, map_location="cpu", weights_only=True)["model_state_dict"]
    keep_enc = {2 * i: i for i in range(args.enc)}    # even indices, renumbered densely
    keep_view = {2 * i: i for i in range(args.view)}

    out, dropped = {}, 0
    for k, v in sd.items():
        m = re.match(r"^(transformer\.layers|view_transformer\.transformer\.layers)\.(\d+)\.(.*)$", k)
        if m is None:
            out[k] = v
            continue
        root, idx, rest = m.group(1), int(m.group(2)), m.group(3)
        keep = keep_enc if root == "transformer.layers" else keep_view
        if idx in keep:
            out[f"{root}.{keep[idx]}.{rest}"] = v
        else:
            dropped += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": out}, args.out)
    n_in = sum(v.numel() for v in sd.values())
    n_out = sum(v.numel() for v in out.values())
    print(f"kept enc layers {sorted(keep_enc)} -> 0..{args.enc-1}, "
          f"view {sorted(keep_view)} -> 0..{args.view-1}")
    print(f"{len(sd)} -> {len(out)} tensors ({dropped} dropped), "
          f"{n_in/1e6:.1f}M -> {n_out/1e6:.1f}M params")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
