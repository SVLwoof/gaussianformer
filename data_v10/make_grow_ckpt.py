"""Function-preserving growth ckpts: FFN widen (zero new w2 cols) / input-head MLP (zero out)."""
from __future__ import annotations
import argparse, re
from pathlib import Path
import torch
from gaussianformer.models.config import GaussianFormerConfig
from gaussianformer.models.gaussianformer import GaussianFormer

ap = argparse.ArgumentParser()
ap.add_argument("--src", type=Path, required=True)
ap.add_argument("--ffn_mult", type=int, default=4)
ap.add_argument("--input_mlp_hidden", type=int, default=0)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
src = torch.load(a.src, map_location="cpu", weights_only=True)["model_state_dict"]
cfg = GaussianFormerConfig(pe_type="rope", dim_feedforward=768*a.ffn_mult,
                           view_transformer_ffn_hidden_dim=768*a.ffn_mult,
                           input_mlp_hidden=a.input_mlp_hidden)
fresh = GaussianFormer(cfg).state_dict()
out = {}
for k, v in fresh.items():
    if k not in src:
        out[k] = v                       # new module (input MLP): fresh init, out already zeroed
    elif v.shape == src[k].shape:
        out[k] = src[k]
    elif re.search(r"ffn\.(w1|w3)\.weight$", k):
        v = v.clone(); v[:src[k].shape[0]] = src[k]; out[k] = v          # old rows kept
    elif re.search(r"ffn\.w2\.weight$", k):
        v = torch.zeros_like(v); v[:, :src[k].shape[1]] = src[k]; out[k] = v  # new cols ZERO
    else:
        raise SystemExit(f"unhandled shape change: {k} {src[k].shape}->{v.shape}")
a.out.parent.mkdir(parents=True, exist_ok=True)
torch.save({"model_state_dict": out}, a.out)
print(f"{sum(t.numel() for t in src.values())/1e6:.1f}M -> {sum(t.numel() for t in out.values())/1e6:.1f}M -> {a.out}")
