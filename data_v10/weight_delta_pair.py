"""How do two fine-tunes of the same seed differ? Per-family comparison of their weight deltas.

For fine-tunes A and B of seed S (dA = A - S, dB = B - S), per module family:
  rel A / rel B      ||dA|| / ||S||, ||dB|| / ||S||           (how far each moved)
  cos(dA, dB)        alignment of the two updates (1 = same direction, 0 = unrelated)
  ||A - B|| / ||S||  distance between the two fine-tunes
  share of ||A-B||^2 which families carry the difference between the two models
Per-layer rows for the view transformer (where P1/P2 act). Keys present in only one model
(new modules) are reported by their Frobenius norm relative to the module they add into.

Usage: uv run --no-sync python data_v10/weight_delta_pair.py SEED.pt A.pt B.pt
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict

import torch

from data_v10.weight_delta import FAMILIES, family, load_sd


def layer_family(name: str) -> str | None:
    m = re.match(r"^view_transformer\.transformer\.layers\.(\d+)\.(self_attn|multihead_attn|ffn)", name)
    if m:
        return f"view L{int(m.group(1)):02d} {m.group(2)}"
    m = re.match(r"^transformer\.layers\.(\d+)\.(multihead_attn|ffn)", name)
    if m:
        return f"scene L{int(m.group(1)):02d} {m.group(2)}"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seed"); ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--layers", action="store_true", help="also print per-layer rows")
    args = ap.parse_args()
    S, A, B = load_sd(args.seed), load_sd(args.a), load_sd(args.b)
    acc = defaultdict(lambda: {"s2": 0.0, "a2": 0.0, "b2": 0.0, "ab": 0.0, "d2": 0.0, "n": 0})
    for k, s in S.items():
        if k not in A or k not in B or A[k].shape != s.shape or B[k].shape != s.shape:
            continue
        da, db = A[k] - s, B[k] - s
        for key in (family(k), layer_family(k)):
            if key is None:
                continue
            r = acc[key]
            r["s2"] += (s ** 2).sum().item(); r["a2"] += (da ** 2).sum().item(); r["b2"] += (db ** 2).sum().item()
            r["ab"] += (da * db).sum().item(); r["d2"] += ((da - db) ** 2).sum().item(); r["n"] += s.numel()
    tot_d2 = sum(r["d2"] for f, r in acc.items() if f in dict(FAMILIES))
    print(f"A = {args.a}\nB = {args.b}\nseed = {args.seed}\n")
    hdr = f"{'family':32s} {'params':>11s} {'relA':>7s} {'relB':>7s} {'cos(dA,dB)':>10s} {'|A-B|/|S|':>9s} {'share':>6s}"
    def row(f, r):
        cos = r["ab"] / ((r["a2"] * r["b2"]) ** 0.5 + 1e-30)
        return (f"{f:32s} {r['n']:11,d} {(r['a2']/r['s2'])**0.5:7.4f} {(r['b2']/r['s2'])**0.5:7.4f} "
                f"{cos:10.3f} {(r['d2']/r['s2'])**0.5:9.4f} {100*r['d2']/tot_d2:5.1f}%")
    print(hdr)
    for f, _ in FAMILIES:
        if f in acc:
            print(row(f, acc[f]))
    if args.layers:
        print("\nper layer (share = of the whole-model ||A-B||^2)")
        print(hdr)
        for f in sorted(k for k in acc if k.startswith(("view L", "scene L"))):
            print(row(f, acc[f]))
    for label, M in (("A", A), ("B", B)):
        new = [k for k in M if k not in S or M[k].shape != S[k].shape]
        if new:
            print(f"\nkeys only in {label} (or reshaped):")
            for k in new:
                print(f"  {k:60s} shape {list(M[k].shape)}  ||W||_F = {M[k].norm().item():.4f}")
    ref = S.get("view_transformer.ray_map_encoder.weight")
    if ref is not None:
        print(f"\nreference: ||ray_map_encoder.weight||_F (seed) = {ref.norm().item():.4f}")


if __name__ == "__main__":
    main()
