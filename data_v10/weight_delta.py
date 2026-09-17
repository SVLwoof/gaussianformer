"""Where did per-object fine-tuning move the weights? Seed vs fine-tuned checkpoint deltas.

Per parameter: rel = ||W_ft - W_0||_F / ||W_0||_F. Grouped by module family. For 2-D weights
also the effective rank of the delta: #singular values needed for 90% / 99% of ||dW||_F^2,
and the fraction of delta energy captured by the top-r components for LoRA-relevant r.

Usage: uv run --no-sync python data_v10/weight_delta.py SEED.pt FT.pt [FT2.pt ...] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import torch

FAMILIES = [
    ("input head (gaussian_encoder)", r"^gaussian_encoder\."),
    ("scene attn in_proj", r"^transformer\.layers\.\d+\.multihead_attn\.in_proj"),
    ("scene attn out_proj", r"^transformer\.layers\.\d+\.multihead_attn\.out_proj"),
    ("scene ffn", r"^transformer\.layers\.\d+\.ffn\."),
    ("scene norms/qk-norm", r"^transformer\.layers\.\d+\..*norm"),
    ("scene final norm / registers", r"^transformer\.(norm|register|cls)"),
    ("ray_map_encoder", r"^view_transformer\.ray_map_encoder"),
    ("view self_attn", r"^view_transformer\.transformer\.layers\.\d+\.self_attn\."),
    ("view cross_attn q", r"^view_transformer\.transformer\.layers\.\d+\.multihead_attn\.q_proj"),
    ("view cross_attn k/v", r"^view_transformer\.transformer\.layers\.\d+\.multihead_attn\.[kv]_proj"),
    ("view cross_attn out", r"^view_transformer\.transformer\.layers\.\d+\.multihead_attn\.out_proj"),
    ("view ffn", r"^view_transformer\.transformer\.layers\.\d+\.ffn\."),
    ("view norms", r"^view_transformer\.transformer\.layers\.\d+\..*norm"),
    ("DPT decoder (out_dpt convs)", r"^view_transformer\.out_dpt\."),
    ("other", r"."),
]
RANKS = (1, 2, 4, 8, 16, 32, 64)


def family(name: str) -> str:
    for fam, pat in FAMILIES:
        if re.search(pat, name):
            return fam
    return "other"


def load_sd(path: str) -> dict[str, torch.Tensor]:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    return {k: v.float() for k, v in sd.items()}


def analyse(seed: dict, ft: dict) -> tuple[dict, list[dict]]:
    fam_num, fam_den, fam_cnt = defaultdict(float), defaultdict(float), defaultdict(int)
    mats = []
    for k, w0 in seed.items():
        d = ft[k] - w0
        n2, d2 = (w0 ** 2).sum().item(), (d ** 2).sum().item()
        f = family(k)
        fam_num[f] += d2
        fam_den[f] += n2
        fam_cnt[f] += w0.numel()
        if w0.ndim == 2 and min(w0.shape) >= 8:
            s = torch.linalg.svdvals(d)
            e = (s ** 2).cumsum(0) / (s ** 2).sum()
            mats.append({
                "name": k, "shape": list(w0.shape), "rel": (d2 / n2) ** 0.5,
                "r90": int((e < 0.90).sum().item()) + 1, "r99": int((e < 0.99).sum().item()) + 1,
                "top_frac": {r: float(e[min(r, len(e)) - 1]) for r in RANKS},
                "family": f,
            })
    fams = {f: {"rel": (fam_num[f] / fam_den[f]) ** 0.5, "params": fam_cnt[f],
                "delta_energy": fam_num[f]} for f in fam_num}
    return fams, mats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seed")
    ap.add_argument("fts", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seed = load_sd(a.seed)
    out = {}
    for ft_path in a.fts:
        ft = load_sd(ft_path)
        fams, mats = analyse(seed, ft)
        out[ft_path] = {"families": fams, "matrices": mats}
        total_e = sum(v["delta_energy"] for v in fams.values())
        print(f"\n=== {ft_path} vs {a.seed} ===")
        print(f"{'family':32s} {'params':>12s} {'rel ||dW||/||W0||':>18s} {'share of dW energy':>19s}")
        for f, v in sorted(fams.items(), key=lambda kv: -kv[1]["rel"]):
            print(f"{f:32s} {v['params']:12,d} {v['rel']:18.4f} {100*v['delta_energy']/total_e:18.1f}%")
        print(f"\nper-family delta spectrum (median over matrices): r90 / r99 = #components for 90% / 99% "
              f"of delta energy; top-r = energy fraction captured by rank r")
        byfam = defaultdict(list)
        for m in mats:
            byfam[m["family"]].append(m)
        print(f"{'family':32s} {'n':>3s} {'min(dim)':>8s} {'r90':>5s} {'r99':>5s} " +
              " ".join(f"top{r:>3d}" for r in RANKS))
        for f, ms in byfam.items():
            med = lambda key: sorted(x[key] for x in ms)[len(ms) // 2]
            tops = " ".join(f"{sorted(x['top_frac'][r] for x in ms)[len(ms)//2]:6.2f}" for r in RANKS)
            print(f"{f:32s} {len(ms):3d} {min(min(x['shape']) for x in ms):8d} {med('r90'):5d} {med('r99'):5d} {tops}")
        print("\ntop-10 individual matrices by relative change:")
        for m in sorted(mats, key=lambda x: -x["rel"])[:10]:
            print(f"  {m['name']:70s} rel {m['rel']:.4f}  r90 {m['r90']:4d}  top4 {m['top_frac'][4]:.2f}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, separators=(",", ":"))  # compact: ~10k lines pretty-printed
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
