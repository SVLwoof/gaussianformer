"""Aggregate eval_val_full.py JSON outputs into a comparison table.

Usage:
  uv run python -m eval_compare eval_results/*.json
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", type=Path)
    args = ap.parse_args()

    rows = []
    for p in args.results:
        d = json.loads(p.read_text())
        s = d["summary"]
        rows.append({
            "tag": p.stem,
            "ckpt": Path(d["checkpoint"]).name,
            "pe": d["pe_type"],
            "h5_dir": d["h5_dir"],
            "n_scenes": s["n_scenes"],
            "psnr_mean": s["psnr_mean"], "psnr_median": s["psnr_median"], "psnr_std": s["psnr_std"],
            "lpips_mean": s["lpips_mean"], "lpips_median": s["lpips_median"], "lpips_std": s["lpips_std"],
            "wall_s": d.get("wall_time_s", 0),
        })
    rows.sort(key=lambda r: (-r["psnr_mean"]))

    w = {"tag": max(20, max(len(r["tag"]) for r in rows)),
         "ckpt": max(20, max(len(r["ckpt"]) for r in rows)),
         "h5_dir": max(15, max(len(r["h5_dir"]) for r in rows))}
    hdr = (f"{'tag':<{w['tag']}}  {'ckpt':<{w['ckpt']}}  {'pe':<6}  "
           f"{'h5_dir':<{w['h5_dir']}}  {'n':>4}  "
           f"{'PSNR(mean/med/std)':>26}  {'LPIPS(mean/med/std)':>24}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['tag']:<{w['tag']}}  {r['ckpt']:<{w['ckpt']}}  {r['pe']:<6}  "
              f"{r['h5_dir']:<{w['h5_dir']}}  {r['n_scenes']:>4}  "
              f"{r['psnr_mean']:>8.3f}/{r['psnr_median']:>6.2f}/{r['psnr_std']:>5.2f}  "
              f"{r['lpips_mean']:>7.4f}/{r['lpips_median']:>6.4f}/{r['lpips_std']:>5.4f}")


if __name__ == "__main__":
    main()
