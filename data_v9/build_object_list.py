"""V9 Phase A: select Objaverse_Splats objects for train + val splits.

Reads the dataset's `completed_3dgs_metadata.csv`, applies a quality filter
(PSNR/LPIPS/num_GS), and samples disjoint train / val sets across multiple
chunks for diversity. Writes JSON lists that `process_objaverse.py` consumes.

Output JSON record format (one per object):
  {"chunk": "000-000", "uid": "abc123", "psnr": 41.5,
   "lpips": 0.024, "caption": "...", "scene_idx": 0}

`scene_idx` is the destination index inside the split (becomes scene_NNNN.h5).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download


CSV_NAME = "completed_3dgs_metadata.csv"
REPO_ID = "ShapeSplats/Objaverse_Splats"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path, default=Path("data_v9"))
    p.add_argument("--n_train", type=int, default=3000)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--min_psnr", type=float, default=32.0)
    p.add_argument("--max_lpips", type=float, default=0.06)
    p.add_argument("--num_gs", type=int, default=50000,
                   help="Required num_GS for the entry (filters truncated fits).")
    p.add_argument("--n_chunks", type=int, default=12,
                   help="How many chunks to draw from. More chunks = more diversity but more "
                   "downloads (~3.3 GB each).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_metadata() -> list[dict]:
    csv_path = hf_hub_download(REPO_ID, CSV_NAME, repo_type="dataset")
    print(f"Reading {csv_path}")
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    print(f"  total entries: {len(rows):,}")
    return rows


def chunk_of(path: str) -> str | None:
    # Path field looks like "glbs/000-000/abc123.glb" -- chunk is the 2nd segment.
    parts = path.split("/")
    if len(parts) >= 3 and parts[0] == "glbs":
        return parts[1]
    return None


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    rows = load_metadata()

    by_chunk: dict[str, list[dict]] = defaultdict(list)
    n_kept = 0
    for r in rows:
        try:
            psnr = float(r["psnr"])
            lpips_v = float(r["lpips"])
            num_gs = int(r["num_GS"])
        except (ValueError, KeyError):
            continue
        if psnr < args.min_psnr or lpips_v > args.max_lpips or num_gs != args.num_gs:
            continue
        ch = chunk_of(r.get("path", ""))
        if ch is None:
            continue
        by_chunk[ch].append({
            "chunk": ch,
            "uid": r["object_name"],
            "psnr": psnr,
            "lpips": lpips_v,
            "caption": r.get("caption", "")[:500],
        })
        n_kept += 1

    print(f"  passed quality filter (PSNR>={args.min_psnr}, LPIPS<={args.max_lpips}, "
          f"num_GS={args.num_gs}): {n_kept:,}")
    print(f"  spread across {len(by_chunk)} chunks "
          f"(top 5: {sorted([(len(v), k) for k, v in by_chunk.items()], reverse=True)[:5]})")

    # Pick the top n_chunks by available passing objects (more usable per download).
    chunks_sorted = sorted(by_chunk.keys(), key=lambda k: -len(by_chunk[k]))
    chosen_chunks = chunks_sorted[: args.n_chunks]
    print(f"\nSelected {len(chosen_chunks)} chunks: {chosen_chunks}")
    pool = []
    for ch in chosen_chunks:
        pool.extend(by_chunk[ch])
    print(f"  pool size: {len(pool):,}")

    needed = args.n_train + args.n_val
    if len(pool) < needed:
        raise RuntimeError(
            f"Pool has {len(pool):,} objects, need {needed:,}. Increase --n_chunks "
            f"or relax --min_psnr / --max_lpips."
        )

    # Stratified sample: take roughly proportional from each chunk to keep diversity.
    rng.shuffle(pool)
    train = pool[: args.n_train]
    val = pool[args.n_train : args.n_train + args.n_val]

    # Assign scene indices.
    for i, r in enumerate(train):
        r["scene_idx"] = i
    for i, r in enumerate(val):
        r["scene_idx"] = i

    train_path = args.out_dir / "object_list_train.json"
    val_path = args.out_dir / "object_list_val.json"
    train_path.write_text(json.dumps(train, indent=2))
    val_path.write_text(json.dumps(val, indent=2))
    print(f"\nWrote {train_path} ({len(train)} entries)")
    print(f"Wrote {val_path} ({len(val)} entries)")

    # Per-chunk distribution.
    train_chunks: dict[str, int] = defaultdict(int)
    val_chunks: dict[str, int] = defaultdict(int)
    for r in train:
        train_chunks[r["chunk"]] += 1
    for r in val:
        val_chunks[r["chunk"]] += 1
    print("\nPer-chunk distribution (train | val):")
    for ch in chosen_chunks:
        print(f"  {ch}: {train_chunks[ch]:>4d}  |  {val_chunks[ch]:>3d}")


if __name__ == "__main__":
    main()
