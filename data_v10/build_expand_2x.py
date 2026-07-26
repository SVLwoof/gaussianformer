"""2x expansion object selection for V10 (data_v11-scale, kept in data_v10 dirs).

Extends the CURRENT data_v10 selection: excludes every uid already in
data_v10/object_list_{train,val}.json, then draws an equal number of fresh objects under the
same quality filter, continuing scene_idx after the current max. Doubles BOTH splits.

Writes:
  - data_v10/object_list_{train,val}_new2x.json   (NEW objects only -> feed process/recovery)
  - data_v10/object_list_{train,val}.json         (combined master, existing first then new)
The old master lists are backed up to *_pre2x.json first.

  uv run --frozen python -m data_v10.build_expand_2x
"""
from __future__ import annotations
import csv, json, random
from collections import defaultdict
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "ShapeSplats/Objaverse_Splats"
MIN_PSNR, MAX_LPIPS, NUM_GS = 32.0, 0.06, 50000
SEED = 2  # distinct from build_additive's SEED=1 so we don't re-draw the same shuffle order
OUT = Path("data_v10")


def chunk_of(path: str) -> str | None:
    parts = path.split("/")
    return parts[1] if len(parts) >= 3 and parts[0] == "glbs" else None


def main():
    rng = random.Random(SEED)
    cur_train = json.loads((OUT / "object_list_train.json").read_text())
    cur_val = json.loads((OUT / "object_list_val.json").read_text())
    used = {o["uid"] for o in cur_train} | {o["uid"] for o in cur_val}
    need_train, need_val = len(cur_train), len(cur_val)  # double each split
    need_total = need_train + need_val
    print(f"current: train {len(cur_train)}, val {len(cur_val)}, used uids {len(used)}")
    print(f"need new: train {need_train}, val {need_val} (total {need_total})")

    csv_path = hf_hub_download(REPO_ID, "completed_3dgs_metadata.csv", repo_type="dataset")
    by_chunk: dict[str, list[dict]] = defaultdict(list)
    kept = 0
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            try:
                psnr, lp, ngs = float(r["psnr"]), float(r["lpips"]), int(r["num_GS"])
            except (ValueError, KeyError):
                continue
            if psnr < MIN_PSNR or lp > MAX_LPIPS or ngs != NUM_GS:
                continue
            uid = r["object_name"]
            if uid in used:
                continue
            ch = chunk_of(r.get("path", ""))
            if ch is None:
                continue
            by_chunk[ch].append({"chunk": ch, "uid": uid, "psnr": psnr,
                                 "lpips": lp, "caption": r.get("caption", "")[:500]})
            kept += 1
    print(f"unused objects passing filter: {kept:,} across {len(by_chunk)} chunks")

    # Accumulate from the richest fresh chunks until we have enough (minimises # chunk downloads).
    pool = []
    for ch in sorted(by_chunk, key=lambda k: -len(by_chunk[k])):
        pool.extend(by_chunk[ch])
        if len(pool) >= need_total:
            break
    if len(pool) < need_total:
        raise RuntimeError(f"pool {len(pool)} < need {need_total}; relax filter or add chunks")
    rng.shuffle(pool)
    new_val = pool[:need_val]
    new_train = pool[need_val:need_val + need_train]

    nt0 = max(o["scene_idx"] for o in cur_train) + 1
    nv0 = max(o["scene_idx"] for o in cur_val) + 1
    for i, o in enumerate(new_train):
        o["scene_idx"] = nt0 + i
    for i, o in enumerate(new_val):
        o["scene_idx"] = nv0 + i
    print(f"new scene_idx: train {nt0}..{nt0+len(new_train)-1}, val {nv0}..{nv0+len(new_val)-1}")

    # back up current masters, then write new-only + combined masters
    for split, cur, new in (("train", cur_train, new_train), ("val", cur_val, new_val)):
        (OUT / f"object_list_{split}_pre2x.json").write_text(json.dumps(cur, indent=1))
        (OUT / f"object_list_{split}_new2x.json").write_text(json.dumps(new, indent=1))
        (OUT / f"object_list_{split}.json").write_text(json.dumps(cur + new, indent=1))

    n_chunks = len({o["chunk"] for o in new_train + new_val})
    print(f"\ncombined master: train {len(cur_train)+len(new_train)}, val {len(cur_val)+len(new_val)}")
    print(f"new objects span {n_chunks} chunks (~{3.3*n_chunks:.0f} GB transient chunk download)")
    print("wrote object_list_{train,val}_new2x.json + updated masters (+ *_pre2x.json backups)")


if __name__ == "__main__":
    main()
