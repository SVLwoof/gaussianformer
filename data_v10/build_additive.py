"""Additive 5x object selection for V10.

Keeps the existing data_v9 selection verbatim (same uids + scene_idx), excludes its uids,
and draws 4x more from fresh chunks under the SAME quality filter, continuing scene_idx.
Writes combined lists to data_v10/object_list_{train,val}.json (existing first, then new).

  uv run --frozen python -m data_v10.build_additive
"""
from __future__ import annotations
import csv, json, random
from collections import defaultdict
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "ShapeSplats/Objaverse_Splats"
MIN_PSNR, MAX_LPIPS, NUM_GS = 32.0, 0.06, 50000
MULT = 5          # target combined = MULT x existing
SEED = 1
OUT = Path("data_v10")


def chunk_of(path: str) -> str | None:
    parts = path.split("/")
    return parts[1] if len(parts) >= 3 and parts[0] == "glbs" else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    ex_train = json.loads(Path("data_v9/object_list_train.json").read_text())
    ex_val = json.loads(Path("data_v9/object_list_val.json").read_text())
    used = {o["uid"] for o in ex_train} | {o["uid"] for o in ex_val}
    need_train = MULT * len(ex_train) - len(ex_train)   # new train entries
    need_val = MULT * len(ex_val) - len(ex_val)
    print(f"existing: train {len(ex_train)}, val {len(ex_val)}, used uids {len(used)}")
    print(f"need new: train {need_train}, val {need_val}")

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

    # Accumulate from the richest fresh chunks until we have enough.
    need_total = need_train + need_val
    pool = []
    for ch in sorted(by_chunk, key=lambda k: -len(by_chunk[k])):
        pool.extend(by_chunk[ch])
        if len(pool) >= need_total:
            break
    if len(pool) < need_total:
        raise RuntimeError(f"pool {len(pool)} < need {need_total}; relax filter or use more chunks")
    rng.shuffle(pool)
    new_val = pool[:need_val]
    new_train = pool[need_val:need_val + need_train]

    # continue scene_idx after the existing max
    nt0 = max(o["scene_idx"] for o in ex_train) + 1
    nv0 = max(o["scene_idx"] for o in ex_val) + 1
    for i, o in enumerate(new_train):
        o["scene_idx"] = nt0 + i
    for i, o in enumerate(new_val):
        o["scene_idx"] = nv0 + i

    comb_train = ex_train + new_train
    comb_val = ex_val + new_val
    (OUT / "object_list_train.json").write_text(json.dumps(comb_train, indent=1))
    (OUT / "object_list_val.json").write_text(json.dumps(comb_val, indent=1))
    n_new_chunks = len({o["chunk"] for o in new_train + new_val})
    print(f"\ncombined: train {len(comb_train)}, val {len(comb_val)}")
    print(f"new objects span {n_new_chunks} fresh chunks (~{3.3*n_new_chunks:.0f} GB transient download)")
    print(f"wrote {OUT}/object_list_train.json + object_list_val.json")


if __name__ == "__main__":
    main()
