"""Split the combined train object list into K chunk-disjoint shards (greedy-balanced by
object count) so they can be processed by parallel jobs without sharing chunks.
Val (small) runs as a single separate job.

  uv run --frozen python -m data_v10.make_shards --k 5
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--list", type=Path, default=Path("data_v10/object_list_train.json"))
    ap.add_argument("--out", type=Path, default=Path("data_v10/shards"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    objs = json.loads(args.list.read_text())
    by_chunk = defaultdict(list)
    for o in objs:
        by_chunk[o["chunk"]].append(o)

    # greedy: assign each chunk (largest first) to the currently-lightest shard
    shards = [[] for _ in range(args.k)]
    for ch in sorted(by_chunk, key=lambda c: -len(by_chunk[c])):
        i = min(range(args.k), key=lambda j: len(shards[j]))
        shards[i].extend(by_chunk[ch])

    for i, sh in enumerate(shards):
        (args.out / f"train_{i}.json").write_text(json.dumps(sh, indent=1))
        nch = len({o["chunk"] for o in sh})
        print(f"train_{i}.json: {len(sh)} objects, {nch} chunks")
    print(f"total {sum(len(s) for s in shards)} objects across {args.k} shards")


if __name__ == "__main__":
    main()
