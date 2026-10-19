#!/usr/bin/env bash
# Builds data/val and data/train (about 1.8k and 27k objects) on one GPU. To use more GPUs, run
# `python -m data.prepare ... --task i --n_tasks n` for i = 0..n-1 in parallel, one per GPU.
set -euo pipefail
for split in val train; do
  python -m data.prepare --objects "data/splits/$split.json" --out "data/$split"
done
