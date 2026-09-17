#!/bin/zsh
#SBATCH --time=6:00:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --output=runs/augradius_%j.out
#SBATCH --job-name=augradius
#SBATCH --gres=gg:g4:1
#SBATCH --requeue
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08,khan-01,khan-02
# Camera-distance augmentation data (plan 2026-09-17 #3), one GPU, resume-safe:
#   1. rebuild the FULL 50k splats of the N=10 train objects (deleted 2026-08-17) from HF
#   2. n10_r3: the same pruned inputs with 14 + 14 (r 1.15) + 14 (r 2.45) views, GT from the full splat
#   3. heldout300 GT renders at r 1.15 (full_h5s_val exists) for the close-range readout
#   sbatch --killable --account=killable-cs data_v10/aug_radius_pipeline.sh   # or -A sagieb
source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1 PYTHONPATH=.
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/cs/labs/sagieb/shahaf_levy/tmp/hf
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_ext_sm${ARCH:-unknown}"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$HF_HOME"
SCENES=$(python3 -c "import json;print(' '.join(map(str,json.load(open('data_v10/nsweep/n10_scenes.json')))))")
set -e
if [ "$(ls data_v10/rebuild_n10/full_h5s/*.h5 2>/dev/null | wc -l)" -lt 10 ]; then
  uv run --no-sync python -m data_v10.process_full --object_list data_v10/object_list_train.json --split train \
    --out_dir data_v10/rebuild_n10 --scenes ${=SCENES} --rm_zips
fi
[ "$(ls data_v10/rebuild_n10/full_h5s/*.h5 | wc -l)" -ge 10 ] || { echo "FATAL: full splats missing"; exit 1; }
uv run --no-sync python data_v10/multi_radius_datagen.py --full_h5_dir data_v10/rebuild_n10/full_h5s \
  --scenes_file data_v10/nsweep/n10_scenes.json --src_h5_dir data_v10/nsweep/n10_h5 --src_renders_dir data_v10/nsweep/n10_renders \
  --out_h5_dir data_v10/nsweep/n10_r3_h5 --out_renders_dir data_v10/nsweep/n10_r3_renders --radii 1.15 2.45
uv run --no-sync python data_v10/multi_radius_datagen.py --full_h5_dir data_v10/full_h5s_val \
  --scenes_file data_v10/nsweep/heldout300_scenes.json --out_renders_dir data_v10/nsweep/heldout300_r115_renders --eval_radius 1.15
echo AUGRADIUS_DONE
