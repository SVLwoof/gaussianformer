#!/bin/zsh
#SBATCH --time=1:30:00
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH --output=runs/lowN_%A_%a.out
#SBATCH --job-name=lowN
#SBATCH --gres=gg:g4:1
#SBATCH --requeue
#SBATCH --array=0-9%3
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08,khan-01,khan-02

# Low-N codec inputs (5k / 2k) for the crossing objects + tomatoes + octopus (2026-09-08).
#   sbatch --killable --account=killable-cs data_v10/codec_lowN_datagen.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_ext_sm${ARCH:-unknown}"
mkdir -p "$TORCH_EXTENSIONS_DIR"

SCENES=(gopro scene_1423 scene_0223 octopus tomatoes)
NS=(5000 2000)
i=${SLURM_ARRAY_TASK_ID:-0}
SCENE=${SCENES[$((i / 2 + 1))]}
N=${NS[$((i % 2 + 1))]}
echo "lowN task $i: $SCENE N=$N on $(hostname) sm_$ARCH"
PYTHONPATH=. uv run --no-sync python data_v10/codec_lowN_datagen.py --scene $SCENE --n $N
