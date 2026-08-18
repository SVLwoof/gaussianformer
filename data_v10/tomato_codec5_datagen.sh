#!/bin/zsh
#SBATCH --time=2:00:00
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH --output=runs/codec5_datagen_%j.out
#SBATCH --job-name=c5data
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08,khan-01,khan-02

# Codec v5 data: symlink codec4's 4500 renders + rasterize 4500 new grazing-elevation views.
#   sbatch --export=NONE -A sagieb data_v10/tomato_codec5_datagen.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_ext_sm${ARCH:-unknown}"
mkdir -p "$TORCH_EXTENSIONS_DIR"

PYTHONPATH=. uv run --no-sync python data_external/tomatoes_codec5_data.py
