#!/bin/zsh
#SBATCH --time=2:00:00
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH --output=runs/codec4_datagen_%j.out
#SBATCH --job-name=c4data
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08,khan-01,khan-02

# Rasterize the codec v4 training set: 4500 randomized views of the full 219k tomato
# splat (free supervision), plus the rec-20k input h5. ~1.5G of PNGs.
#   sbatch --export=NONE -A sagieb data_v10/tomato_codec4_datagen.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_ext_sm${ARCH:-unknown}"
mkdir -p "$TORCH_EXTENSIONS_DIR"

PYTHONPATH=. uv run --no-sync python data_external/tomatoes_codec4_data.py
