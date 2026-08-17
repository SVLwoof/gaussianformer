#!/bin/zsh
#SBATCH --time=2:00:00
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH --output=runs/codec4_eval_%j.out
#SBATCH --job-name=c4eval
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08,khan-01,khan-02

# Codec v4 verdict eval. CKPT/TAG env-overridable (defaults: cycle-2 final, tag codec4).
#   sbatch --export=NONE -A sagieb data_v10/tomato_codec4_eval.sh
#   sbatch --export=CKPT=checkpoints_tomato_codec4/phase2_epoch_27.pt,TAG=codec4_c1 -A sagieb ...

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_ext_sm${ARCH:-unknown}"
mkdir -p "$TORCH_EXTENSIONS_DIR"

PYTHONPATH=. uv run --no-sync python data_external/tomatoes_codec4_eval.py
