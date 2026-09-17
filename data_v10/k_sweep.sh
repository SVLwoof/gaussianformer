#!/bin/zsh
#SBATCH --time=6:00:00
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH --output=runs/ksweep_%j.out
#SBATCH --job-name=ksweep
#SBATCH --gres=gg:g4:1
#SBATCH --exclude=cyril-01,firefoot-01,khan-01,khan-02,firefoot-13
# Usage: sbatch -A sagieb --export=SCENE=scene_0031,CKPT=checkpoints_codec_so_scene_0031_r2/phase2_epoch_27.pt data_v10/k_sweep.sh
source /etc/profile.d/huji-lmod.sh; module load nvidia; module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_ext_sm${ARCH:-unknown}"
: ${SCENE:?} ${CKPT:?}
PYTHONPATH=. uv run --no-sync python data_v10/k_sweep.py
