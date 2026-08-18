#!/bin/zsh
#SBATCH --time=2:00:00
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH --output=runs/codec_so_eval_%j.out
#SBATCH --job-name=csoeval
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08,khan-01,khan-02

# Scale-out codec verdict for one (SCENE, CKPT). TAG optional.
#   sbatch --export=SCENE=scene_0959,CKPT=checkpoints_codec_so_scene_0959/phase2_epoch_27.pt -A sagieb data_v10/codec_scaleout_eval.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_ext_sm${ARCH:-unknown}"
mkdir -p "$TORCH_EXTENSIONS_DIR"

: ${SCENE:?} ${CKPT:?}
PYTHONPATH=. SCENE=$SCENE CKPT=$CKPT TAG=${TAG:-} uv run --no-sync python data_external/codec_scaleout_eval.py
