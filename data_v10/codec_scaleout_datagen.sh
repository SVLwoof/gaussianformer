#!/bin/zsh
#SBATCH --time=3:00:00
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH --output=runs/codec_so_datagen_%j.out
#SBATCH --job-name=csodata
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08,khan-01,khan-02

# One scale-out object's codec training data (9000 train + 40 eval renders from the
# full splat). Resume-safe; killable. First job runs ALONE to pre-warm gsplat JIT on
# any cold arch, the other nine chain behind it (see [[feedback_gsplat_jit_prewarm]]).
#   sbatch --export=SCENE=scene_0959 -A sagieb data_v10/codec_scaleout_datagen.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_ext_sm${ARCH:-unknown}"
mkdir -p "$TORCH_EXTENSIONS_DIR"

SCENE=${SCENE:?SCENE required, e.g. scene_0959}
PYTHONPATH=. uv run --no-sync python data_v10/codec_scaleout_datagen.py --scene $SCENE
