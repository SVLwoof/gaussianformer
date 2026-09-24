#!/bin/zsh
#SBATCH --time=2:00:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --output=runs/pair_%j.out
#SBATCH --job-name=pair
#SBATCH --gres=gg:g4:1
#SBATCH --requeue
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08,khan-01,khan-02
# Two objects in one scene (render_pair.py). SCENES = two n10 scene ids, ARGS = everything after render_pair.py.
#   sbatch --killable --account=killable-cs --export=PATH,HOME,USER,SCENES="387 1875",CKPT=...,ARGS="--out docs/report/pair" data_v10/render_pair.sh
source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_ext_sm${ARCH:-unknown}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
: ${SCENES:?} ${CKPT:?} ${ARGS:?}
PYTHONPATH=. SCENES=$SCENES CKPT=$CKPT uv run --no-sync python data_v10/render_pair.py "${(Q)${(z)ARGS}[@]}"
