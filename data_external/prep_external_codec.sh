#!/bin/zsh
#SBATCH --time=4:00:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --output=runs/prep_ext_%j.out
#SBATCH --job-name=prepext
#SBATCH --gres=gg:g4:1
#SBATCH --account=sagieb
#SBATCH --killable
#SBATCH --requeue
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08,khan-01,khan-02
# External-scan codec prep (NAME, SEED_BASE, FLIP=1 optional, PROBE=1 optional).
source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_ext_sm${ARCH:-unknown}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
EXTRA=()
[ -n "$FLIP" ] && EXTRA+=(--flip_x)
[ -n "$PROBE" ] && EXTRA+=(--probe)
PYTHONPATH=. uv run --no-sync python data_external/prep_external_codec.py --name $NAME --seed_base $SEED_BASE $EXTRA
