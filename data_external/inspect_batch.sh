#!/bin/zsh
#SBATCH --time=0:30:00
#SBATCH -c 4
#SBATCH --mem=24GB
#SBATCH --output=runs/inspect_%j.out
#SBATCH --job-name=inspect
#SBATCH --gres=gg:g4:1
#SBATCH --account=sagieb
#SBATCH --killable
#SBATCH --requeue
# Orbit-render vet of candidate splats (PLY env var, OUT env var).
source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_ext_sm${ARCH:-unknown}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
PYTHONPATH=. uv run --no-sync python data_external/inspect_splat.py --ply $PLY --out $OUT
