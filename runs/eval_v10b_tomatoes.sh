#!/bin/zsh
#SBATCH --time=1:00:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --output=runs/eval_v10b_%j.out
#SBATCH --job-name=v10b_eval
#SBATCH --gres=gg:g4:1

# Tomatoes external-scene eval for V10b (LPIPS fine-tune, bs=4 / 4 GPU restart).
# Submit with EP=NN env var:
#   sbatch --export=ALL,EP=10 runs/eval_v10b_tomatoes.sh

source "${LMOD_INIT:-/etc/profile.d/lmod.sh}"
module load nvidia
module load cuda
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

if [ -z "$EP" ]; then
  echo "ERROR: set EP=<epoch number>" >&2
  exit 1
fi

CKPT="checkpoints_v10b/phase2_epoch_${EP}.pt"
TAG="v10b_ep${EP}"
LOG="runs/eval_v10b_ep${EP}.log"

echo "== eval $CKPT -> $TAG  (log: $LOG)"
uv run python -m data_external.run_tomatoes --checkpoint "$CKPT" --tag "$TAG" 2>&1 | tee "$LOG"
