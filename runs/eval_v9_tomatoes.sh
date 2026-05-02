#!/bin/zsh
#SBATCH --time=1:00:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --output=runs/eval_v9_%j.out
#SBATCH --job-name=v9_eval
#SBATCH --gres=gg:g4:1

# Tomatoes external-scene eval. Submit with EP=NN env var:
#   sbatch --export=ALL,EP=40 runs/eval_v9_tomatoes.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd /cs/labs/tomhope/shahaf_levy/gaussianformer
export PYTHONUNBUFFERED=1

if [ -z "$EP" ]; then
  echo "ERROR: set EP=<epoch number>" >&2
  exit 1
fi

CKPT="checkpoints_v9/phase2_epoch_${EP}.pt"
TAG="v9_ep${EP}"
LOG="runs/eval_v9_ep${EP}.log"

echo "== eval $CKPT -> $TAG  (log: $LOG)"
uv run python -m data_external.run_tomatoes --checkpoint "$CKPT" --tag "$TAG" 2>&1 | tee "$LOG"
