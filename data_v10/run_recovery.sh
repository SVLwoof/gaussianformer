#!/bin/zsh
#SBATCH --time=8:00:00
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH --job-name=v10_rec
#SBATCH --output=runs/v10_rec_%x_%j.out
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue

# Prune-and-recovery validation/processing. Real zsh sbatch (not --wrap) for gsplat CUDA.
# Pass SCENES (comma list), ITERS, OUT, SAVE (optional h5 dir) via --export.
source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

: ${ITERS:=1500}
: ${OUT:=data_v10/recovery_eval}
: ${SPLIT:=train}
ARGS=(--split $SPLIT --iters $ITERS --out $OUT)
[ -n "$SCENES_FILE" ] && ARGS+=(--scenes_file "$SCENES_FILE")
[ -n "$SCENES" ] && ARGS+=(--scenes "$SCENES")
[ -n "$SAVE" ] && ARGS+=(--save_h5_dir "$SAVE")
[ -n "$COMPARE" ] && ARGS+=(--compare_dir "$COMPARE")
[ -n "$SAVE_ONLY" ] && ARGS+=(--save_only)

uv run --frozen python -m data_v10.prune_recovery $ARGS
