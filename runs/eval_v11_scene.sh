#!/bin/zsh
#SBATCH --time=1:00:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --output=runs/eval_v11_%j.out
#SBATCH --job-name=v11_eval
#SBATCH --gres=gg:g4:1

# Per-scene, per-checkpoint eval. Submit with SCENE and EP env vars (CKPTDIR and
# PE_TYPE default to V12's concat encoder; override for V11 / older runs):
#   sbatch --export=ALL,SCENE=tomatoes,EP=75 runs/eval_v11_scene.sh                  # V12 default
#   sbatch --export=ALL,SCENE=house,EP=75,CKPTDIR=checkpoints_v12 runs/eval_v11_scene.sh
#   sbatch --export=ALL,SCENE=dragon,EP=20,CKPTDIR=checkpoints_v11,PE_TYPE=nerf_perfield runs/eval_v11_scene.sh
#
# SCENE must be one of: tomatoes, house, dragon, cartoon.
# For Objaverse scenes (house/dragon/cartoon), prep_eval_v11_scenes.sh must have
# run once first to populate data_external/<scene>/raw.ply and renders/gsplat_full/.

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1

if [ -z "$SCENE" ] || [ -z "$EP" ]; then
  echo "ERROR: set SCENE=<name> and EP=<epoch number>" >&2
  exit 1
fi

PHASE="${PHASE:-phase2}"
CKPTDIR="${CKPTDIR:-checkpoints_v12}"
PE_TYPE="${PE_TYPE:-nerf}"
CKPT="${CKPTDIR}/${PHASE}_epoch_${EP}.pt"
TAG="${CKPTDIR##*/}_${PHASE}_ep${EP}"
LABEL="${LABEL:-${CKPTDIR#*checkpoints_} ep${EP}}"
LOG="runs/eval_${CKPTDIR##*/}_${SCENE}_${PHASE}_ep${EP}.log"

if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  exit 1
fi

echo "== eval $CKPT on scene=$SCENE -> $TAG  (pe_type=$PE_TYPE, label='$LABEL', log: $LOG)"
uv run --frozen python -m data_external.run_scene \
  --scene "$SCENE" \
  --checkpoint "$CKPT" \
  --tag "$TAG" \
  --pe_type "$PE_TYPE" \
  --scale_pe_num_freqs 6 \
  --label "$LABEL" 2>&1 | tee "$LOG"
