#!/bin/zsh
#SBATCH --time=72:00:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --job-name=overfit
#SBATCH --output=runs/overfit_%x_%j.out
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue

# Single-object OVERFIT capacity probe. Trains the model exclusively on ONE object's 14
# views (augmentation OFF -- we want to memorise exact poses) to find the architecture's
# representational ceiling, decoupled from data/generalisation. See README.md.
#
# Parameterised by two env vars (pass via --export):
#   OBJ  = boxes | tomatoes
#   INIT = base    -> RenderFormer transfer, Phase 1 (encoder warmup) + Phase 2 (joint)
#          v14best -> warm-start weights from V14best; --init_from auto-skips Phase 1.
#
# Submit all four simultaneously (killable bypasses the lab GPU quota):
#   for OBJ in boxes tomatoes; do for INIT in base v14best; do
#     sbatch --job-name=ovf_${OBJ}_${INIT} \
#            --export=ALL,OBJ=$OBJ,INIT=$INIT experiments/overfit/run_overfit.sh
#   done; done

set -e
source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: ${OBJ:?set OBJ=boxes|tomatoes}
: ${INIT:?set INIT=base|v14best}
PHASE2_LR=${PHASE2_LR:-5e-5}   # override to escape the joint-unfreeze plateau (e.g. 2e-4)
TAG=${TAG:-}                   # optional save_dir/job suffix for rescue/variant runs

case "$OBJ" in
  boxes)    H5DIR=experiments/overfit/data/boxes/h5s;    RENDERS=data_v9/renders ;;
  tomatoes) H5DIR=experiments/overfit/data/tomatoes/h5s; RENDERS=experiments/overfit/data/tomatoes/renders ;;
  *) echo "bad OBJ=$OBJ"; exit 1 ;;
esac
SAVEDIR=experiments/overfit/ckpt/${OBJ}_${INIT}${TAG:+_$TAG}
mkdir -p $SAVEDIR
echo "OBJ=$OBJ INIT=$INIT PHASE2_LR=$PHASE2_LR SAVEDIR=$SAVEDIR"

# Resume-aware seed (zsh (Nom) arrays -- newest-first, null-glob; do NOT word-split a scalar).
# Priority: own checkpoints (--resume) > v14best warm-start (--init_from) > RF base (fresh).
files=( $SAVEDIR/phase2_epoch_*.pt(Nom) $SAVEDIR/phase1_epoch_*.pt(Nom) )
if [ ${#files} -gt 0 ]; then
  SEED=(--resume ${files[1]}); echo "SEED: resume ${files[1]}"
elif [ "$INIT" = "v14best" ]; then
  SEED=(--init_from checkpoints_v14auglp10/phase2_epoch_10.pt); echo "SEED: warm-start V14best (Phase 1 skipped)"
else
  SEED=(); echo "SEED: RenderFormer base (fresh, Phase 1 + Phase 2)"
fi

# Single GPU: 14 samples -> multi-GPU/large-batch is meaningless. bs=1 = most update steps.
# Phase 1 = 50 ep encoder warmup (avoids the 0.0138 plateau on RF-base; ignored on --init_from).
# Phase 2 = 1500 ep joint to drive the overfit to its ceiling (~21k steps over 14 samples).
# No val (skipped) -> overfit tracked via train loss + offline eval_overfit.py on checkpoints.
uv run --frozen torchrun --standalone --nproc_per_node=1 -m training.train \
  --gaussian_h5_dir $H5DIR --renders_dir $RENDERS \
  --save_dir $SAVEDIR \
  --batch_size 1 --resolution 512 \
  --pe_type rope \
  --phase1_epochs 50 --phase1_lr 1e-3 \
  --phase2_epochs 1500 --phase2_lr $PHASE2_LR \
  --save_interval 250 --keep_last_n 6 \
  --log_loss_weight 1.0 --lpips_loss_weight 0.0 \
  --num_workers 4 $SEED
