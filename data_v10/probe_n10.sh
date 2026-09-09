#!/bin/zsh
#SBATCH --time=36:00:00
#SBATCH -c 16
#SBATCH --mem=48GB
#SBATCH --output=runs/probe_%j.out
#SBATCH --job-name=probe
#SBATCH --gres=gg:g4:4
#SBATCH --requeue
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08,khan-01,khan-02,firefoot-13,firefoot-04,firefoot-16

# N=10 architecture probe (2026-09-08). Exactly the N-sweep n10 baseline schedule (seed
# v18_256-ep30, 30k optimizer steps, 4 x bs1 @512, LPIPS 0.5, rot-aug, wd 0.01; recorded
# baseline: train-fit margin 7.57 / heldout300 20.47) with GaussianFormerConfig overrides.
# STAGE_R (steps) prepends a 256px log-L1 recovery stage for changes that alter pretrained
# function (RoPE dim/frequency); zero-init additions (gates, canvas) run without it.
#   sbatch -A sagieb --export=TAG=p3_rope32,MODEL_CFG="rope_dim=32 rope_pos_scale=4",STAGE_R=3000 data_v10/probe_n10.sh
#   eval: sbatch --dependency=afterok:<id> --export=N=10,TAG=probe_<TAG>,CKPT=...,EXTRA="--model_cfg ..." data_v10/run_nsweep_eval.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.local/bin:$PATH"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_ext_sm${ARCH:-unknown}"
mkdir -p "$TORCH_EXTENSIONS_DIR"

TAG=${TAG:?TAG required}
MODEL_CFG=${MODEL_CFG:-}
STAGE_R=${STAGE_R:-0}
N=10; NGPU=4; TARGET_STEPS=${TARGET_STEPS:-30000}
STEPS_PER_EPOCH=$(( N * 4 / NGPU ))            # views_per_epoch=4, bs1
EPOCHS=$(( TARGET_STEPS / STEPS_PER_EPOCH ))
SAVE_INT=$(( EPOCHS / 15 )); [ $SAVE_INT -lt 1 ] && SAVE_INT=1
SEED=${SEED:-checkpoints_v18_256/phase2_epoch_30.pt}
SAVE=checkpoints_probe_${TAG}
# MODEL_CFG arrives ;-separated (sbatch --export cannot carry spaces reliably)
CFG=(); [ -n "$MODEL_CFG" ] && CFG=(--model_cfg ${(s:;:)MODEL_CFG})
# EXTRA_TRAIN: ;-separated extra train.py args (e.g. "--fg_bg_weight;0.05")
XT=(); [ -n "$EXTRA_TRAIN" ] && XT=(${(s:;:)EXTRA_TRAIN})
COMMON=(--gaussian_h5_dir data_v10/nsweep/n${N}_h5 --renders_dir data_v10/nsweep/n${N}_renders
        --val_h5_dir data_v10/nsweep/val100_h5 --val_renders_dir data_v10/nsweep/val100_renders
        --batch_size 1 --pe_type rope --augment_rotation --views_per_epoch 4
        --phase2_lr 5e-5 --keep_last_n 2 --num_workers 8 "${CFG[@]}" "${XT[@]}")
echo "PROBE $TAG cfg=[$MODEL_CFG] extra=[$EXTRA_TRAIN] seed=$SEED stage_r=$STAGE_R steps=$TARGET_STEPS epochs=$EPOCHS node=$(hostname) sm_${ARCH}"
[ -f $SEED ] || { echo "FATAL: missing $SEED"; exit 1; }

# --- Stage R: 256px log-L1 recovery (resume-safe) ---
if [ "$STAGE_R" -gt 0 ]; then
  SAVE_R=${SAVE}_r
  R_EPOCHS=$(( STAGE_R / STEPS_PER_EPOCH ))
  r=( ${SAVE_R}/phase2_epoch_*.pt(Nom) )
  if [ ${#r} -gt 0 ] && [ "${r[1]:t:r}" = "phase2_epoch_${R_EPOCHS}" ]; then
    echo "stage R done: ${r[1]}"
  else
    RES_R=(); [ ${#r} -gt 0 ] && RES_R=(--resume ${r[1]}) || RES_R=(--init_from $SEED)
    uv run --no-sync torchrun --standalone --nproc_per_node=$NGPU -m training.train "${COMMON[@]}" \
      --save_dir $SAVE_R --resolution 256 --phase2_epochs $R_EPOCHS --save_interval $(( R_EPOCHS / 5 + 1 )) \
      --log_loss_weight 1.0 --lpips_loss_weight 0.0 "${RES_R[@]}" || exit 1
  fi
  SEED=${SAVE_R}/phase2_epoch_${R_EPOCHS}.pt
fi

# --- Main: the n10 baseline schedule ---
p2=( ${SAVE}/phase2_epoch_*.pt(Nom) )
if [ ${#p2} -gt 0 ]; then RESUME=(--resume ${p2[1]}); echo "RESUME from ${p2[1]}"
else RESUME=(--init_from $SEED); echo "INIT from $SEED"; fi
uv run --no-sync torchrun --standalone --nproc_per_node=$NGPU -m training.train "${COMMON[@]}" \
  --save_dir $SAVE --resolution 512 --phase2_epochs $EPOCHS --save_interval $SAVE_INT \
  --log_loss_weight 0.5 --lpips_loss_weight 0.5 "${RESUME[@]}"
rc=$?
echo "PROBE_DONE $TAG rc=$rc"
exit $rc
