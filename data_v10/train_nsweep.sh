#!/bin/zsh
#SBATCH --time=24:00:00
#SBATCH -c 16
#SBATCH --mem=64GB
#SBATCH --output=runs/nsweep_%j.out
#SBATCH --job-name=nsweep
#SBATCH --gres=gg:g4:4
#SBATCH --account=sagieb
#SBATCH --killable
#SBATCH --requeue
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08

# N-SWEEP: does the 14 dB gap to the rec-GT ceiling depend on TRAINING-SET SIZE?
#
# We only have two points -- N=1 (June overfit probe: 49-53 dB, fits its object) and N=13,405
# (V17: ~14 dB below ceiling on its OWN training data). Nothing in between. The shape of
# margin-vs-N localises the bottleneck:
#   smooth degradation with N -> capacity; flat then a cliff -> optimisation at scale;
#   already broken at N=100  -> representation/routing, and scale is not the fix.
#
# Design decisions, all aimed at making the three runs comparable to EACH OTHER:
#  * NESTED subsets (n10 subset of n100 subset of n1000) so differences are not object selection.
#  * SAME init for all N: checkpoints_v18_256/phase2_epoch_30.pt -- a converged 256 base we
#    already paid for. Caveat to remember when writing up: that base saw ALL objects, so this
#    measures whether the 512 refine stage can fit N objects, not training from scratch.
#  * EQUAL OPTIMIZER STEPS, not equal epochs. At N=10 an epoch is 10 steps, so epoch counts are
#    meaningless across N; TARGET_STEPS is converted to epochs per N below.
#  * Same effective batch (4 GPUs x bs1) for all three. bs2@512 OOMs on 45 G, and 8-GPU jobs
#    won't schedule right now, so eff batch is 4 here vs V17's 8 -- fine for within-sweep
#    comparison, a footnote when comparing to V17 itself.
#  * save_interval doubles as the VALIDATION interval (train.py:280), so it is set per N to keep
#    validation from dominating at small N; val is a small 100-object dir for the same reason.
#
#   N=10 sbatch --export=N=10 data_v10/train_nsweep.sh

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

N=${N:?N required (10|100|1000)}
TARGET_STEPS=${TARGET_STEPS:-30000}
NGPU=4
EFF_BATCH=$NGPU                      # bs1 per rank
STEPS_PER_EPOCH=$(( N * 4 / EFF_BATCH ))          # views_per_epoch=4
EPOCHS=$(( TARGET_STEPS / STEPS_PER_EPOCH ))
[ $EPOCHS -lt 1 ] && EPOCHS=1
SAVE_INT=$(( EPOCHS / 15 )); [ $SAVE_INT -lt 1 ] && SAVE_INT=1

SEED=checkpoints_v18_256/phase2_epoch_30.pt
SAVE=checkpoints_nsweep_n${N}
echo "N=$N steps/epoch=$STEPS_PER_EPOCH epochs=$EPOCHS (~$TARGET_STEPS steps) save/val every $SAVE_INT"
echo "node=$(hostname) sm_${ARCH} seed=$SEED"

RESUME_ARG=()
p2=( ${SAVE}/phase2_epoch_*.pt(Nom) )
if [ ${#p2} -gt 0 ]; then
  RESUME_ARG=(--resume ${p2[1]}); echo "RESUME from ${p2[1]}"
else
  if [ ! -f $SEED ]; then echo "FATAL: missing $SEED"; exit 1; fi
  RESUME_ARG=(--init_from $SEED); echo "INIT from $SEED"
fi

uv run --no-sync torchrun --standalone --nproc_per_node=$NGPU -m training.train \
  --gaussian_h5_dir data_v10/nsweep/n${N}_h5 \
  --renders_dir     data_v10/nsweep/n${N}_renders \
  --val_h5_dir      data_v10/nsweep/val100_h5 \
  --val_renders_dir data_v10/nsweep/val100_renders \
  --save_dir $SAVE \
  --batch_size 1 --resolution 512 \
  --pe_type rope --augment_rotation --views_per_epoch 4 \
  --phase2_epochs $EPOCHS --phase2_lr 5e-5 \
  --save_interval $SAVE_INT --keep_last_n 2 \
  --log_loss_weight 0.5 --lpips_loss_weight 0.5 \
  --num_workers 8 $RESUME_ARG
