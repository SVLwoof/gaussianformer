#!/bin/zsh
#SBATCH --time=24:00:00
#SBATCH -c 16
#SBATCH --mem=64GB
#SBATCH --output=runs/nsweep_ctrl_%j.out
#SBATCH --job-name=nswctrl
#SBATCH --gres=gg:g4:8
#SBATCH --account=sagieb
#SBATCH --killable
#SBATCH --requeue
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08

# N=10 MEMORIZATION CONTROL -- can the model fit 10 objects when NOTHING fights memorization?
#
# The N-sweep's headline ("cannot fit even 10 objects to the ceiling; 7.6 dB short") has two
# recipe-side confounds the June N=1 probe (which DID reach the ceiling) did not share:
#   1. --augment_rotation was ON: the sweep model had to learn a rotation-EQUIVARIANT rendering,
#      not memorize 10 fixed objects. The N=1 probe ran aug OFF.
#   2. weight_decay 0.01 on every parameter: AdamW decay is a memorization suppressor, and at
#      3,000 epochs it has plenty of time to bite.
# This run removes both (aug off, wd 0) on the SAME nested n10 subset, same init, same step
# budget. Binary readout:
#   near-ceiling  -> the "floor" is partly the recipe fighting memorization; re-frame.
#   still ~7 dB short -> recipe exonerated; the floor is architectural (capacity vs routing next).
#
# 8 GPUs x bs1 (khan-02 idle at submit) => effective batch 8 vs the sweep's 4 -- acceptable for a
# binary readout, footnote for exact-number comparisons. steps/epoch = 10*4/8 = 5.
#
#   sbatch data_v10/train_nsweep_ctrl.sh

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

NGPU=8
TARGET_STEPS=${TARGET_STEPS:-30000}
STEPS_PER_EPOCH=$(( 10 * 4 / NGPU ))
EPOCHS=$(( TARGET_STEPS / STEPS_PER_EPOCH ))
SAVE_INT=$(( EPOCHS / 15 )); [ $SAVE_INT -lt 1 ] && SAVE_INT=1

SEED=checkpoints_v18_256/phase2_epoch_30.pt
SAVE=checkpoints_nsweep_n10_ctrl
echo "CTRL: N=10 aug=OFF wd=0 | $NGPU GPUs, $EPOCHS epochs (~$TARGET_STEPS steps), save/val every $SAVE_INT"
echo "node=$(hostname) sm_${ARCH}"

RESUME_ARG=()
p2=( ${SAVE}/phase2_epoch_*.pt(Nom) )
if [ ${#p2} -gt 0 ]; then
  RESUME_ARG=(--resume ${p2[1]}); echo "RESUME from ${p2[1]}"
else
  [ -f $SEED ] || { echo "FATAL: missing $SEED"; exit 1; }
  RESUME_ARG=(--init_from $SEED); echo "INIT from $SEED"
fi

uv run --no-sync torchrun --standalone --nproc_per_node=$NGPU -m training.train \
  --gaussian_h5_dir data_v10/nsweep/n10_h5 \
  --renders_dir     data_v10/nsweep/n10_renders \
  --val_h5_dir      data_v10/nsweep/val100_h5 \
  --val_renders_dir data_v10/nsweep/val100_renders \
  --save_dir $SAVE \
  --batch_size 1 --resolution 512 \
  --pe_type rope --views_per_epoch 4 \
  --weight_decay 0 \
  --phase2_epochs $EPOCHS --phase2_lr 5e-5 \
  --save_interval $SAVE_INT --keep_last_n 2 \
  --log_loss_weight 0.5 --lpips_loss_weight 0.5 \
  --num_workers 4 $RESUME_ARG
