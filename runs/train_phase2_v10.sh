#!/bin/zsh
#SBATCH --time=72:00:00
#SBATCH -c 32
#SBATCH --mem=256GB
#SBATCH --output=runs/train_phase2_v10_%j.out
#SBATCH --job-name=gformer_phase2_v10
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --gres=gg:g4:3

# v10 = LPIPS fine-tune from a V9 checkpoint. V9 plateaued on tomatoes vs-full
# at ~26/27/28/28.6 dB across N=5/10/20/30k since ep30; pure log-HDR L1 has
# saturated at this data scale. V10 keeps the V9 data and adds a perceptual
# gradient (LPIPS-VGG, weight 0.2 on display-space LDR) to break the
# regression-to-the-mean smoothness.
#
# Submit with RESUME_EP env var (which V9 epoch to warm-start from):
#   sbatch --export=ALL,RESUME_EP=60 runs/train_phase2_v10.sh
#
# Save every 2 epochs to catch any early NaN/divergence from the LPIPS term
# and to let us evaluate the perceptual effect after a handful of epochs.
# 30 epochs total: cosine 2e-5 -> 2e-7. Lower start LR than V9's 5e-5 because
# we're continuing from a partly-trained checkpoint, not warm-starting from
# phase1 only.
#
# VRAM note: VGG forward at 512px bs=2 adds ~1.5 GB on top of v9's ~44/48 GB.
# If first epoch OOMs, drop to bs=1 (also handles per-step compute increase).

source "${LMOD_INIT:-/etc/profile.d/lmod.sh}"
module load nvidia
module load cuda
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

if [ -z "$RESUME_EP" ]; then
  echo "ERROR: set RESUME_EP=<V9 epoch to warm-start from, e.g. 60>" >&2
  exit 1
fi

CKPT="checkpoints_v9/phase2_epoch_${RESUME_EP}.pt"
if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  exit 1
fi

echo "== v10 fine-tune: warm-start from $CKPT, LPIPS weight 0.2"

uv run torchrun --standalone --nproc_per_node=3 -m training.train \
  --gaussian_h5_dir data_v9/h5s \
  --renders_dir data_v9/renders \
  --save_dir checkpoints_v10 \
  --batch_size 2 --resolution 512 \
  --skip_phase1 \
  --resume "$CKPT" \
  --phase2_epochs 30 \
  --phase2_lr 2e-5 \
  --save_interval 2 \
  --val_h5_dir data_v9/h5s_val \
  --val_renders_dir data_v9/renders_val \
  --log_loss_weight 1.0 \
  --lpips_loss_weight 0.2 \
  --num_workers 8
