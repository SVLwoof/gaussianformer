#!/bin/zsh
#SBATCH --time=72:00:00
#SBATCH -c 32
#SBATCH --mem=256GB
#SBATCH --output=runs/train_phase2_v10b_%j.out
#SBATCH --job-name=gformer_phase2_v10b
#SBATCH --mail-type=END,FAIL,BEGIN
# #SBATCH --mail-user=your-email@example.com
#SBATCH --gres=gg:g4:4

# v10b = v10 restart with bs=4 + 4 GPUs (effective batch 16, up from 6).
# Resume from checkpoints_v10/phase2_epoch_4.pt and run 26 fresh epochs of
# cosine 5e-5 -> 5e-7. LR linear-scaled from v10's 2e-5 by the batch ratio
# (16/6 ~= 2.67) -- ~5.3e-5 rounded to 5e-5.
#
# VRAM budget: v10 bs=2 sat at ~22 GB; bs=4 should land ~40-44 GB on 46 GB.
# If OOM on first epoch, fall back to bs=3 (still a 1.5x bump over v10).
#
# Submit:
#   sbatch runs/train_phase2_v10b.sh

source "${LMOD_INIT:-/etc/profile.d/lmod.sh}"
module load nvidia
module load cuda
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

CKPT="checkpoints_v10/phase2_epoch_4.pt"
if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  exit 1
fi

echo "== v10b restart: warm-start from $CKPT, bs=4, 4 GPUs, lr=5e-5, 26 epochs"

uv run torchrun --standalone --nproc_per_node=4 -m training.train \
  --gaussian_h5_dir data_v9/h5s \
  --renders_dir data_v9/renders \
  --save_dir checkpoints_v10b \
  --batch_size 4 --resolution 512 \
  --skip_phase1 \
  --resume "$CKPT" \
  --phase2_epochs 26 \
  --phase2_lr 5e-5 \
  --save_interval 2 \
  --val_h5_dir data_v9/h5s_val \
  --val_renders_dir data_v9/renders_val \
  --log_loss_weight 1.0 \
  --lpips_loss_weight 0.2 \
  --num_workers 8
