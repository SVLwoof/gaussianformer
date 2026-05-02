#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --mem=256GB
#SBATCH --output=runs/train_phase2_v9_%j.out
#SBATCH --job-name=gformer_phase2_v9
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --gres=gg:g4:3

# v9 = phase 2 fine-tune on Objaverse_Splats single-object data (2667 train, 183 val).
# Same loss as v6 (log-HDR L1) for the data-first experiment: clean isolated
# objects + ~2.7x v6's dataset size. LPIPS plumbing is already in
# training/train.py; if first run plateaus, re-launch with
# --lpips_loss_weight 0.5.
#
# DDP across 3 L40S/A6000-class GPUs via torchrun. Per-device bs=2, global bs=6
# (vs v6's bs=2). LR kept at 5e-5 for the first run -- prefer attributing
# improvement to data over LR; sqrt-rule bump (5e-5 -> ~9e-5) is the natural
# next-iter ablation if needed. Resumes from v4's phase1 input-encoder warmup.
#
# Wall-clock measured: ~58 min/epoch at 1.78 step/s sustained (workers=4, persistent).
# DDP comm overhead is much heavier than the back-of-envelope ~10% predicted -- per-GPU
# rate is ~1/4 of v6's single-GPU rate. Likely culprit: 200M-param all-reduce on
# inter-GPU PCIe (no NVLink confirmed). 100 epochs => ~97h, so 168h alloc (max on
# `medium` partition) gives comfortable headroom.
#
# (3 GPUs not 4 because our group's GRES cap is 4 total and the interactive
# session is holding one slot; bumping to 4 here would block forever on
# AssocGrpGRES. Trivial bump back to 4 once the interactive ends.)

# Cluster's lmod isn't sourced into non-interactive zsh by default; pull it
# in explicitly so `module load` actually resolves.
source "${LMOD_INIT:-/etc/profile.d/lmod.sh}"
module load nvidia
module load cuda
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1

torchrun --standalone --nproc_per_node=3 -m training.train \
  --gaussian_h5_dir data_v9/h5s \
  --renders_dir data_v9/renders \
  --save_dir checkpoints_v9 \
  --batch_size 2 --resolution 512 \
  --skip_phase1 \
  --resume checkpoints_v4/phase1_epoch_20.pt \
  --phase2_epochs 100 \
  --phase2_lr 5e-5 \
  --save_interval 5 \
  --val_h5_dir data_v9/h5s_val \
  --val_renders_dir data_v9/renders_val \
  --log_loss_weight 1.0 \
  --lpips_loss_weight 0.0 \
  --num_workers 8
