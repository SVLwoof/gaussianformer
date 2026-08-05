#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --mem=256GB
#SBATCH --output=runs/train_v11b_perfield_fixed_%j.out
#SBATCH --job-name=gformer_v11b_fixed
#SBATCH --mail-type=END,FAIL,BEGIN
# #SBATCH --mail-user=your-email@example.com
#SBATCH --gres=gg:g4:4

# v11b = v11 per-field encoder with the scale-encoding bug fixed.
#
# v11 plateaued at train loss ~0.0138 across every Phase 2 config (vs V9's 0.0009).
# Root cause: construct_sequence NeRF-encoded log(scale). Real scales are [1e-6, 0.11]
# so log_scale spans [-13.8, -2.2] -- ~14x outside NeRF's [0,1] working range. Every
# sinusoidal band aliased; the model could not read Gaussian scale (= splat footprint),
# so it defaulted to an average blur and the loss froze.
#
# Fix (gaussianformer/models/gaussianformer.py, nerf_perfield branch):
#   1. Scale is no longer NeRF-encoded -- plain Linear(3,768) on raw log_scale.
#   2. The 5 per-field RMSNorms are replaced by one final norm on the summed token
#      (summing 5 separately-normed embeddings ran the token ~sqrt(6) hot vs the
#      sqrt(2) the pretrained backbone expects).
# Position still gets NeRF PE (its range is in-spec; that part was never broken).
#
# Recipe mirrors the short-Phase-1 schedule: 2-epoch encoder warmup then full
# fine-tune. Fresh init from RenderFormer-v1-base (the per-field encoder is
# structurally new and not transferable from v10b). train.py saves phase1_epoch_2.pt
# at the end of Phase 1 as a fallback for a separate-job Phase 2 if ever needed.
#
# Submit:
#   sbatch runs/train_v11b_perfield_fixed.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1

uv run --frozen torchrun --standalone --nproc_per_node=4 -m training.train \
  --gaussian_h5_dir data_v9/h5s \
  --renders_dir data_v9/renders \
  --save_dir checkpoints_v11b \
  --batch_size 5 --resolution 512 \
  --pe_type nerf_perfield \
  --phase1_epochs 2 \
  --phase1_lr 3e-4 \
  --phase2_epochs 100 \
  --phase2_lr 5e-5 \
  --save_interval 5 \
  --val_h5_dir data_v9/h5s_val \
  --val_renders_dir data_v9/renders_val \
  --log_loss_weight 1.0 \
  --lpips_loss_weight 0.0 \
  --num_workers 8
