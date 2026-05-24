#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --mem=256GB
#SBATCH --output=runs/train_v11e_perfield_jointscratch_%j.out
#SBATCH --job-name=gformer_v11e
#SBATCH --mail-type=END,FAIL,BEGIN
# #SBATCH --mail-user=your-email@example.com
#SBATCH --gres=gg:g4:4

# v11e = discriminating experiment for the per-field Phase-2 plateau.
#
# Background: with find_unused_parameters fixed, the rope encoder descends on the
# current codebase but nerf_perfield still plateaus at 0.0138. The forward is
# healthy (compare_forward.py: per-field token propagates through the backbone
# within ~10% of a rope token at every layer), so it is a training-dynamics bug.
# Two confounded causes:
#   (A) checkpoint provenance -- every per-field run resumed a per-field Phase-1
#       checkpoint that itself plateaued in Phase 1.
#   (B) the per-field architecture itself (5 forced-equal-magnitude fields).
#
# This run separates them: --skip_phase1 with NO --resume. The model is a fresh
# RenderFormer transfer with a RANDOM per-field encoder; Phase 2 trains everything
# jointly from step 1. No Phase-1 checkpoint in the picture.
#   plateaus at 0.0138  -> cause (B), the architecture.
#   descends toward ~0.004 -> cause (A); the fix is a proper Phase-1 warmup.
#
# --skip_phase1 => find_unused_parameters=False (the rope-confirmed-correct path).
# bs=4 fits any g4 card (the bs=5 OOM lesson).
#
# Submit:
#   sbatch runs/train_v11e_perfield_jointscratch.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run --frozen torchrun --standalone --nproc_per_node=4 -m training.train \
  --gaussian_h5_dir data_v9/h5s \
  --renders_dir data_v9/renders \
  --save_dir checkpoints_v11e \
  --skip_phase1 \
  --batch_size 4 --resolution 512 \
  --pe_type nerf_perfield \
  --phase2_epochs 100 \
  --phase2_lr 5e-5 \
  --save_interval 5 \
  --val_h5_dir data_v9/h5s_val \
  --val_renders_dir data_v9/renders_val \
  --log_loss_weight 1.0 \
  --lpips_loss_weight 0.0 \
  --num_workers 8
