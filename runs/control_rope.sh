#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --mem=256GB
#SBATCH --output=runs/control_rope_%j.out
#SBATCH --job-name=gformer_control_rope
#SBATCH --mail-type=END,FAIL,BEGIN
# #SBATCH --mail-user=your-email@example.com
#SBATCH --gres=gg:g4:4

# CONTROL EXPERIMENT: V9's exact recipe (pe_type=rope) on the current codebase.
#
# Every V11 run (7 of them) plateaued Phase 2 at train loss 0.0138, regardless of
# encoder fixes (RoPE, scale, field balance). That rules out the input encoder.
# This control isolates the remaining suspect: shared code that changed since V9
# (the train.py diff -- notably find_unused_parameters=True -- deps, environment).
#
#   pe_type=rope               -> V9's monolithic Linear(14,768) encoder
#   --skip_phase1 --resume       checkpoints_v4/phase1_epoch_20.pt (V9's exact init)
#   --phase2_lr 5e-5             V9's recipe
#
# Verdict from ep 1-3 (~2 h):
#   loss descends toward ~0.004  -> V9 reproduces; the bug IS the per-field encoder.
#   loss plateaus at ~0.0138     -> the bug is in shared code changed since V9.
#
# Submit:
#   sbatch runs/control_rope.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run --frozen torchrun --standalone --nproc_per_node=4 -m training.train \
  --gaussian_h5_dir data_v9/h5s \
  --renders_dir data_v9/renders \
  --save_dir checkpoints_control_rope \
  --resume checkpoints_v4/phase1_epoch_20.pt \
  --skip_phase1 \
  --batch_size 4 --resolution 512 \
  --pe_type rope \
  --phase2_epochs 100 \
  --phase2_lr 5e-5 \
  --save_interval 5 \
  --val_h5_dir data_v9/h5s_val \
  --val_renders_dir data_v9/renders_val \
  --log_loss_weight 1.0 \
  --lpips_loss_weight 0.0 \
  --num_workers 8
