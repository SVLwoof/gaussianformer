#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --mem=256GB
#SBATCH --output=runs/train_v12_%j.out
#SBATCH --job-name=gformer_v12
#SBATCH --mail-type=END,FAIL,BEGIN
# #SBATCH --mail-user=your-email@example.com
#SBATCH --gres=gg:g4:4
#
# V12 = the nerf-concat encoder under the CORRECTED two-phase recipe.
#
# The V11 plateau (0.0138) was never the encoder: rope, per-field, and concat all
# plateau identically under joint Phase 2 from a random encoder (the rope-jointscratch
# control, job 30629854, settled bin-for-bin on top of v11e/v11f). The cause was the
# recipe -- V11 dropped Phase 1 (`--skip_phase1`) because in-process Phase 1 crashed
# under DDP. train.py is now fixed: DDP is constructed per phase, after the backbone
# freeze, so Phase 1 runs with find_unused_parameters=False and no crash.
#
# This is a single end-to-end run: RenderFormer transfer -> Phase 1 (encoder warmup,
# frozen backbone) -> Phase 2 (joint fine-tune). No --skip_phase1, no --resume.
# It is the first run that can actually test the V11 hypothesis (NeRF position
# encoding for blur) under a recipe that converges.
#
# Verdict signals (see PROGRESS.md):
#   - Phase 1 ep 1 does not crash (the freeze-before-DDP-wrap fix).
#   - Phase 1 loss descends across its 20 epochs.
#   - Phase 1->2 transition prints `phase2: ~194.9M trainable`, no crash (DDP re-wrap).
#   - Phase 2 ep 1 within-epoch binning DESCENDS (V9 Phase 2 ep 1 was 0.003839),
#     not flat at 0.0138 -> the recipe is fixed.
#
# bs=4 fits any g4 card (the bs=5 OOM lesson).
#
# Submit:
#   sbatch runs/train_v12.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run --frozen torchrun --standalone --nproc_per_node=4 -m training.train \
  --gaussian_h5_dir data_v9/h5s \
  --renders_dir data_v9/renders \
  --save_dir checkpoints_v12 \
  --batch_size 4 --resolution 512 \
  --pe_type nerf \
  --phase1_epochs 20 --phase1_lr 1e-3 \
  --phase2_epochs 100 --phase2_lr 5e-5 \
  --save_interval 5 \
  --val_h5_dir data_v9/h5s_val \
  --val_renders_dir data_v9/renders_val \
  --log_loss_weight 1.0 \
  --lpips_loss_weight 0.0 \
  --num_workers 8
