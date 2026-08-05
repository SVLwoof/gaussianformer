#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --mem=256GB
#SBATCH --output=runs/train_v13_%j.out
#SBATCH --job-name=gformer_v13
#SBATCH --mail-type=END,FAIL,BEGIN
# #SBATCH --mail-user=your-email@example.com
#SBATCH --gres=gg:g4:4
#
# V13 = time-boxed exploratory run testing the "5k Gaussians per scene is too
# compressed" hypothesis. Trains at N=20,000 (4x more Gaussians per scene than
# V12's N=5,000), bs=2 per GPU (vs V12's bs=4), end-to-end recipe.
#
# Memory: bs=2 N=20k probe measured 39.4 GB single-GPU peak. With DDP overhead
# (~1-2 GB) the production peak should sit ~41-42 GB -- fits 44 GB g4 cards
# with thin margin, comfortable on 46 GB. expandable_segments is on.
#
# Compute: ~3.3 s/step bs=2 N=20k. At 4-GPU DDP with 4625 per-rank steps per
# epoch, that is ~4 h 13 m per epoch. 30 epochs budget -> ~126 h (5.25 days),
# within the 168 h walltime cap.
#
# Epoch split: 10 Phase 1 (encoder warmup; the recipe-critical step that lets
# Phase 2 escape the 0.0138 attractor -- V12 took 8 plateau-epochs to break out
# WITH a warmed encoder) + 20 Phase 2 (joint fine-tune, where the actual quality
# signal lives).
#
# Data: data_v9_n20k/h5s* are produced by runs/regen_data_n20k.sh (re-pruning
# the Objaverse sources to N=20k). GT renders are FULL-scene rasterizations,
# target_n-independent, so we reuse data_v9/renders/ unchanged.
#
# Verdict question (after run): does V13 at N=20k inference beat V12 at N=20k
# inference by a meaningful margin on the eval suite? If yes, the compression
# hypothesis is validated and we scale up. If no, the architecture / dataset
# ceiling -- not the input compression -- is what was limiting V12.
#
# LRs left at V12 defaults (phase1=1e-3, phase2=5e-5). Effective batch dropped
# from 16 (V12) to 8 here, so gradients are ~sqrt(2) noisier. Acceptable for
# a time-boxed exploratory run; revisit if Phase 1 diverges.
#
# Submit (after regen_data_n20k.sh completes):
#   sbatch runs/train_v13.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run --frozen torchrun --standalone --nproc_per_node=4 -m training.train \
  --gaussian_h5_dir data_v9_n20k/h5s \
  --renders_dir data_v9/renders \
  --save_dir checkpoints_v13 \
  --batch_size 2 --resolution 512 \
  --pe_type nerf \
  --phase1_epochs 10 --phase1_lr 1e-3 \
  --phase2_epochs 20 --phase2_lr 5e-5 \
  --save_interval 5 \
  --val_h5_dir data_v9_n20k/h5s_val \
  --val_renders_dir data_v9/renders_val \
  --log_loss_weight 1.0 \
  --lpips_loss_weight 0.0 \
  --num_workers 8
