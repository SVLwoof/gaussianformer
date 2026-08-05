#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --mem=256GB
#SBATCH --output=runs/train_v11_perfield_%j.out
#SBATCH --job-name=gformer_v11_perfield
#SBATCH --mail-type=END,FAIL,BEGIN
# #SBATCH --mail-user=your-email@example.com
#SBATCH --gres=gg:g4:4

# v11 = per-field input encoder with NeRF positional encoding on position + log-scale.
# Fresh init from RenderFormer-v1-base (NOT a resume from v10b): the monolithic
# Linear(14, 768) is replaced by 5 per-field projections summed additively, so the
# v10b encoder weights are not transferable. Backbone (transformer, view_transformer,
# DPT) still copies from RenderFormer-v1-base via training.weight_transfer.
#
# Recipe mirrors v9 (clean log-L1 baseline, no LPIPS) so the comparison isolates the
# architectural change. Phase 1 warms up the new per-field encoder with the backbone
# frozen, then Phase 2 unfreezes all. 4 GPUs, bs=5 -> global bs 20 (vs v10b's 16).
#
# bs=5 chosen via VRAM probe sweep (jobs 30617186-89, 2026-05-19): bs=4 peak 33.2 GB,
# bs=5 peak 35.9 GB, bs=8 OOM at rank 2. Extrapolating the V10b probe-to-production
# delta (~7.4 GB), bs=5 production peak ~43/46 GB with safe headroom.
#
# Submit:
#   sbatch runs/train_v11_perfield.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1

uv run --frozen torchrun --standalone --nproc_per_node=4 -m training.train \
  --gaussian_h5_dir data_v9/h5s \
  --renders_dir data_v9/renders \
  --save_dir checkpoints_v11 \
  --batch_size 5 --resolution 512 \
  --pe_type nerf_perfield \
  --scale_pe_num_freqs 6 \
  --phase1_epochs 5 \
  --phase1_lr 1e-3 \
  --phase2_epochs 100 \
  --phase2_lr 5e-5 \
  --save_interval 5 \
  --val_h5_dir data_v9/h5s_val \
  --val_renders_dir data_v9/renders_val \
  --log_loss_weight 1.0 \
  --lpips_loss_weight 0.0 \
  --num_workers 8
