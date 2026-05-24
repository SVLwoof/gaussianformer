#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --mem=256GB
#SBATCH --output=runs/train_v11c_perfield_%j.out
#SBATCH --job-name=gformer_v11c
#SBATCH --mail-type=END,FAIL,BEGIN
# #SBATCH --mail-user=your-email@example.com
#SBATCH --gres=gg:g4:4

# v11c = per-field encoder with all three v11 bugs fixed.
#
# v11/v11b plateaued at train loss ~0.0138 across every config (vs V9's ~0.001).
# Three bugs found, all in the nerf_perfield path:
#
#   1. RoPE was disabled. The nerf_perfield branch left rope_dim=None in BOTH the
#      view-independent transformer and the view transformer. The RenderFormer
#      backbone is RoPE-pretrained, so its attention weights require rotary Q/K --
#      running them without RoPE broke attention everywhere. This is the primary
#      cause: it is input-encoder-independent, which is why every run plateaued at
#      the identical 0.0138 regardless of LR or encoder details. Fixed: rope_dim is
#      now set for nerf_perfield in gaussianformer.py and view_transformer.py; the
#      per-field NeRF encoding is additive input richness on top of RoPE.
#
#   2. Scale was NeRF-encoded out of range. log(scale) spans ~[-14,-2], far outside
#      NeRF's [0,1] working range -- every band aliased. Fixed: plain Linear on
#      log-scale (position still gets NeRF; its range is in-spec).
#
#   3. Token ran sqrt(6) hot. Summing 5 separately-normed per-field embeddings.
#      Fixed: one final norm on the summed token (RMS now ~sqrt(2), matching the
#      rope path the backbone expects).
#
# Recipe: short 2-epoch encoder warmup then full fine-tune, fresh from
# RenderFormer-v1-base. train.py saves phase1_epoch_2.pt at the end of Phase 1.
#
# Submit:
#   sbatch runs/train_v11c_perfield.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1

uv run --frozen torchrun --standalone --nproc_per_node=4 -m training.train \
  --gaussian_h5_dir data_v9/h5s \
  --renders_dir data_v9/renders \
  --save_dir checkpoints_v11c \
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
