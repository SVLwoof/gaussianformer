#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --mem=256GB
#SBATCH --output=runs/train_v11d_perfield_%j.out
#SBATCH --job-name=gformer_v11d
#SBATCH --mail-type=END,FAIL,BEGIN
# #SBATCH --mail-user=your-email@example.com
#SBATCH --gres=gg:g4:4

# v11d = per-field encoder, fully corrected after the v11c render+instrument debug.
#
# v11/v11b/v11c rendered a featureless grey blob (no color, no structure) and
# plateaued at train loss ~0.0138. Bugs found and fixed in the nerf_perfield path:
#
#   1. RoPE was disabled. The nerf_perfield branch left rope_dim=None in BOTH the
#      view-independent transformer and the view transformer. The RenderFormer
#      backbone is RoPE-pretrained, so its attention weights require rotary Q/K --
#      running them without RoPE broke attention everywhere. Fixed: rope_dim set
#      for nerf_perfield in gaussianformer.py and view_transformer.py.
#
#   2. Scale was NeRF-encoded out of range. log(scale) spans ~[-14,-2], far outside
#      NeRF's [0,1] working range -- every band aliased. Fixed: plain Linear on
#      log-scale (position still gets NeRF; its range is in-spec).
#
#   3. Per-field magnitude imbalance. Feeding raw log-scale (input magnitude ~10 vs
#      ~1 for other fields) into a plain Linear made scale_emb ~7x hot; it dominated
#      the summed token (ablation: removing scale shifted the token 117%, removing
#      color only 13%) -- the model saw scale and nothing else, hence the colorless
#      blob. Fixed: per-field RMSNorm on all 5 projections balances every field to
#      RMS 1.0 before summing, then one final norm sets token scale to ~sqrt(2).
#
# Recipe: short 2-epoch encoder warmup then full fine-tune, fresh from
# RenderFormer-v1-base. train.py saves phase1_epoch_2.pt at the end of Phase 1.
#
# Submit:
#   sbatch runs/train_v11d_perfield.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1

uv run --frozen torchrun --standalone --nproc_per_node=4 -m training.train \
  --gaussian_h5_dir data_v9/h5s \
  --renders_dir data_v9/renders \
  --save_dir checkpoints_v11d \
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
