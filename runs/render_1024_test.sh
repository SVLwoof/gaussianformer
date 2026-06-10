#!/bin/zsh
#SBATCH --time=00:40:00
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --output=runs/render_1024_test_%j.out
#SBATCH --job-name=render_1024
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue

# Native-1024 render test: does the finer token grid (1024/8 = 128x128, 4x denser
# than native 512's 64x64) surface more detail, or break down OOD? Inference-only,
# no weight change -- higher res = finer grid via res/patch_size.
# Columns: GT(512 upscaled, loose ctx) | pruned-GT(gsplat 20k @1024, crisp ref) |
#          V10b@1024 | V13 ep20@1024.

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run --frozen python -m render_compare \
  --models checkpoints_v10b/phase2_epoch_26.pt:V10b:rope \
           checkpoints_v13/phase2_epoch_20.pt:V13:nerf \
  --pruned_gt \
  --scenes 30,75,180 \
  --views 0 \
  --h5_dir data_v9_n20k/h5s_val \
  --gt_dir data_v9/renders_val \
  --out_dir compare_renders/res1024_v10b_v13 \
  --resolution 1024 \
  --tone_mapper none
