#!/bin/zsh
#SBATCH --time=01:00:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --output=runs/render_compare_v13b_ep4_%j.out
#SBATCH --job-name=rc_v13b_ep4
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue

# FINAL V13b comparison: ep4 is the keeper (32.83 dB / 0.0209 LPIPS, the cosine
# LR peak of the LPIPS fine-tune; run1 was preempted before a near-zero-LR ep5).
# Columns: GT | pruned-GT (if scene pruned) | V10b | V13b-ep4.
# Uses the md5-verified preserved checkpoint (evaluated_run1).
#
#   sbatch runs/render_compare_v13b_ep4.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1

uv run --frozen python -m render_compare \
  --models checkpoints_v10b/phase2_epoch_26.pt:V10b:rope \
           checkpoints_v13b/evaluated_run1/phase2_epoch_4.pt:V13b-ep4:nerf \
  --scenes 30,60,75,90,120,150,180 \
  --views 0,7 \
  --h5_dir data_v9_n20k/h5s_val \
  --gt_dir data_v9/renders_val \
  --out_dir compare_renders/v13b_ep4_FINAL_vs_v10b \
  --resolution 512 \
  --tone_mapper none \
  --pruned_gt
