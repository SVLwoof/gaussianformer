#!/bin/zsh
#SBATCH --time=01:00:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --output=runs/render_compare_v14na_final_%j.out
#SBATCH --job-name=rc_v14na_fin
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue

# V14 no-aug final (ep20) comparison vs V13b. Columns: GT | pruned-GT | V13b | V14na-ep20.
#
#   sbatch runs/render_compare_v14na_final.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1

uv run --frozen python -m render_compare \
  --models checkpoints_v13b/evaluated_run1/phase2_epoch_4.pt:V13b:nerf \
           checkpoints_v14_noaug/phase2_epoch_20.pt:V14na:rope \
  --scenes 30,60,75,90,120,150,180 \
  --views 0,7 \
  --h5_dir data_v9_n20k/h5s_val \
  --gt_dir data_v9/renders_val \
  --out_dir compare_renders/v14na_ep20_FINAL \
  --resolution 512 \
  --tone_mapper none \
  --pruned_gt
