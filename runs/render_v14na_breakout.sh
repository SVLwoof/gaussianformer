#!/bin/zsh
#SBATCH --time=00:20:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --output=runs/render_v14na_breakout_%j.out
#SBATCH --job-name=rc_v14na_bo
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue

# V14 no-aug post-breakout render (ep11, avg 0.0087) vs current best V13b. Are objects
# materializing now? Columns: GT | V14na-ep11 | V13b.
#
#   sbatch runs/render_v14na_breakout.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1

uv run --frozen python -m render_compare \
  --models checkpoints_v14_noaug/phase2_epoch_11.pt:V14na-ep11:rope \
           checkpoints_v13b/evaluated_run1/phase2_epoch_4.pt:V13b:nerf \
  --scenes 30,75,90,180 \
  --views 0 \
  --h5_dir data_v9_n20k/h5s_val \
  --gt_dir data_v9/renders_val \
  --out_dir compare_renders/v14na_ep11_check \
  --resolution 512 \
  --tone_mapper none
