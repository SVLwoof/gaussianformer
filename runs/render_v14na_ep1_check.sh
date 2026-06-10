#!/bin/zsh
#SBATCH --time=00:20:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --output=runs/render_v14na_ep1_check_%j.out
#SBATCH --job-name=rc_v14na_ep1
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue

# Definitive blob-vs-content check for the NO-AUG V14 phase2_epoch_1 (loss ~0.0139,
# same plateau value as the aug run -- but is it black like aug, or real content?).
# Columns: GT | V14na-ep1.
#
#   sbatch runs/render_v14na_ep1_check.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1

uv run --frozen python -m render_compare \
  --models checkpoints_v14_noaug/phase2_epoch_1.pt:V14na-ep1:rope \
  --scenes 30,75,90,180 \
  --views 0 \
  --h5_dir data_v9_n20k/h5s_val \
  --gt_dir data_v9/renders_val \
  --out_dir compare_renders/v14na_ep1_check \
  --resolution 512 \
  --tone_mapper none
