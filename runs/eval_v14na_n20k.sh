#!/bin/zsh
#SBATCH --time=02:00:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --output=runs/eval_v14na_n20k_%j.out
#SBATCH --job-name=eval_v14na
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue

# V14 no-aug (rope @ N=20k, log-L1) final ep20 eval on the full 183-scene val set.
# The decisive comparison: rope@N=20k vs nerf@N=20k. Bars: V13 (nerf base) 33.58 dB /
# 0.0349; V13b (nerf + LPIPS FT) 32.83 / 0.0209. V14 is a log-L1 base, so V13 is the
# apples-to-apples comparator.
#
#   sbatch runs/eval_v14na_n20k.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1

mkdir -p eval_results
uv run --frozen python -m eval_val_full \
  --checkpoint checkpoints_v14_noaug/phase2_epoch_20.pt \
  --pe_type rope \
  --h5_dir data_v9_n20k/h5s_val \
  --gt_dir data_v9/renders_val \
  --out_json eval_results/v14na_ep20_n20k_val.json \
  --resolution 512 \
  --tone_mapper none
