#!/bin/zsh
#SBATCH --time=02:00:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --output=runs/eval_v13b_n20k_%j.out
#SBATCH --job-name=eval_v13b_n20k
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue

# V13b LPIPS fine-tune (warm-started from V13 ep20, pe_type=nerf) eval at
# INFERENCE N=20k on the full 183-scene val set. Decisive metric: val LPIPS vs
# V10b's 0.0283 bar; PSNR vs V13's +2.67 dB lead over production.
#
# CKPT is passed via env so the same script serves each epoch:
#   CKPT=checkpoints_v13b/phase2_epoch_1.pt sbatch runs/eval_v13b_n20k.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1

CKPT="${CKPT:-checkpoints_v13b/phase2_epoch_1.pt}"
EP="${CKPT:t:r}"  # e.g. phase2_epoch_1

mkdir -p eval_results
uv run --frozen python -m eval_val_full \
  --checkpoint "$CKPT" \
  --pe_type nerf \
  --h5_dir data_v9_n20k/h5s_val \
  --gt_dir data_v9/renders_val \
  --out_json "eval_results/v13b_${EP}_n20k_val.json" \
  --resolution 512 \
  --tone_mapper none
