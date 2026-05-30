#!/bin/zsh
#SBATCH --time=02:00:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --output=runs/eval_v13_n20k_%j.out
#SBATCH --job-name=eval_v13_n20k
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue

# V13 ep20 (pe_type=nerf, trained at N=20k) eval at INFERENCE N=20k on the full
# 183-scene val set. This is the apples-to-best-apples comparator: V13 at its
# trained N vs V10b@N5k (25.597 dB) and V12@N5k (25.482 dB) -- each model at its
# own training distribution. Decides whether training at N=20k beats production.
#
# Submit:
#   sbatch runs/eval_v13_n20k.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1

mkdir -p eval_results
uv run --frozen python -m eval_val_full \
  --checkpoint checkpoints_v13/phase2_epoch_20.pt \
  --pe_type nerf \
  --h5_dir data_v9_n20k/h5s_val \
  --gt_dir data_v9/renders_val \
  --out_json eval_results/v13_ep20_n20k_val.json \
  --resolution 512 \
  --tone_mapper none
