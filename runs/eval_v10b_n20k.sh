#!/bin/zsh
#SBATCH --time=02:00:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --output=runs/eval_v10b_n20k_%j.out
#SBATCH --job-name=eval_v10b_n20k
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue

# V10b ep26 (pe_type=rope, LPIPS-fine-tuned from V9, trained at N=5k) eval at
# INFERENCE N=20k on the full 183-scene val set. Production-model baseline: does
# even the published model benefit from more inference-time Gaussians?
#
# Submit:
#   sbatch runs/eval_v10b_n20k.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1

mkdir -p eval_results
uv run --frozen python -m eval_val_full \
  --checkpoint checkpoints_v10b/phase2_epoch_26.pt \
  --pe_type rope \
  --h5_dir data_v9_n20k/h5s_val \
  --gt_dir data_v9/renders_val \
  --out_json eval_results/v10b_ep26_n20k_val.json \
  --resolution 512 \
  --tone_mapper none
