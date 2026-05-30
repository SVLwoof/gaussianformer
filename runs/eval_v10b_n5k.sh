#!/bin/zsh
#SBATCH --time=02:00:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --output=runs/eval_v10b_n5k_%j.out
#SBATCH --job-name=eval_v10b_n5k
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue

# V10b ep26 (pe_type=rope, LPIPS-FT from V9) at INFERENCE N=5k -- V10b's
# training distribution. Production-model "trained-as-intended" baseline.

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1

mkdir -p eval_results
uv run --frozen python -m eval_val_full \
  --checkpoint checkpoints_v10b/phase2_epoch_26.pt \
  --pe_type rope \
  --h5_dir data_v9/h5s_val \
  --gt_dir data_v9/renders_val \
  --out_json eval_results/v10b_ep26_n5k_val.json \
  --resolution 512 \
  --tone_mapper agx
