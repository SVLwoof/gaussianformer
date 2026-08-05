#!/bin/zsh
#SBATCH --time=02:00:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --output=runs/eval_v14auglp_%j.out
#SBATCH --job-name=eval_v14auglp
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue
source /etc/profile.d/huji-lmod.sh
module load nvidia; module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
CKPT="${CKPT:-checkpoints_v14aug_lpips/phase2_epoch_1.pt}"
EP="${CKPT:t:r}"
mkdir -p eval_results
uv run --frozen python -m eval_val_full \
  --checkpoint "$CKPT" --pe_type rope \
  --h5_dir data_v9_n20k/h5s_val --gt_dir data_v9/renders_val \
  --out_json "eval_results/v14auglp_${EP}_n20k_val.json" \
  --resolution 512 --tone_mapper none
