#!/bin/zsh
#SBATCH --time=02:00:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --output=runs/eval_v12_n5k_%j.out
#SBATCH --job-name=eval_v12_n5k
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue

# V12 ep75 (pe_type=nerf) at INFERENCE N=5k -- V12's training distribution.
# Pairs with eval_v12_n20k.sh to form the V12-side of the 2x2 N-sweep on the
# full 183-scene val set.

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1

mkdir -p eval_results
uv run --frozen python -m eval_val_full \
  --checkpoint checkpoints_v12/phase2_epoch_75.pt \
  --pe_type nerf \
  --h5_dir data_v9/h5s_val \
  --gt_dir data_v9/renders_val \
  --out_json eval_results/v12_ep75_n5k_val.json \
  --resolution 512 \
  --tone_mapper agx
