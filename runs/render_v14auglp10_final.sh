#!/bin/zsh
#SBATCH --time=00:25:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --output=runs/render_v14auglp10_final_%j.out
#SBATCH --job-name=rc_lp10
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue
source /etc/profile.d/huji-lmod.sh
module load nvidia; module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
uv run --frozen python -m render_compare \
  --models checkpoints_v13b/evaluated_run1/phase2_epoch_4.pt:V13b:nerf \
           checkpoints_v14auglp10/phase2_epoch_10.pt:V14best:rope \
  --scenes 30,60,75,90,120,150,180 --views 0,7 \
  --h5_dir data_v9_n20k/h5s_val --gt_dir data_v9/renders_val \
  --out_dir compare_renders/v14auglp10_ep10_BEST --resolution 512 --tone_mapper none --pruned_gt
