#!/bin/zsh
#SBATCH --time=01:00:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --output=runs/render_compare_v13_ep10_%j.out
#SBATCH --job-name=render_v13_ep10
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue

# 4-way visual compare on val scenes: GT | V10b ep26 | V12 ep75 | V13 ep10.
# All three models inferred at N=20k on data_v9_n20k/h5s_val. 7 scenes, 2 views.
# Outputs per-(scene, view) strips + a grid_overview.png under
# compare_renders/v13_ep10_vs_baselines/.

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1

uv run --frozen python -m render_compare \
  --models checkpoints_v10b/phase2_epoch_26.pt:V10b:rope \
           checkpoints_v12/phase2_epoch_75.pt:V12:nerf \
           checkpoints_v13/phase2_epoch_10.pt:V13:nerf \
  --scenes 0,30,60,90,120,150,180 \
  --views 0,7 \
  --h5_dir data_v9_n20k/h5s_val \
  --gt_dir data_v9/renders_val \
  --out_dir compare_renders/v13_ep10_vs_baselines \
  --resolution 512 \
  --tone_mapper agx
