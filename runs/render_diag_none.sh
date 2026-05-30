#!/bin/zsh
#SBATCH --time=00:30:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --output=runs/render_diag_none_%j.out
#SBATCH --job-name=render_diag_none
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue

# DIAGNOSTIC: re-render the highlight scenes with tone_mapper=none (clip), which
# is how the models were actually trained (GT = no-tonemap LDR). Compares against
# the AGX versions to isolate how much of the "washed color" was an AGX artifact
# introduced only in eval, not present in training/canonical inference.

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1

uv run --frozen python -m render_compare \
  --models checkpoints_v10b/phase2_epoch_26.pt:V10b:rope \
           checkpoints_v12/phase2_epoch_75.pt:V12:nerf \
           checkpoints_v13/phase2_epoch_20.pt:V13:nerf \
  --scenes 0,30,75,180 \
  --views 0 \
  --h5_dir data_v9_n20k/h5s_val \
  --gt_dir data_v9/renders_val \
  --out_dir compare_renders/v13_ep20_TONEMAP_none \
  --resolution 512 \
  --tone_mapper none
