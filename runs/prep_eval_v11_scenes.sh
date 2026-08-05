#!/bin/zsh
#SBATCH --time=01:00:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --output=runs/prep_eval_v11_scenes_%j.out
#SBATCH --job-name=prep_v11_scenes
#SBATCH --gres=gg:g4:1

# One-shot prep for the 3 new V11 eval scenes: house, dragon, cartoon.
# Downloads each scene's PLY from ShapeSplats/Objaverse_Splats and renders the
# 14-view full-gsplat reference images. After this completes, the scenes are
# ready for `python -m data_external.run_scene --scene <name> ...`.
#
# Submit (when GPU GRES is free; will likely queue behind V11 training):
#   sbatch runs/prep_eval_v11_scenes.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1

uv run --frozen python -m data_external.prep_objaverse_scene \
  --scene house \
  --scene dragon \
  --scene cartoon
