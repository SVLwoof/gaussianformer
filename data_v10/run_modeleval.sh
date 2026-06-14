#!/bin/zsh
#SBATCH --time=2:00:00
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH --job-name=v10_modeleval
#SBATCH --output=runs/v10_modeleval_%j.out
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue
source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
uv run --frozen python -m data_v10.model_on_v10 --scenes_file data_v10/scenes_show.json --out data_v10/model_eval
echo DONE_WRAP
