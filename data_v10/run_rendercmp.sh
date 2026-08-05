#!/bin/zsh
#SBATCH --time=1:00:00
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH --job-name=v10_rendercmp
#SBATCH --output=runs/v10_rendercmp_%j.out
#SBATCH --gres=gg:g4:1
#SBATCH --account=sagieb
#SBATCH --killable
#SBATCH --requeue
source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
SCENES=${SCENES:-data_v10/scenes_rendercmp.json}
OUT=${OUT:-data_v10/recovery_eval/final_compare}
VIEWS=${VIEWS:-0,6}
echo "scenes=$SCENES out=$OUT views=$VIEWS"
uv run --frozen python -m data_v10.render_recovery_compare --scenes_file $SCENES --out $OUT --views $VIEWS
