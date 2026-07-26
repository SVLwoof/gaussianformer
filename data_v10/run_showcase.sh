#!/bin/zsh
#SBATCH --time=2:00:00
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH --job-name=v10_showcase
#SBATCH --output=runs/v10_showcase_%j.out
#SBATCH --gres=gg:g4:1
#SBATCH --account=sagieb
#SBATCH --killable
#SBATCH --requeue

# Paper showcase: GT | V14best | V16+LPIPS | V17-ep36 rows over random unseen objects.
# Killable 1-GPU; loads all 3 gens on one GPU. gsplat JIT cache already warm from ep36 eval.
#   SCENES_FILE=... OUTNAME=... VIEWS=0,7 sbatch data_v10/run_showcase.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

SCENES_FILE=${SCENES_FILE:-data_v10/showcase_scenes/shard_00.json}
OUTNAME=${OUTNAME:-showcase_00}
VIEWS=${VIEWS:-0,3,7,10}

echo "scenes_file=$SCENES_FILE out_name=$OUTNAME views=$VIEWS"
uv run --frozen python -m data_v10.showcase_versions \
  --scenes_file $SCENES_FILE --views $VIEWS \
  --out data_v10/showcase --out_name $OUTNAME
echo DONE_WRAP
