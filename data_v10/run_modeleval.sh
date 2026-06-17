#!/bin/zsh
#SBATCH --time=2:00:00
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH --job-name=v10_modeleval
#SBATCH --output=runs/v10_modeleval_%j.out
#SBATCH --gres=gg:g4:1
#SBATCH --account=sagieb
#SBATCH --killable
#SBATCH --requeue

# Render GT | pruned-GT | model strips (model_on_v10.py format) for an arbitrary checkpoint.
# Killable 1-GPU eval; does not touch the training jobs. gsplat JIT cache warmed from prior
# recovery runs (single job -> no compile race).
#   CKPT=... LABEL=... OUTNAME=... SCENES=... VIEWS=... sbatch data_v10/run_modeleval.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

CKPT=${CKPT:-checkpoints_v15_lpips/phase2_epoch_2.pt}
LABEL=${LABEL:-V15-lpips-ep2}
SCENES=${SCENES:-data_v10/scenes_preview.json}
OUTNAME=${OUTNAME:-v15_on_v10}
VIEWS=${VIEWS:-0,6}
INPUT_MODE=${INPUT_MODE:-naive}

echo "ckpt=$CKPT label=$LABEL scenes=$SCENES views=$VIEWS input_mode=$INPUT_MODE"
uv run --frozen python -m data_v10.model_on_v10 \
  --scenes_file $SCENES --ckpt $CKPT --label $LABEL \
  --out data_v10/model_eval --out_name $OUTNAME --views $VIEWS --input_mode $INPUT_MODE
echo DONE_WRAP
