#!/bin/zsh
#SBATCH --time=24:00:00
#SBATCH -c 16
#SBATCH --mem=64GB
#SBATCH --job-name=v10_proc
#SBATCH --output=runs/v10_proc_%x_%j.out
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue

# V10 full-splat data gen. Real zsh script (NOT --wrap) so module load cuda works -> gsplat.
# Pass OBJLIST, SPLIT, OUTDIR via --export. Resume-friendly (skips already-written scenes).
#   sbatch --export=ALL,OBJLIST=data_v10/object_list_TEST.json,SPLIT=train,OUTDIR=data_v10_test data_v10/run_process.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: ${OBJLIST:?set OBJLIST}
: ${SPLIT:=train}
: ${OUTDIR:=data_v10}

uv run --frozen python -m data_v10.process_full \
  --object_list $OBJLIST --split $SPLIT --out_dir $OUTDIR
