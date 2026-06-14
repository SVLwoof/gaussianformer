#!/bin/zsh
#SBATCH --time=2:00:00
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH --job-name=ovf_eval
#SBATCH --output=runs/ovf_eval_%j.out
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue

# Three-way eval of the NEWEST phase2 checkpoint of each overfit run (skips runs with no
# phase2 checkpoint yet). Reusable: run mid-training to validate the eval path, and again
# at the end for final numbers. Writes JSON + strips to experiments/overfit/eval/.
#
#   sbatch experiments/overfit/run_eval.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

eval_one() {  # obj init h5 realgt_dir
  local obj=$1 init=$2 h5=$3 realgt=$4
  local files=( experiments/overfit/ckpt/${obj}_${init}/phase2_epoch_*.pt(Nom) )
  if [ ${#files} -eq 0 ]; then echo "SKIP ${obj}_${init}: no phase2 checkpoint yet"; return; fi
  echo "=== eval ${obj}_${init}  ${files[1]} ==="
  uv run --frozen python -m experiments.overfit.eval_overfit \
    --ckpt ${files[1]} --h5 $h5 --realgt_dir $realgt \
    --label ${obj}_${init} --out_dir experiments/overfit/eval
}

BOX_H5=experiments/overfit/data/boxes/h5s/scene_1441.h5
TOM_H5=experiments/overfit/data/tomatoes/h5s/tomatoes_n20000.h5
TOM_GT=experiments/overfit/data/tomatoes/renders

eval_one boxes    base    $BOX_H5 data_v9/renders
eval_one boxes    v14best $BOX_H5 data_v9/renders
eval_one tomatoes base    $TOM_H5 $TOM_GT
eval_one tomatoes v14best $TOM_H5 $TOM_GT
echo "DONE"
