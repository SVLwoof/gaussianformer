#!/bin/zsh
#SBATCH --time=1:00:00
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH --job-name=ovf_novel
#SBATCH --output=runs/ovf_novel_%j.out
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue

# Novel-view generalisation probe for the tomatoes overfits. MUST be a real zsh sbatch
# script (not sbatch --wrap, which runs under /bin/sh where `source`/`module` don't exist,
# so `module load cuda` silently fails -> gsplat's CUDA backend can't load -> _C=None).
#
#   sbatch experiments/overfit/run_novel.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TOM_H5=experiments/overfit/data/tomatoes/h5s/tomatoes_n20000.h5
for lbl in tomatoes_v14best tomatoes_base; do
  echo "##### $lbl #####"
  uv run --frozen python -m experiments.overfit.novel_view \
    --ckpt experiments/overfit/ckpt/$lbl/phase2_epoch_1500.pt \
    --scene tomatoes --h5 $TOM_H5 --label $lbl --out_dir experiments/overfit/eval
done
echo DONE_NOVEL
