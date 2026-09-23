#!/bin/zsh
#SBATCH --time=12:00:00
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH --output=runs/mradfull_%A_%a.out
#SBATCH --job-name=mradfull
#SBATCH --gres=gg:g4:1
#SBATCH --requeue
#SBATCH --array=0-10
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08,khan-01,khan-02
# Full-data multi-radius views (data_v10/multi_radius_full.py), one chunk group per array task, resume-safe.
# Chunk zips (~3 GB each) go to a node-local HF cache, never the lab share.
#   sbatch --killable --account=killable-cs data_v10/multi_radius_full.sh
source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1 PYTHONPATH=.
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HF_HOME=/tmp/hf_mradfull_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
trap 'rm -rf $HF_HOME' EXIT
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_ext_sm${ARCH:-unknown}"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$HF_HOME"
echo "task ${SLURM_ARRAY_TASK_ID} node=$(hostname) sm_${ARCH} tmp_free=$(df -h /tmp | tail -1 | awk '{print $4}')"
uv run --no-sync python data_v10/multi_radius_full.py --task ${SLURM_ARRAY_TASK_ID} --n_tasks 11
