#!/bin/zsh
#SBATCH --time=00:30:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --output=runs/probe_bs2_n_%j.out
#SBATCH --job-name=probe_bs2_n
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run --frozen python -m tmp.probe_bs2_n_scaling
