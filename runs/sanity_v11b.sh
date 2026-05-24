#!/bin/zsh
#SBATCH --time=00:20:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --output=runs/sanity_v11b_%j.out
#SBATCH --job-name=v11b_sanity
#SBATCH --gres=gg:g4:1

# GPU sanity for the fixed nerf_perfield encoder: builds the model, runs a fake
# forward under bf16 autocast (flash-attn), checks output shape + finiteness +
# gradient flow into every per-field projection, and that weight_transfer
# identifies the Gaussian-specific params.

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1

uv run --frozen python -m tmp.sanity_perfield
