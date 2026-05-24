#!/bin/zsh
#SBATCH --time=00:30:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --output=runs/probe_bs1_n_%j.out
#SBATCH --job-name=probe_bs1_n
#SBATCH --gres=gg:g4:1

# One-off probe: peak GPU memory at bs=1 across N in {5k, 7.5k, ..., 30k} on
# a g4 card with the V12 (pe_type=nerf) encoder, 512x512 res, AdamW + bf16
# autocast. The python script catches OOM per-N and continues, so the output
# shows where the ceiling actually lands.
#
# Submit:
#   sbatch runs/probe_bs1_n.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run --frozen python -m tmp.probe_bs1_n_scaling
