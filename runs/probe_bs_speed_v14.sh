#!/bin/zsh
#SBATCH --time=00:30:00
#SBATCH -c 64
#SBATCH --mem=480GB
#SBATCH --output=runs/probe_bs_speed_v14_%j.out
#SBATCH --job-name=probe_bs_v14
#SBATCH --gres=gg:g4:8
#SBATCH --killable
#SBATCH --requeue

# V14 batch-size decision: measure true 8-GPU DDP throughput (compute + comms) of the
# Phase-2 path at N=20k, pe_type=rope, for bs=1 and bs=2. Pick the faster wall/epoch
# that fits memory. If bs=2 wins, scale LRs by sqrt(2) for the 2x effective batch.
#
#   sbatch runs/probe_bs_speed_v14.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run --frozen torchrun --standalone --nproc_per_node=8 -m tmp.probe_bs_speed
