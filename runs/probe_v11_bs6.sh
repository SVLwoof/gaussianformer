#!/bin/zsh
#SBATCH --time=00:30:00
#SBATCH -c 32
#SBATCH --mem=128GB
#SBATCH --output=runs/probe_v11_bs6_%j.out
#SBATCH --job-name=probe_v11_bs6
#SBATCH --gres=gg:g4:4

# V11 VRAM probe at bs=4 (V10b parity). 1 phase-2 epoch on max_samples=32 to
# saturate steady-state attention memory without committing to a full run.
# nvidia-smi sidecar logs peak memory.used per GPU every 10s.

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1

SMI_LOG="runs/probe_v11_bs6_${SLURM_JOB_ID}_smi.log"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv -l 10 > "$SMI_LOG" &
SMI_PID=$!
trap "kill $SMI_PID 2>/dev/null" EXIT

uv run --frozen torchrun --standalone --nproc_per_node=4 -m training.train \
  --gaussian_h5_dir data_v9/h5s \
  --renders_dir data_v9/renders \
  --save_dir checkpoints_probe_v11_bs6 \
  --batch_size 6 --resolution 512 \
  --pe_type nerf_perfield \
  --scale_pe_num_freqs 6 \
  --skip_phase1 \
  --phase2_epochs 1 \
  --phase2_lr 5e-5 \
  --save_interval 999 \
  --max_samples 32 \
  --log_loss_weight 1.0 \
  --lpips_loss_weight 0.0 \
  --num_workers 8
