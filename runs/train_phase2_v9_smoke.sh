#!/bin/zsh
#SBATCH --time=2:00:00
#SBATCH -c 16
#SBATCH --mem=96GB
#SBATCH --output=runs/train_phase2_v9_smoke_%j.out
#SBATCH --job-name=v9_smoke
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gg:g4:2

# DDP smoke test: 2 GPUs via torchrun, 50 samples, 1 epoch.
# Verifies: process group init, DistributedSampler shards correctly, DDP
# all-reduce works, val metric aggregation, rank-0-gated checkpoint save.
# Will produce one phase2_epoch_1.pt under /tmp/v9_smoke_ckpt; then deletes.

source "${LMOD_INIT:-/etc/profile.d/lmod.sh}"
module load nvidia
module load cuda
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1

# torchrun env: torchrun sets WORLD_SIZE/RANK/LOCAL_RANK per process.
torchrun --standalone --nproc_per_node=2 -m training.train \
  --gaussian_h5_dir data_v9/h5s \
  --renders_dir data_v9/renders \
  --val_h5_dir data_v9/h5s_val \
  --val_renders_dir data_v9/renders_val \
  --save_dir /tmp/v9_smoke_ckpt \
  --batch_size 2 --resolution 512 \
  --skip_phase1 \
  --resume checkpoints_v4/phase1_epoch_20.pt \
  --phase2_epochs 1 \
  --phase2_lr 5e-5 \
  --save_interval 1 \
  --max_samples 50 \
  --log_loss_weight 1.0 \
  --lpips_loss_weight 0.0
