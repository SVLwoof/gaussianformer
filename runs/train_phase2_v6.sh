#!/bin/zsh
#SBATCH --time=72:00:00
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --output=runs/train_phase2_v6_%j.out
#SBATCH --job-name=gformer_phase2_v6
#SBATCH --mail-type=END,FAIL,BEGIN
# #SBATCH --mail-user=your-email@example.com
#SBATCH --gres=gg:g4:1

# v6 = v4's simple log-HDR loss on the bumped-N (5000 vs 3000) dataset.
# Resumes from v4's phase1 checkpoint to skip the 20-epoch backbone
# warmup; RoPE handles the seqlen jump without issues. Expected to run
# slower than v5 (~25 min/epoch vs 12) due to O(N^2) attention growth
# even with flash-attn.

module load nvidia
module load cuda
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1

python -m training.train \
  --gaussian_h5_dir data_v2/h5s_n5k \
  --renders_dir data_v2/renders_n5k \
  --save_dir checkpoints_v6 \
  --batch_size 2 --resolution 512 \
  --skip_phase1 \
  --resume checkpoints_v4/phase1_epoch_20.pt \
  --phase2_epochs 100 \
  --phase2_lr 5e-5 \
  --save_interval 5 \
  --val_h5_dir data_v2/h5s_n5k_val \
  --val_renders_dir data_v2/renders_n5k_val
