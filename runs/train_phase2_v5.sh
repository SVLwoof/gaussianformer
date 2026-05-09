#!/bin/zsh
#SBATCH --time=72:00:00
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --output=runs/train_phase2_v5_%j.out
#SBATCH --job-name=gformer_phase2_v5
#SBATCH --mail-type=END,FAIL,BEGIN
# #SBATCH --mail-user=your-email@example.com
#SBATCH --gres=gg:g4:1

module load nvidia
module load cuda
cd "$(dirname "$0")/.."
source .venv/bin/activate

export PYTHONUNBUFFERED=1

# v5 differs from v4 in two things only: the inverted-sRGB L1 loss term is
# enabled (the perceptual signal v4 was missing), and flash-attn + bf16
# autocast are now active (transparent — no flag needed). Same data, same
# resume point, same epoch budget — this is the loss-only ablation.
python -m training.train \
  --gaussian_h5_dir data_v2/h5s \
  --renders_dir data_v2/renders \
  --save_dir checkpoints_v5 \
  --batch_size 2 --resolution 512 \
  --skip_phase1 \
  --resume checkpoints_v4/phase1_epoch_20.pt \
  --phase2_epochs 100 \
  --phase2_lr 5e-5 \
  --save_interval 5 \
  --val_h5_dir data_v2/h5s_val \
  --val_renders_dir data_v2/renders_val
