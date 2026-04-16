#!/bin/zsh
#SBATCH --time=72:00:00
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --output=runs/train_phase2_%j.out
#SBATCH --job-name=gformer_phase2
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=shahaf.valenciale@mail.huji.ac.il
#SBATCH --gres=gg:g4:1

module load nvidia
module load cuda
cd /cs/labs/tomhope/shahaf_levy/gaussianformer
source .venv/bin/activate

export PYTHONUNBUFFERED=1

python -m training.train \
  --gaussian_h5_dir data_v2/h5s \
  --renders_dir data_v2/renders \
  --save_dir checkpoints_v4 \
  --batch_size 2 --resolution 512 \
  --skip_phase1 \
  --resume checkpoints_v4/phase1_epoch_20.pt \
  --phase2_epochs 100 \
  --phase2_lr 5e-5 \
  --save_interval 5 \
  --val_h5_dir data_v2/h5s_val \
  --val_renders_dir data_v2/renders_val
