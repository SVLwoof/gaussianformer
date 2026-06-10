#!/bin/zsh
#SBATCH --time=00:20:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --output=runs/test_rotation_aug_%j.out
#SBATCH --job-name=test_rot_aug
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue

# V14 augmentation correctness gate: rasterize a scene unrotated vs rotated-scene+camera
# and assert the images match (PSNR > 40 dB). Must pass before V14 training.
#
#   sbatch runs/test_rotation_aug.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1

uv run --frozen python -m tmp.test_rotation_aug_invariance
