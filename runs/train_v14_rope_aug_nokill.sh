#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 64
#SBATCH --mem=480GB
#SBATCH --output=runs/train_v14_rope_aug_nk_%j.out
#SBATCH --job-name=gformer_v14_nk
#SBATCH --gres=gg:g4:8

# NON-KILLABLE twin of runs/train_v14_rope_aug.sh -- same V14 (rope encoder + RoMa aug),
# same checkpoints_v14, same resume-aware launch (so it CONTINUES from phase1_epoch_3,
# no progress lost). Drops --killable/--requeue: normal-priority quota allocation, which
# is far more likely to actually grab 8xg4 than the lowest-priority killable queue. Use
# this when the killable job is stuck PENDING on Resources.
#
# IMPORTANT: only ONE V14 job may run against checkpoints_v14 at a time. Cancel the
# killable job (30706467) before/when submitting this, or they will clobber checkpoints.
#
#   sbatch runs/train_v14_rope_aug_nokill.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Resume-aware launch (zsh (Nom) null-glob array; files[1] = newest or empty).
files=( checkpoints_v14/phase2_epoch_*.pt(Nom) )
[ ${#files} -eq 0 ] && files=( checkpoints_v14/phase1_epoch_*.pt(Nom) )
LATEST=${files[1]}
# RESUME_ARG MUST be a zsh ARRAY (zsh doesn't word-split an unquoted scalar -- a string
# would reach argparse as one token "--resume X" and crash). Empty array -> nothing.
RESUME_ARG=()
if [ -n "$LATEST" ]; then
  RESUME_ARG=(--resume "$LATEST")
  echo "RESUME-AWARE: found $LATEST -> --resume $LATEST"
else
  echo "RESUME-AWARE: no checkpoint found -> fresh run from RenderFormer transfer"
fi

uv run --frozen torchrun --standalone --nproc_per_node=8 -m training.train \
  --gaussian_h5_dir data_v9_n20k/h5s --renders_dir data_v9/renders \
  --save_dir checkpoints_v14 \
  --batch_size 1 --resolution 512 \
  --pe_type rope --augment_rotation \
  --phase1_epochs 10 --phase1_lr 1e-3 \
  --phase2_epochs 20 --phase2_lr 5e-5 \
  --save_interval 1 --keep_last_n 3 \
  --val_h5_dir data_v9_n20k/h5s_val --val_renders_dir data_v9/renders_val \
  --log_loss_weight 1.0 --lpips_loss_weight 0.0 \
  --num_workers 8 $RESUME_ARG
