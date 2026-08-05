#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --mem=256GB
#SBATCH --output=runs/train_v14_aug_phase2_%j.out
#SBATCH --job-name=gformer_v14aug
#SBATCH --gres=gg:g4:8
#SBATCH --killable
#SBATCH --requeue

# V14 WITH augmentation, retry: the no-aug run proved this recipe has a long ~10-epoch
# pre-breakout plateau, then descends fine (rope no-aug broke out at ep11 -> 31.49 dB).
# The earlier aug run was killed at ep2 -- prematurely, mistaking the plateau for failure.
# This time: aug ON, run to completion, and Phase 2 EXTENDED to 30 epochs because aug
# makes the task harder (likely later break-out) and the longer cosine keeps LR higher
# (~3.5e-5 at ep11 vs 2.1e-5 on the 20-epoch schedule) for a real post-breakout runway.
#
# Clean aug-isolation test: identical to V14na (rope, N=20k, log-L1) EXCEPT aug -> compare
# the final number to V14na's 31.49 / 0.0486 to read off augmentation's effect.
#
# Seed: the aug-warmed phase1_epoch_10 (skip Phase 1, fresh Phase 2). /dev/shm staging
# (9.4 GB) kills NFS I/O (3.0 -> 1.2 s/step). 8-GPU killable, khan excluded.
#
#   sbatch runs/train_v14_aug_phase2.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Two-tier resume-aware (zsh ARRAY): own Phase-2 progress, else seed the warmup.
files=( checkpoints_v14_aug/phase2_epoch_*.pt(Nom) )
if [ ${#files} -gt 0 ]; then
  LATEST=${files[1]}
  echo "RESUME-AWARE: continuing aug Phase 2 from $LATEST"
else
  LATEST="checkpoints_v14/phase1_epoch_10.pt"
  echo "RESUME-AWARE: seeding fresh aug Phase 2 from warmup $LATEST"
fi
RESUME_ARG=(--resume "$LATEST")

# Stage dataset to RAM (/dev/shm) to eliminate NFS I/O. Per-node, cleared on job end.
SHM=/dev/shm/v14aug_${SLURM_JOB_ID}
trap "rm -rf $SHM" EXIT
mkdir -p $SHM
echo "staging dataset -> $SHM ..."
cp -r data_v9_n20k/h5s      $SHM/h5s
cp -r data_v9_n20k/h5s_val  $SHM/h5s_val
cp -r data_v9/renders       $SHM/renders
cp -r data_v9/renders_val   $SHM/renders_val
echo "staging done: $(du -sh $SHM | cut -f1) in $SHM"

uv run --frozen torchrun --standalone --nproc_per_node=8 -m training.train \
  --gaussian_h5_dir $SHM/h5s --renders_dir $SHM/renders \
  --save_dir checkpoints_v14_aug \
  --batch_size 1 --resolution 512 \
  --pe_type rope --augment_rotation \
  --phase1_epochs 10 --phase1_lr 1e-3 \
  --phase2_epochs 30 --phase2_lr 5e-5 \
  --save_interval 1 --keep_last_n 3 \
  --val_h5_dir $SHM/h5s_val --val_renders_dir $SHM/renders_val \
  --log_loss_weight 1.0 --lpips_loss_weight 0.0 \
  --num_workers 8 $RESUME_ARG
