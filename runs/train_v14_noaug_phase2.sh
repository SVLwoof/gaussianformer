#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --mem=256GB
#SBATCH --output=runs/train_v14_noaug_phase2_%j.out
#SBATCH --job-name=gformer_v14na
#SBATCH --gres=gg:g4:8
#SBATCH --killable
#SBATCH --requeue
#SBATCH --exclude=khan-01,khan-02

# V14 no-aug rescue: the augmented Phase 2 collapsed to BLACK output at the unfreeze
# (loss pinned at the 0.0138 black-on-mostly-black-bg value; render confirmed empty).
# Phase 1 (with aug) was fine -> reuse that warmup, run Phase 2 WITHOUT augmentation.
# This is also the clean rope@N=20k control. If it descends like V9, aug caused the
# collapse and we have a working model.
#
# SEED: resumes from checkpoints_v14/phase1_epoch_10.pt (aug-warmed encoder, backbone
# frozen). --resume on a phase1 checkpoint at epoch==phase1_epochs makes the Phase-1 loop
# empty (range(10,10)) and drops straight into a FRESH Phase 2 (new optimizer + cosine
# from 5e-5). New checkpoints go to checkpoints_v14_noaug/ (keeps the warmup pristine).
#
# 8-GPU KILLABLE (firefoot l40s ~72 min/epoch, 2x faster than 4-GPU non-killable on l40s
# now that khan is full). khan excluded (aggressive preemption). Resume-aware + atomic
# saves + keep_last_n=3 -> a preemption costs <=1 epoch and self-resumes.
#
#   sbatch runs/train_v14_noaug_phase2.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Two-tier resume-aware (zsh ARRAY -- zsh doesn't word-split unquoted scalars):
#  1) our own no-aug Phase-2 progress, if any (preemption recovery);
#  2) else seed from the aug-warmed Phase-1 ep10 (skips Phase 1, fresh Phase 2).
files=( checkpoints_v14_noaug/phase2_epoch_*.pt(Nom) )
if [ ${#files} -gt 0 ]; then
  LATEST=${files[1]}
  echo "RESUME-AWARE: continuing no-aug Phase 2 from $LATEST"
else
  LATEST="checkpoints_v14/phase1_epoch_10.pt"
  echo "RESUME-AWARE: seeding fresh no-aug Phase 2 from warmup $LATEST"
fi
RESUME_ARG=(--resume "$LATEST")

# Stage the dataset (9.4 GB) to node-local RAM (/dev/shm tmpfs) to kill NFS I/O: the
# synthetic-data probe was 0.93 s/step but real NFS-backed loading ran 3.0 s/step
# (GPUs I/O-starved ~2/3 of the time). Both H5s AND GT render images are read every
# step, so stage all four. /dev/shm is per-node and cleared on job end, so on a
# requeue this simply re-copies (~3 min) -- checkpoints live on NFS (--save_dir) and
# persist independently. Job-specific path avoids cross-job collisions.
SHM=/dev/shm/v14_${SLURM_JOB_ID}
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
  --save_dir checkpoints_v14_noaug \
  --batch_size 1 --resolution 512 \
  --pe_type rope \
  --phase1_epochs 10 --phase1_lr 1e-3 \
  --phase2_epochs 20 --phase2_lr 5e-5 \
  --save_interval 1 --keep_last_n 3 \
  --val_h5_dir $SHM/h5s_val --val_renders_dir $SHM/renders_val \
  --log_loss_weight 1.0 --lpips_loss_weight 0.0 \
  --num_workers 8 $RESUME_ARG
