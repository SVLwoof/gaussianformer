#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --mem=256GB
#SBATCH --output=runs/train_v14auglp_%j.out
#SBATCH --job-name=gformer_v14auglp
# No --mail-type: killable jobs requeue often and BEGIN spams on every restart.
#SBATCH --gres=gg:g4:8
#SBATCH --killable
#SBATCH --requeue

# V14aug-LPIPS: LPIPS fine-tune of V14aug (rope+aug, ep20, our best-PSNR model at
# 33.70 dB / 0.0319 LPIPS), mirroring V13->V13b. Goal: beat V13b (32.83 / 0.0209) on
# BOTH axes -- V14aug already leads PSNR by ~0.9 dB; an LPIPS FT should drop LPIPS below
# 0.02. Aug stays ON (matched to the aug-trained base -> no plateau). 5 epochs, fresh
# cosine from 5e-5, lpips_w 0.2 (the V9->V10b / V13->V13b setting). save_interval=1 so we
# can eval each epoch and take the best (V13b peaked at ep4 of 5).
#
#   sbatch runs/train_v14auglp.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Seed: own FT progress (--resume) if any, else fresh FT from V14aug ep20 (--init_from:
# weights only, fresh Phase 2 + new cosine + LPIPS loss). zsh ARRAY (no scalar word-split).
files=( checkpoints_v14aug_lpips/phase2_epoch_*.pt(Nom) )
if [ ${#files} -gt 0 ]; then
  SEED_ARG=(--resume ${files[1]})
  echo "RESUME-AWARE: continuing LPIPS FT from ${files[1]}"
else
  SEED_ARG=(--init_from checkpoints_v14_aug/phase2_epoch_20.pt)
  echo "RESUME-AWARE: fresh LPIPS FT --init_from V14aug ep20"
fi

# Stage dataset to RAM. rm -rf first so repeated preemptions can't nest/accumulate
# (SIGKILL skips the EXIT trap, so a prior run's staging may linger on the same node).
SHM=/dev/shm/v14auglp_${SLURM_JOB_ID}
trap "rm -rf $SHM" EXIT
rm -rf $SHM; mkdir -p $SHM
echo "staging dataset -> $SHM ..."
cp -r data_v9_n20k/h5s      $SHM/h5s
cp -r data_v9_n20k/h5s_val  $SHM/h5s_val
cp -r data_v9/renders       $SHM/renders
cp -r data_v9/renders_val   $SHM/renders_val
echo "staging done: $(du -sh $SHM | cut -f1) in $SHM"

uv run --frozen torchrun --standalone --nproc_per_node=8 -m training.train \
  --gaussian_h5_dir $SHM/h5s --renders_dir $SHM/renders \
  --save_dir checkpoints_v14aug_lpips \
  --batch_size 1 --resolution 512 \
  --pe_type rope --augment_rotation \
  --phase1_epochs 10 --phase1_lr 1e-3 \
  --phase2_epochs 5 --phase2_lr 5e-5 \
  --save_interval 1 --keep_last_n 3 \
  --val_h5_dir $SHM/h5s_val --val_renders_dir $SHM/renders_val \
  --log_loss_weight 1.0 --lpips_loss_weight 0.2 \
  --num_workers 8 $SEED_ARG
