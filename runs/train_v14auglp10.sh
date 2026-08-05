#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --mem=256GB
#SBATCH --output=runs/train_v14auglp10_%j.out
#SBATCH --job-name=gformer_v14auglp10
# No --mail-type (killable requeue spams BEGIN).
#SBATCH --gres=gg:g4:8
#SBATCH --killable
#SBATCH --requeue

# V14aug-LPIPS-10: the 5-epoch FT (ep5 = 33.57/0.0195, new best, beat V13b on both axes)
# was STILL climbing at its final epoch (PSNR +0.6, LPIPS -0.001 ep4->ep5) -- the cosine
# cut it off mid-ascent. This re-runs the LPIPS FT from V14aug ep20 with a 10-EPOCH cosine
# (slower anneal -> more runway) to push PSNR past the base's 33.70 and LPIPS lower still.
# Identical otherwise: rope+aug, lpips_w 0.2, fresh cosine from 5e-5, --init_from V14aug ep20.
#
#   sbatch runs/train_v14auglp10.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Seed: own progress (--resume) if any, else fresh FT from V14aug ep20 (--init_from).
files=( checkpoints_v14auglp10/phase2_epoch_*.pt(Nom) )
if [ ${#files} -gt 0 ]; then
  SEED_ARG=(--resume ${files[1]})
  echo "RESUME-AWARE: continuing 10-ep LPIPS FT from ${files[1]}"
else
  SEED_ARG=(--init_from checkpoints_v14_aug/phase2_epoch_20.pt)
  echo "RESUME-AWARE: fresh 10-ep LPIPS FT --init_from V14aug ep20"
fi

SHM=/dev/shm/v14auglp10_${SLURM_JOB_ID}
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
  --save_dir checkpoints_v14auglp10 \
  --batch_size 1 --resolution 512 \
  --pe_type rope --augment_rotation \
  --phase1_epochs 10 --phase1_lr 1e-3 \
  --phase2_epochs 10 --phase2_lr 5e-5 \
  --save_interval 1 --keep_last_n 3 \
  --val_h5_dir $SHM/h5s_val --val_renders_dir $SHM/renders_val \
  --log_loss_weight 1.0 --lpips_loss_weight 0.2 \
  --num_workers 8 $SEED_ARG
