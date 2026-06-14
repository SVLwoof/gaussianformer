#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --mem=200GB
#SBATCH --output=runs/train_v15_512_%j.out
#SBATCH --job-name=gformer_v15_512
#SBATCH --gres=gg:g4:8
#SBATCH --killable
#SBATCH --requeue

# V15 curriculum stage 2: refine the 256-trained model at 512^2 (RenderFormer's 100k@512 step).
# --init_from the v15_256 final checkpoint -> fresh Phase 2 at 512 (Phase 1 auto-skipped).
# Short (high-freq refinement only). After this, a 512 LPIPS fine-tune mirrors V13b/V14auglp10.
#
#   EDIT V15_256_CKPT below to the chosen 256 checkpoint, then: sbatch data_v10/train_v15_512.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

V15_256_CKPT=${V15_256_CKPT:-checkpoints_v15_256/phase2_epoch_10.pt}

SHM=/dev/shm/v15b_${SLURM_JOB_ID}
trap "rm -rf $SHM" EXIT
rm -rf $SHM; mkdir -p $SHM
cp -r data_v10/h5s_20k_rec $SHM/h5s; cp -r data_v10/h5s_20k_rec_val $SHM/h5s_val
cp -r data_v10/renders $SHM/renders; cp -r data_v10/renders_val $SHM/renders_val
echo "staged: $(du -sh $SHM | cut -f1)"

# Resume own progress if any, else init from the 256 model.
files=( checkpoints_v15_512/phase2_epoch_*.pt(Nom) )
if [ ${#files} -gt 0 ]; then SEED=(--resume ${files[1]}); echo "RESUME ${files[1]}";
else SEED=(--init_from $V15_256_CKPT); echo "INIT from $V15_256_CKPT"; fi

uv run --frozen torchrun --standalone --nproc_per_node=8 -m training.train \
  --gaussian_h5_dir $SHM/h5s --renders_dir $SHM/renders \
  --save_dir checkpoints_v15_512 \
  --batch_size 1 --resolution 512 \
  --pe_type rope --augment_rotation \
  --phase1_epochs 5 --phase1_lr 1e-3 \
  --phase2_epochs 3 --phase2_lr 5e-5 \
  --save_interval 1 --keep_last_n 3 \
  --val_h5_dir $SHM/h5s_val --val_renders_dir $SHM/renders_val \
  --log_loss_weight 1.0 --lpips_loss_weight 0.0 \
  --num_workers 8 $SEED
