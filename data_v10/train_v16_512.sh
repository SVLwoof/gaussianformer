#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --mem=200GB
#SBATCH --output=runs/train_v16_512_%j.out
#SBATCH --job-name=gformer_v16_512
#SBATCH --gres=gg:g4:8
#SBATCH --account=sagieb
#SBATCH --killable
#SBATCH --requeue

# V16 curriculum stage 2: refine the 256-trained model at 512^2. --init_from the v16_256 final
# -> fresh Phase 2 at 512 (Phase 1 auto-skipped). 12 epochs (was 3 in V15 -- the under-trained
# target resolution was a key V15 softness cause). Then a 512 LPIPS FT (train_v16_lpips.sh).
#
#   EDIT V16_256_CKPT below if needed, then: sbatch data_v10/train_v16_512.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

V16_256_CKPT=${V16_256_CKPT:-checkpoints_v16_256/phase2_epoch_20.pt}

# Stage to node-local RAM via parallel tar-extract of the prebuilt shards (data_v10/build_tars.sh).
SHM=/dev/shm/v16b_${SLURM_JOB_ID}
TARS=data_v10/tars
trap "rm -rf $SHM" EXIT
rm -rf $SHM; mkdir -p $SHM/h5s $SHM/h5s_val $SHM/renders $SHM/renders_val
echo "staging (parallel tar extract) -> $SHM ..."
pids=()
for t in $TARS/renders_[0-9]*.tar;     do tar -xf $t -C $SHM/renders     & pids+=($!); done
for t in $TARS/h5s_[0-9]*.tar;         do tar -xf $t -C $SHM/h5s         & pids+=($!); done
for t in $TARS/renders_val_[0-9]*.tar; do tar -xf $t -C $SHM/renders_val & pids+=($!); done
for t in $TARS/h5s_val_[0-9]*.tar;     do tar -xf $t -C $SHM/h5s_val     & pids+=($!); done
wait $pids
echo "staged: $(du -sh $SHM|cut -f1) | h5s=$(ls $SHM/h5s|wc -l) renders=$(ls $SHM/renders|wc -l) val_h5=$(ls $SHM/h5s_val|wc -l)"

# Resume own progress if any, else init from the 256 model.
files=( checkpoints_v16_512/phase2_epoch_*.pt(Nom) )
if [ ${#files} -gt 0 ]; then SEED=(--resume ${files[1]}); echo "RESUME ${files[1]}";
else SEED=(--init_from $V16_256_CKPT); echo "INIT from $V16_256_CKPT"; fi

uv run --frozen torchrun --standalone --nproc_per_node=8 -m training.train \
  --gaussian_h5_dir $SHM/h5s --renders_dir $SHM/renders \
  --save_dir checkpoints_v16_512 \
  --batch_size 1 --resolution 512 \
  --pe_type rope --augment_rotation --views_per_epoch 4 \
  --phase1_epochs 5 --phase1_lr 1e-3 \
  --phase2_epochs 12 --phase2_lr 5e-5 \
  --save_interval 1 --keep_last_n 3 \
  --val_h5_dir $SHM/h5s_val --val_renders_dir $SHM/renders_val \
  --log_loss_weight 1.0 --lpips_loss_weight 0.0 \
  --num_workers 8 $SEED
