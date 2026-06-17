#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --mem=200GB
#SBATCH --output=runs/train_v16_lpips_%j.out
#SBATCH --job-name=gformer_v16_lpips
#SBATCH --gres=gg:g4:8
#SBATCH --account=sagieb
#SBATCH --killable
#SBATCH --requeue

# V16 curriculum stage 3: LPIPS fine-tune the 512-refined model (mirrors V13b / v14auglp10).
# --init_from the v16_512 final -> weights-only warm start -> fresh Phase 2 at 512, fresh cosine
# from phase2_lr (Phase 1 auto-skipped). Deltas vs stage 2: lpips_loss_weight 0.2, 12 epochs,
# keep_last_n 12 (every epoch an eval candidate), separate save_dir.
#
#   EDIT V16_512_CKPT below if needed, then: sbatch data_v10/train_v16_lpips.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

V16_512_CKPT=${V16_512_CKPT:-checkpoints_v16_512/phase2_epoch_12.pt}

# Stage to node-local RAM via parallel tar-extract of the prebuilt shards (data_v10/build_tars.sh).
SHM=/dev/shm/v16lp_${SLURM_JOB_ID}
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

# Resume own progress if any, else init from the 512 model.
files=( checkpoints_v16_lpips/phase2_epoch_*.pt(Nom) )
if [ ${#files} -gt 0 ]; then SEED=(--resume ${files[1]}); echo "RESUME ${files[1]}";
else SEED=(--init_from $V16_512_CKPT); echo "INIT from $V16_512_CKPT"; fi

uv run --frozen torchrun --standalone --nproc_per_node=8 -m training.train \
  --gaussian_h5_dir $SHM/h5s --renders_dir $SHM/renders \
  --save_dir checkpoints_v16_lpips \
  --batch_size 1 --resolution 512 \
  --pe_type rope --augment_rotation --views_per_epoch 4 \
  --phase1_epochs 5 --phase1_lr 1e-3 \
  --phase2_epochs 12 --phase2_lr 5e-5 \
  --save_interval 1 --keep_last_n 12 \
  --val_h5_dir $SHM/h5s_val --val_renders_dir $SHM/renders_val \
  --log_loss_weight 1.0 --lpips_loss_weight 0.2 \
  --num_workers 8 $SEED
