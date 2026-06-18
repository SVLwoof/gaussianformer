#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --mem=200GB
#SBATCH --output=runs/train_v16_256_%j.out
#SBATCH --job-name=gformer_v16_256
#SBATCH --gres=gg:g4:8
#SBATCH --account=sagieb
#SBATCH --killable
#SBATCH --requeue

# V16 = V15 recipe (5x data, recovered 20k, rope+aug, 256->512->LPIPS, K=4 view-subsampling,
# tar-staging) with a PROPERLY-SIZED budget. V15 was under-trained: phase-2 loss still
# descending, LPIPS 0.029, ~9x less per-object exposure than V14. The batch probe proved the
# fix can only be MORE total samples (compute-bound on the 20k attention -> bs>1 gives no
# throughput, 256 maxes at bs=2 / 512 at bs=1 on 46GB nodes; khan 96GB was unavailable).
# So V16 keeps bs=1 + K=4 (both free/optimal) and ~doubles the budget, weighted toward 512.
# Budget: 256 P1 5 + P2 20 (was 10); 512 P2 12 (was 3); LPIPS P2 12 (was 10).
#
#   sbatch data_v10/train_v16_256.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Stage to node-local RAM via parallel tar-extract of the prebuilt shards (data_v10/build_tars.sh).
SHM=/dev/shm/v16_${SLURM_JOB_ID}
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

# Resume-aware (zsh arrays; (Nom) = null-glob + mtime-newest). CRITICAL: never --resume a
# *phase1* checkpoint into phase2 -- that path builds a no-op phase1 DDP wrapper whose stale
# reducer hooks corrupt phase2 gradients (phase2 then "trains" without learning, val loss
# frozen -- observed 2026-06-18). Once phase1 is complete, seed phase2 cleanly with --init_from
# (skips the phase1 block entirely, the same path the LPIPS stage uses).
RESUME_ARG=()
p2=( checkpoints_v16_256/phase2_epoch_*.pt(Nom) )
if [ ${#p2} -gt 0 ]; then
  RESUME_ARG=(--resume ${p2[1]}); echo "RESUME phase2 from ${p2[1]}"
elif [ -f checkpoints_v16_256/phase1_epoch_5.pt ]; then   # phase1 done (== --phase1_epochs)
  RESUME_ARG=(--init_from checkpoints_v16_256/phase1_epoch_5.pt); echo "INIT phase2 (clean) from phase1_epoch_5"
else
  p1=( checkpoints_v16_256/phase1_epoch_*.pt(Nom) )
  [ ${#p1} -gt 0 ] && { RESUME_ARG=(--resume ${p1[1]}); echo "RESUME phase1 from ${p1[1]}"; }
fi

uv run --frozen torchrun --standalone --nproc_per_node=8 -m training.train \
  --gaussian_h5_dir $SHM/h5s --renders_dir $SHM/renders \
  --save_dir checkpoints_v16_256 \
  --batch_size 1 --resolution 256 \
  --pe_type rope --augment_rotation --views_per_epoch 4 \
  --phase1_epochs 5 --phase1_lr 1e-3 \
  --phase2_epochs 20 --phase2_lr 5e-5 \
  --save_interval 1 --keep_last_n 3 \
  --val_h5_dir $SHM/h5s_val --val_renders_dir $SHM/renders_val \
  --log_loss_weight 1.0 --lpips_loss_weight 0.0 \
  --num_workers 8 $RESUME_ARG
