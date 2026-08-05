#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --mem=200GB
#SBATCH --output=runs/train_v17_256_%j.out
#SBATCH --job-name=gformer_v17_256
#SBATCH --gres=gg:g4:8
#SBATCH --account=sagieb
#SBATCH --killable
#SBATCH --requeue

# V17 stage 1/2 -- the LONG 256 bulk stage. 60 phase-2 epochs (V16 ran 20).
#
# WHY LONGER: every V16 stage was still descending monotonically at its final epoch (256 ep20
# 0.001034, 512 ep12 0.000956, LPIPS ep12 0.017893) -- V16 was budget-cut, never converged. And
# the budget never needed rationing: with --views_per_epoch 4 an epoch is 53,620 samples (not
# 187,670), so 256 ran at ~23 min/epoch and the WHOLE V16 chain was ~15 h, not the "5-7 days" the
# compute-bound analysis feared. V16's chain = ~12.5 passes over the data; V14 got ~30 passes over
# its 3k objects -> V16 saw each object-view less than half as often, on 4.5x more objects. The
# "generalisation gap" may just be an under-training gap. 60 epochs ~ 23 h.
#
# Cosine T_max = phase2_epochs, so the LR schedule stretches with the budget (no code change).
# Loss stays pure log-L1 here: this is the cheap bulk stage that learns geometry/colour. The
# perceptual pressure goes into the 512 stage (train_v17_512lp.sh), applied THROUGHOUT rather than
# bolted on as a tail FT.
#
#   sbatch data_v10/train_v17_256.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Stage to node-local RAM via parallel tar-extract of the prebuilt shards (data_v10/build_tars.sh).
SHM=/dev/shm/v17_${SLURM_JOB_ID}
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

# Seed phase2 from V15's KNOWN-GOOD phase1 warmup and skip our own phase1 (--init_from goes
# straight to a clean phase2). V16's own phase1 silently failed to train (val 0.0135 vs V15's
# 0.0038) and starved phase2; V15's phase1_epoch_5 is the same arch/data/recipe and is the warmup
# V16/V17's phase1 should have produced. Once our own phase2 has checkpointed, resume from that.
SEED_CKPT=checkpoints_v15_256/phase1_epoch_5.pt
RESUME_ARG=()
p2=( checkpoints_v17_256/phase2_epoch_*.pt(Nom) )
if [ ${#p2} -gt 0 ]; then
  RESUME_ARG=(--resume ${p2[1]}); echo "RESUME phase2 from ${p2[1]}"
else
  RESUME_ARG=(--init_from $SEED_CKPT); echo "INIT phase2 (clean) from $SEED_CKPT"
fi

uv run --frozen torchrun --standalone --nproc_per_node=8 -m training.train \
  --gaussian_h5_dir $SHM/h5s --renders_dir $SHM/renders \
  --save_dir checkpoints_v17_256 \
  --batch_size 1 --resolution 256 \
  --pe_type rope --augment_rotation --views_per_epoch 4 \
  --phase1_epochs 5 --phase1_lr 1e-3 \
  --phase2_epochs 60 --phase2_lr 5e-5 \
  --save_interval 1 --keep_last_n 3 \
  --val_h5_dir $SHM/h5s_val --val_renders_dir $SHM/renders_val \
  --log_loss_weight 1.0 --lpips_loss_weight 0.0 \
  --num_workers 8 $RESUME_ARG
