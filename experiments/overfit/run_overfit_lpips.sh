#!/bin/zsh
#SBATCH --time=72:00:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --job-name=ovf_lpips
#SBATCH --output=runs/ovf_lpips_%x_%j.out
#SBATCH --gres=gg:g4:1
#SBATCH --account=sagieb
#SBATCH --killable
#SBATCH --requeue

# CAPACITY PROBE v2: can a DETAIL-oriented loss pull fine texture out of the model?
# Overfit ONE detailed data_v10 object (default skull scene_9869) on its 14 views, starting
# from the V16-512 checkpoint, varying ONLY the loss. Run two jobs (L1 vs high-LPIPS) to
# isolate whether perceptual loss recovers the fine engravings the recovered-20k input clearly
# carries (~40 dB foreground) but the L1 model smears away. Uses the RECOVERED 20k h5 (V16's
# training-matched input). Single object -> single GPU, bs=1 = most update steps.
#
#   for L in L1 hi; do
#     [ $L = hi ] && W="LOG_W=0.5 LPIPS_W=1.0" || W="LOG_W=1.0 LPIPS_W=0.0"
#     sbatch --job-name=ovf9869_$L --export=ALL,TAG=$L,$W experiments/overfit/run_overfit_lpips.sh
#   done

set -e
source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

SCENE=${SCENE:-9869}
LOG_W=${LOG_W:-0.5}
LPIPS_W=${LPIPS_W:-1.0}
TAG=${TAG:-hi}
SEED_CKPT=${SEED_CKPT:-checkpoints_v16_512/phase2_epoch_12.pt}
S=$(printf "%04d" $SCENE)

# Build single-object h5 + renders dirs (symlinks; the dataset globs a directory).
WORK=experiments/overfit/data/v10_${S}
mkdir -p $WORK/h5s $WORK/renders
ln -sf $(realpath data_v10/h5s_20k_rec/scene_${S}.h5) $WORK/h5s/scene_${S}.h5
for r in data_v10/renders/scene_${S}_view_*.png; do ln -sf $(realpath $r) $WORK/renders/$(basename $r); done
echo "object scene_$S: $(ls $WORK/h5s/*.h5|wc -l) h5, $(ls $WORK/renders/*.png|wc -l) renders"

SAVEDIR=experiments/overfit/ckpt/v10_${S}_${TAG}
mkdir -p $SAVEDIR
echo "TAG=$TAG LOG_W=$LOG_W LPIPS_W=$LPIPS_W SAVEDIR=$SAVEDIR seed=$SEED_CKPT"

# Resume own progress if any, else warm-start from V16-512 (--init_from skips Phase 1 -> overfit
# is pure Phase-2 joint fine-tune; same path V16's LPIPS stage uses).
files=( $SAVEDIR/phase2_epoch_*.pt(Nom) )
if [ ${#files} -gt 0 ]; then SEED=(--resume ${files[1]}); echo "resume ${files[1]}";
else SEED=(--init_from $SEED_CKPT); echo "init_from $SEED_CKPT"; fi

# 14 samples/epoch. 600 ep ~ 8400 steps -- plenty to overfit one object from a good init.
uv run --frozen torchrun --standalone --nproc_per_node=1 -m training.train \
  --gaussian_h5_dir $WORK/h5s --renders_dir $WORK/renders \
  --save_dir $SAVEDIR \
  --batch_size 1 --resolution 512 \
  --pe_type rope \
  --phase1_epochs 50 --phase1_lr 1e-3 \
  --phase2_epochs 600 --phase2_lr 5e-5 \
  --save_interval 100 --keep_last_n 4 \
  --log_loss_weight $LOG_W --lpips_loss_weight $LPIPS_W \
  --num_workers 4 $SEED
