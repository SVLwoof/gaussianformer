#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --mem=200GB
#SBATCH --output=runs/train_v15_lpips_%j.out
#SBATCH --job-name=gformer_v15_lpips
#SBATCH --gres=gg:g4:8
#SBATCH --account=sagieb
#SBATCH --killable
#SBATCH --requeue

# V15 curriculum stage 3: LPIPS fine-tune the 512-refined model (mirrors V13b / v14auglp10 --
# the stage that produced every prior best checkpoint; v14auglp10 ep10 = 33.835 dB / 0.0185,
# still climbing at ep10). --init_from the v15_512 final -> weights-only warm start -> fresh
# Phase 2 at 512 with a fresh cosine from phase2_lr (Phase 1 auto-skipped by --init_from).
# Only deltas vs stage 2 (train_v15_512.sh): lpips_loss_weight 0.2, phase2_epochs 10,
# keep_last_n 10 (every epoch is an eval candidate), separate save_dir.
#
#   EDIT V15_512_CKPT below to the chosen 512 checkpoint, then: sbatch data_v10/train_v15_lpips.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

V15_512_CKPT=${V15_512_CKPT:-checkpoints_v15_512/phase2_epoch_3.pt}

SHM=/dev/shm/v15lp_${SLURM_JOB_ID}
trap "rm -rf $SHM" EXIT
rm -rf $SHM; mkdir -p $SHM
cp -r data_v10/h5s_20k_rec $SHM/h5s; cp -r data_v10/h5s_20k_rec_val $SHM/h5s_val
cp -r data_v10/renders $SHM/renders; cp -r data_v10/renders_val $SHM/renders_val
echo "staged: $(du -sh $SHM | cut -f1)"

# Resume own progress if any, else init from the 512 model.
files=( checkpoints_v15_lpips/phase2_epoch_*.pt(Nom) )
if [ ${#files} -gt 0 ]; then SEED=(--resume ${files[1]}); echo "RESUME ${files[1]}";
else SEED=(--init_from $V15_512_CKPT); echo "INIT from $V15_512_CKPT"; fi

uv run --frozen torchrun --standalone --nproc_per_node=8 -m training.train \
  --gaussian_h5_dir $SHM/h5s --renders_dir $SHM/renders \
  --save_dir checkpoints_v15_lpips \
  --batch_size 1 --resolution 512 \
  --pe_type rope --augment_rotation \
  --phase1_epochs 5 --phase1_lr 1e-3 \
  --phase2_epochs 10 --phase2_lr 5e-5 \
  --save_interval 1 --keep_last_n 10 \
  --val_h5_dir $SHM/h5s_val --val_renders_dir $SHM/renders_val \
  --log_loss_weight 1.0 --lpips_loss_weight 0.2 \
  --num_workers 8 $SEED
