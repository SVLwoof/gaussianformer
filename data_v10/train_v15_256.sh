#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --mem=200GB
#SBATCH --output=runs/train_v15_256_%j.out
#SBATCH --job-name=gformer_v15_256
#SBATCH --gres=gg:g4:8
#SBATCH --account=sagieb
#SBATCH --killable
#SBATCH --requeue

# V15 = 5x data (data_v10, recovered 20k) + rope + RoMa aug, trained at 256^2 (curriculum
# stage 1; a 512 refine + LPIPS FT follow -- see train_v15_512.sh). From RenderFormer transfer.
#
# WHY 256 first: RenderFormer's curriculum (500k @256 then 100k @512). At 5x data the bulk is
# expensive; 256 is ~2x cheaper/step (the view-dependent/decoder half; the 20k-token encoder is
# resolution-independent) -> afford the data. Detail is refined in the 512 stage.
# DATA: recovered 20k (data_v10/h5s_20k_rec, LightGaussian prune+recovery -> held-out pruned-GT
# 35->49.5 dB) -- the model's input ceiling is now ~14 dB higher than the old naive prune.
# Epoch budget SCALED DOWN for 5x data: ~5x more samples/epoch, so far fewer epochs than V14.
#
# bs=1 x 8 GPU = eff batch 8 (== V14) -> phase1_lr 1e-3 / phase2_lr 5e-5 carry over.
# Resume-aware + /dev/shm staging (66 GB recovered-h5s+renders -> needs mem>=~100GB).
# PRE-WARM gsplat cache is N/A here (training doesn't use gsplat); but for any gsplat swarm see
# the pre-warm note. Launch ONLY after step-A recovery completes (h5s_20k_rec{,_val} populated).
#
#   sbatch data_v10/train_v15_256.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Stage recovered 20k h5s + GT renders to node-local RAM (kills NFS I/O at 5x scale).
SHM=/dev/shm/v15_${SLURM_JOB_ID}
trap "rm -rf $SHM" EXIT
rm -rf $SHM; mkdir -p $SHM
echo "staging recovered data -> $SHM ..."
cp -r data_v10/h5s_20k_rec      $SHM/h5s
cp -r data_v10/h5s_20k_rec_val  $SHM/h5s_val
cp -r data_v10/renders          $SHM/renders
cp -r data_v10/renders_val      $SHM/renders_val
echo "staged: $(du -sh $SHM | cut -f1)"

# Resume-aware (zsh array; (Nom) = null-glob + mtime-newest).
files=( checkpoints_v15_256/phase2_epoch_*.pt(Nom) checkpoints_v15_256/phase1_epoch_*.pt(Nom) )
RESUME_ARG=()
[ ${#files} -gt 0 ] && { RESUME_ARG=(--resume ${files[1]}); echo "RESUME from ${files[1]}"; }

uv run --frozen torchrun --standalone --nproc_per_node=8 -m training.train \
  --gaussian_h5_dir $SHM/h5s --renders_dir $SHM/renders \
  --save_dir checkpoints_v15_256 \
  --batch_size 1 --resolution 256 \
  --pe_type rope --augment_rotation \
  --phase1_epochs 5 --phase1_lr 1e-3 \
  --phase2_epochs 10 --phase2_lr 5e-5 \
  --save_interval 1 --keep_last_n 3 \
  --val_h5_dir $SHM/h5s_val --val_renders_dir $SHM/renders_val \
  --log_loss_weight 1.0 --lpips_loss_weight 0.0 \
  --num_workers 8 $RESUME_ARG
