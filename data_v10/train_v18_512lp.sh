#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --mem=200GB
#SBATCH --output=runs/train_v18_512lp_%j.out
#SBATCH --job-name=gformer_v18_512lp
#SBATCH --gres=gg:g4:4
#SBATCH --account=sagieb
#SBATCH --requeue

# V17 stage 2/2 -- 512 refine with LPIPS applied THROUGHOUT (log_w 0.5 / lpips_w 0.5), 36 epochs.
#
# Two changes vs V16, both evidence-backed:
#  1. LPIPS is IN this stage from epoch 1, not bolted on as a 12-epoch tail FT. A tail-FT can only
#     polish sharpness onto a model that already learned to blur; training with the perceptual term
#     should let it learn the sharp mapping instead of learning-then-correcting.
#  2. lpips_w 0.5 (not 1.0). The V16 sweep settled this: FG-cropped on 3 unseen objects, mid(0.5)
#     >= hi(1.0) on BOTH PSNR and LPIPS on EVERY scene -> cranking past 0.5 is pure PSNR cost.
# 36 epochs = 3x V16's 512 budget (V16's 512 val was still descending at ep12). ~14-20 h.
#
# GATE THIS RUN ON THE LPIPS VAL TERM + periodic `--crop_fg` renders, NOT on log-L1. V4/V5/V6
# precedent: log-L1 improved repeatedly while perceptual quality did not move, and its cousin
# (whole-image PSNR) already fooled us once on this exact data.
#
#   sbatch --dependency=afterok:<v18_256 jobid> data_v10/train_v18_512lp.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Standard setup: read directly from NFS, NO /dev/shm tar staging (doesn't scale as data grows).
GH5=data_v10/h5s_20k_rec;     REN=data_v10/renders
VH5=data_v10/h5s_20k_rec_val; VREN=data_v10/renders_val
echo "data (direct NFS): h5s=$(ls $GH5|wc -l) renders=$(ls $REN|wc -l) val_h5=$(ls $VH5|wc -l)"

# Resume own progress if any, else init from the NEWEST v18_256 phase2 checkpoint (glob, not a
# hardcoded epoch -- keep_last_n prunes old ones and the exact final epoch can shift on requeue).
RESUME_ARG=()
p2=( checkpoints_v18_512lp/phase2_epoch_*.pt(Nom) )
if [ ${#p2} -gt 0 ]; then
  RESUME_ARG=(--resume ${p2[1]}); echo "RESUME phase2 from ${p2[1]}"
else
  seed=( checkpoints_v18_256/phase2_epoch_*.pt(Nom) )
  if [ ${#seed} -eq 0 ]; then echo "FATAL: no checkpoints_v18_256/phase2_epoch_*.pt to init from"; exit 1; fi
  RESUME_ARG=(--init_from ${seed[1]}); echo "INIT phase2 (clean) from ${seed[1]}"
fi

uv run --frozen torchrun --standalone --nproc_per_node=4 -m training.train \
  --gaussian_h5_dir $GH5 --renders_dir $REN \
  --save_dir checkpoints_v18_512lp \
  --batch_size 2 --resolution 512 \
  --pe_type rope --augment_rotation --views_per_epoch 4 \
  --phase1_epochs 5 --phase1_lr 1e-3 \
  --phase2_epochs 20 --phase2_lr 5e-5 \
  --save_interval 1 --keep_last_n 12 \
  --val_h5_dir $VH5 --val_renders_dir $VREN \
  --log_loss_weight 0.5 --lpips_loss_weight 0.5 \
  --num_workers 8 $RESUME_ARG
