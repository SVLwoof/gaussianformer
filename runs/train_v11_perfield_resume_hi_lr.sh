#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --mem=256GB
#SBATCH --output=runs/train_v11_perfield_resume_hi_lr_%j.out
#SBATCH --job-name=gformer_v11_resume_hi_lr
#SBATCH --mail-type=END,FAIL,BEGIN
# #SBATCH --mail-user=your-email@example.com
#SBATCH --gres=gg:g4:4

# v11 Phase 2 restart from phase1_epoch_5.pt with 4x higher LR.
#
# Why: the original v11 (job 30617577) Phase 2 ran at lr=5e-5 (V9/V10b fine-tune LR)
# and was flat-to-rising over 5 epochs (val 0.0087 -> 0.0145 vs Phase 1). Hypothesis:
# the per-field encoder pushed inputs into a basin the backbone hadn't seen, so the
# backbone needed to move farther than a fine-tune LR allows.
#
# Recipe: resume from phase1_epoch_5.pt (best v11 ckpt — only encoder trained, backbone
# at RenderFormer init), skip Phase 1, run Phase 2 with lr=2e-4 over 100 epochs cosine.
# Outputs to a fresh checkpoint dir so we don't clobber the failed phase2_epoch_5.pt.
#
# Submit:
#   sbatch runs/train_v11_perfield_resume_hi_lr.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1

uv run --frozen torchrun --standalone --nproc_per_node=4 -m training.train \
  --gaussian_h5_dir data_v9/h5s \
  --renders_dir data_v9/renders \
  --save_dir checkpoints_v11_hi_lr \
  --resume checkpoints_v11/phase1_epoch_5.pt \
  --skip_phase1 \
  --batch_size 5 --resolution 512 \
  --pe_type nerf_perfield \
  --scale_pe_num_freqs 6 \
  --phase2_epochs 100 \
  --phase2_lr 2e-4 \
  --save_interval 5 \
  --val_h5_dir data_v9/h5s_val \
  --val_renders_dir data_v9/renders_val \
  --log_loss_weight 1.0 \
  --lpips_loss_weight 0.0 \
  --num_workers 8
