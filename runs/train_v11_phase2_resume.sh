#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --mem=256GB
#SBATCH --output=runs/train_v11_phase2_resume_%j.out
#SBATCH --job-name=gformer_v11_p2resume
#SBATCH --mail-type=END,FAIL,BEGIN
# #SBATCH --mail-user=your-email@example.com
#SBATCH --gres=gg:g4:4

# v11 Phase 2 as a SEPARATE job — works around the DDP freeze/unfreeze bug.
#
# Root cause of the V11 Phase 2 plateau: train.py wraps the model in DDP
# (find_unused_parameters=True) and THEN calls freeze_backbone for Phase 1.
# DDP rebuilds its gradient-reduction buckets after the first iteration based
# on which params produced grads. In Phase 1 only the 19 encoder params ever
# participate, so the rebuilt buckets contain only those 19. When Phase 2's
# unfreeze_all re-enables the 195M backbone, those params are in no synced
# bucket -> their grads are computed per-rank but never all-reduced -> the 4
# DDP replicas drift apart -> the effective model is incoherent -> loss stuck
# ~0.0138 regardless of LR or Phase 1 length. (Three in-process Phase 2 runs
# all plateaued there; Phase 1 itself trains fine because its 19 params ARE
# correctly bucketed.)
#
# V9/V10/V10b never tripped this — they all used --skip_phase1 --resume, so
# DDP's first iteration exercised all params and bucketed them correctly.
#
# Fix here: do exactly that. Resume from the short-Phase-1 encoder warmup
# (phase1_epoch_2.pt, loss ~0.0126) and run Phase 2 with --skip_phase1, so DDP
# is constructed once with every param trainable. V9's exact, proven recipe.
#
# Submit:
#   sbatch runs/train_v11_phase2_resume.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1

uv run --frozen torchrun --standalone --nproc_per_node=4 -m training.train \
  --gaussian_h5_dir data_v9/h5s \
  --renders_dir data_v9/renders \
  --save_dir checkpoints_v11_p2resume \
  --resume checkpoints_v11_short_p1/phase1_epoch_2.pt \
  --skip_phase1 \
  --batch_size 5 --resolution 512 \
  --pe_type nerf_perfield \
  --scale_pe_num_freqs 6 \
  --phase2_epochs 100 \
  --phase2_lr 5e-5 \
  --save_interval 5 \
  --val_h5_dir data_v9/h5s_val \
  --val_renders_dir data_v9/renders_val \
  --log_loss_weight 1.0 \
  --lpips_loss_weight 0.0 \
  --num_workers 8
