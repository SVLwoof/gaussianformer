#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --mem=256GB
#SBATCH --output=runs/train_v11_perfield_short_phase1_%j.out
#SBATCH --job-name=gformer_v11_short_p1
#SBATCH --mail-type=END,FAIL,BEGIN
# #SBATCH --mail-user=your-email@example.com
#SBATCH --gres=gg:g4:4

# v11 with shortened, gentler Phase 1 to avoid the freeze-then-unfreeze trap.
#
# Diagnosis: V11 Phase 1 (lr=1e-3, 5 epochs) drove the encoder onto a plateau
# (loss ~0.008) where it over-specialized to the frozen RenderFormer-init backbone.
# Both Phase 2 retries (lr=5e-5 plateaued at 0.0138; lr=2e-4 plateaued at 0.0148)
# failed to escape that saddle.
#
# Recipe: cut Phase 1 to 2 epochs at 3x lower LR — same descent shape as ep 1-2
# of the original Phase 1, stops before the plateau (where the trap formed). Then
# standard V9-style Phase 2 fine-tune across the full backbone.
#
#   Phase 1: lr=3e-4 (vs 1e-3), 2 epochs cosine -> 1e-5
#   Phase 2: lr=5e-5, 100 epochs cosine (V9 recipe)
#   Init:    RenderFormer-v1-base (no --resume)
#   Arch:    --pe_type nerf_perfield --scale_pe_num_freqs 6
#
# Submit:
#   sbatch runs/train_v11_perfield_short_phase1.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1

uv run --frozen torchrun --standalone --nproc_per_node=4 -m training.train \
  --gaussian_h5_dir data_v9/h5s \
  --renders_dir data_v9/renders \
  --save_dir checkpoints_v11_short_p1 \
  --batch_size 5 --resolution 512 \
  --pe_type nerf_perfield \
  --scale_pe_num_freqs 6 \
  --phase1_epochs 2 \
  --phase1_lr 3e-4 \
  --phase2_epochs 100 \
  --phase2_lr 5e-5 \
  --save_interval 5 \
  --val_h5_dir data_v9/h5s_val \
  --val_renders_dir data_v9/renders_val \
  --log_loss_weight 1.0 \
  --lpips_loss_weight 0.0 \
  --num_workers 8
