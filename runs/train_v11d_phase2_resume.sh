#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --mem=256GB
#SBATCH --output=runs/train_v11d_phase2_resume_%j.out
#SBATCH --job-name=gformer_v11d_p2
#SBATCH --mail-type=END,FAIL,BEGIN
# #SBATCH --mail-user=your-email@example.com
#SBATCH --gres=gg:g4:4

# v11d Phase 2, resumed after the bs=5 OOM.
#
# The original v11d run (30623103) finished Phase 1 cleanly (phase1_epoch_2.pt saved)
# then OOM'd on all 4 ranks at the start of Phase 2: it landed on 44 GB g4 cards, and
# bs=5 Phase 2 peaks ~43 GB -- fits the 48 GB g4 cards but not the 44 GB ones.
#
# Fix: resume Phase 2 from phase1_epoch_2.pt at bs=4 (V10b's proven size, ~40.6 GB
# peak -- fits any g4 card). Phase 1 is the encoder warmup; batch size does not
# affect its saved weights, so no need to redo it. expandable_segments guards
# against the fragmentation the OOM trace flagged (~3 GB reserved-but-unallocated).
#
# Submit:
#   sbatch runs/train_v11d_phase2_resume.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run --frozen torchrun --standalone --nproc_per_node=4 -m training.train \
  --gaussian_h5_dir data_v9/h5s \
  --renders_dir data_v9/renders \
  --save_dir checkpoints_v11d \
  --resume checkpoints_v11d/phase1_epoch_2.pt \
  --skip_phase1 \
  --batch_size 4 --resolution 512 \
  --pe_type nerf_perfield \
  --phase2_epochs 100 \
  --phase2_lr 5e-5 \
  --save_interval 5 \
  --val_h5_dir data_v9/h5s_val \
  --val_renders_dir data_v9/renders_val \
  --log_loss_weight 1.0 \
  --lpips_loss_weight 0.0 \
  --num_workers 8
