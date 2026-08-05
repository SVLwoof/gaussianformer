#!/bin/zsh
#SBATCH --time=12:00:00
#SBATCH -c 32
#SBATCH --mem=256GB
#SBATCH --output=runs/train_v13b_lpips_%j.out
#SBATCH --job-name=gformer_v13b
#SBATCH --gres=gg:g4:8
#SBATCH --killable
#SBATCH --requeue

# V13b = LPIPS fine-tune of V13 (phase2_epoch_20.pt), mirroring the V9->V10b recipe.
# Warm-starts V13 weights (--init_from: weights only, fresh Phase 2, fresh cosine
# from 5e-5, Phase 1 skipped), turns on LPIPS=0.2, and blazes a short fine-tune on
# 8 GPUs.
#
# Effective batch = bs 1 x 8 GPUs = 8, identical to V13 (bs 2 x 4), so the 5e-5 LR
# transfers with no rescaling. bs=1 N=20k (~19-22 GB) + LPIPS net (~5 GB) fits the
# 44-46 GB g4 cards with wide headroom (bs=2+LPIPS would risk OOM at ~45 GB).
#
# 5 epochs, save_interval=1 -> a checkpoint + val every epoch so we can eval the
# trajectory and stop as soon as LPIPS beats V10b (0.0283). ~20 min/epoch -> ~1.7 h.
#
# killable+requeue per explicit ask (fast turnaround). NOTE: on preemption, --requeue
# restarts from scratch (--init_from always re-warms from V13 ep20) -- acceptable for
# a ~1.7 h job; per-epoch checkpoints are for eval, not crash-resume.
#
# Submit:
#   sbatch runs/train_v13b_lpips.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT="checkpoints_v13/phase2_epoch_20.pt"
if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  exit 1
fi

uv run --frozen torchrun --standalone --nproc_per_node=8 -m training.train \
  --init_from "$CKPT" \
  --gaussian_h5_dir data_v9_n20k/h5s \
  --renders_dir data_v9/renders \
  --save_dir checkpoints_v13b \
  --batch_size 1 --resolution 512 \
  --pe_type nerf \
  --phase2_epochs 5 --phase2_lr 5e-5 \
  --save_interval 1 \
  --val_h5_dir data_v9_n20k/h5s_val \
  --val_renders_dir data_v9/renders_val \
  --log_loss_weight 1.0 \
  --lpips_loss_weight 0.2 \
  --num_workers 8
