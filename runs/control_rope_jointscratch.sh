#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --mem=256GB
#SBATCH --output=runs/control_rope_jointscratch_%j.out
#SBATCH --job-name=gformer_rope_js
#SBATCH --mail-type=END,FAIL,BEGIN
# #SBATCH --mail-user=your-email@example.com
#SBATCH --gres=gg:g4:4

# CONTROL: rope encoder, random init, joint Phase 2 from scratch.
#
# The discriminator the investigation has been missing. The rope control that
# "descended" (control_rope.sh) RESUMED checkpoints_v4/phase1_epoch_20.pt -- an
# already-trained rope encoder. v11e (per-field) and v11f (concat) started from
# RANDOM encoders. So "rope descends / nerf plateaus" is confounded: encoder init
# (trained vs random) AND architecture (rope vs nerf) both varied.
#
# This run is v11e/v11f's EXACT recipe (--skip_phase1, NO --resume: fresh
# RenderFormer transfer + random encoder, joint Phase 2) with pe_type=rope. The
# only variable vs v11f is the encoder architecture.
#   descends (toward ~0.008 within ep 1)  -> random-rope-joint works; the nerf
#                                            architecture is genuinely the problem.
#   plateaus at 0.0138                    -> random-joint plateaus regardless of
#                                            encoder; the plateau is the RECIPE
#                                            (a Phase-1 encoder warmup is needed),
#                                            not the architecture.
# Verdict from the within-epoch binning of Phase 2 ep 1 (~46 min).
#
# --skip_phase1 => find_unused_parameters=False (the rope-confirmed-correct path).
# bs=4 fits any g4 card (the bs=5 OOM lesson).
#
# Submit:
#   sbatch runs/control_rope_jointscratch.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run --frozen torchrun --standalone --nproc_per_node=4 -m training.train \
  --gaussian_h5_dir data_v9/h5s \
  --renders_dir data_v9/renders \
  --save_dir checkpoints_rope_jointscratch \
  --skip_phase1 \
  --batch_size 4 --resolution 512 \
  --pe_type rope \
  --phase2_epochs 100 \
  --phase2_lr 5e-5 \
  --save_interval 5 \
  --val_h5_dir data_v9/h5s_val \
  --val_renders_dir data_v9/renders_val \
  --log_loss_weight 1.0 \
  --lpips_loss_weight 0.0 \
  --num_workers 8
