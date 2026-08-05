#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --mem=256GB
#SBATCH --output=runs/train_v11f_concat_%j.out
#SBATCH --job-name=gformer_v11f
#SBATCH --mail-type=END,FAIL,BEGIN
# #SBATCH --mail-user=your-email@example.com
#SBATCH --gres=gg:g4:4

# v11f = the concat encoder. Fixes the per-field plateau diagnosed by v11e.
#
# v11e proved cause (B): the per-field architecture itself caps the model at 0.0138.
# Per-field projections summed WITHOUT the per-field norms are mathematically a single
# linear map over the concatenated features -- so the entire difference between
# per-field and a plain concat encoder is the 5 per-field RMSNorms, which normalize
# each field per-Gaussian and erase relative salience. The concat encoder removes them:
#   feat  = cat[ nerf(pos)=75, log_scale=3, quat=4, color=3, opacity=1 ]
#   token = gaussian_token + norm( Linear(86,768)(feat) )
# One shared projection weights every field freely (like rope's Linear(14,768)) but
# with position lifted into a NeRF basis. Its bias absorbs per-field offset and its
# weights absorb per-field magnitude -- no per-field normalization, no dataset
# constants. That a shared projection needs no per-field magnitude matching is exactly
# why it fixes per-field, where isolated projections made magnitude matter.
#
# Recipe is IDENTICAL to v11e (--skip_phase1, no --resume: fresh RenderFormer transfer
# + random concat encoder, joint Phase 2 from step 1) so the ONLY variable vs v11e is
# the encoder architecture -- a clean A/B on cause (B).
#   descends (like the rope control, ~0.008 within ep 1) -> the concat encoder is the fix.
#   plateaus at 0.0138                                    -> the bug is deeper than the encoder.
# Verdict from the within-epoch binning of Phase 2 ep 1 (~46 min).
#
# --skip_phase1 => find_unused_parameters=False (the rope-confirmed-correct path).
# bs=4 fits any g4 card (the bs=5 OOM lesson).
#
# Submit:
#   sbatch runs/train_v11f_concat.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run --frozen torchrun --standalone --nproc_per_node=4 -m training.train \
  --gaussian_h5_dir data_v9/h5s \
  --renders_dir data_v9/renders \
  --save_dir checkpoints_v11f \
  --skip_phase1 \
  --batch_size 4 --resolution 512 \
  --pe_type nerf \
  --phase2_epochs 100 \
  --phase2_lr 5e-5 \
  --save_interval 5 \
  --val_h5_dir data_v9/h5s_val \
  --val_renders_dir data_v9/renders_val \
  --log_loss_weight 1.0 \
  --lpips_loss_weight 0.0 \
  --num_workers 8
