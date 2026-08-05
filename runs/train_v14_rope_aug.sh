#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 64
#SBATCH --mem=480GB
#SBATCH --output=runs/train_v14_rope_aug_%j.out
#SBATCH --job-name=gformer_v14
#SBATCH --gres=gg:g4:8
#SBATCH --killable
#SBATCH --requeue

# V14 = RoPE encoder (pe_type=rope) at N=20k + RoMa scene-rotation augmentation.
# From scratch: RenderFormer transfer -> Phase 1 (encoder warmup) -> Phase 2 (joint).
#
# WHY rope: the RenderFormer paper (p.6) reports NeRF-on-position is unstable / prone to
# a suboptimal local minimum (our V11 0.0138 plateau). RoPE is the relative-position
# encoding light transport needs; the nerf lift is redundant (RoPE already on). Our rope
# lineage was only ever N=5k (V10b) -- this is the missing N=20k control + the better design.
# WHY aug: the model is rotation-variant; a joint scene+camera rotation is image-preserving
# for our data (sh_degree=None constant color, no world-fixed lighting), so it's free
# robustness -- high-value given our ~2,667-scene starvation. Verified by
# runs/test_rotation_aug.sh (PSNR > 40 dB) BEFORE this run.
#
# KILLABLE + SELF-RESUMING: --killable --requeue per ask. A from-scratch job naively
# requeues from Phase 1 and loses everything, so this script is RESUME-AWARE: it globs the
# latest checkpoint and appends --resume (phase/epoch/optimizer/scheduler aware). Works for
# both SLURM auto-requeue and any manual resubmit. save_interval=1 + keep_last_n=3 caps
# preemption loss at <=1 epoch and storage at ~3x2.3 GB. Checkpoints are written atomically.
#
# BATCH/LR: bs=1 x 8 GPUs = effective batch 8 (== V13), so phase1_lr 1e-3 / phase2_lr 5e-5
# carry over with no rescaling. (If the bs speed probe -- runs/probe_bs_speed_v14.sh --
# favours bs=2, switch to --batch_size 2 and sqrt-scale: phase1_lr 1.41e-3, phase2_lr 7.1e-5.)
#
# Submit (only after the aug invariance + bs probe gates pass):
#   sbatch runs/train_v14_rope_aug.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Resume-aware launch: prefer the latest phase2 checkpoint, else the latest phase1.
# Use a zsh glob ARRAY with the (Nom) qualifier -- N=null_glob (empty, no error, when
# nothing matches), om=order by mtime newest-first. files[1] is the newest, or empty if
# none. (Do NOT pipe a bare `ls` here: a null-glob would leave `ls` argument-less and it
# would list the cwd -- the bug that produced `--resume runs` on the first launch.)
files=( checkpoints_v14/phase2_epoch_*.pt(Nom) )
[ ${#files} -eq 0 ] && files=( checkpoints_v14/phase1_epoch_*.pt(Nom) )
LATEST=${files[1]}
# RESUME_ARG MUST be a zsh ARRAY, not a string: zsh does NOT word-split an unquoted
# scalar, so RESUME_ARG="--resume X" would reach argparse as ONE token "--resume X"
# (the "unrecognized arguments: --resume ..." crash). An array expands to separate
# words, and an empty array expands to nothing.
RESUME_ARG=()
if [ -n "$LATEST" ]; then
  RESUME_ARG=(--resume "$LATEST")
  echo "RESUME-AWARE: found $LATEST -> --resume $LATEST"
else
  echo "RESUME-AWARE: no checkpoint found -> fresh run from RenderFormer transfer"
fi

uv run --frozen torchrun --standalone --nproc_per_node=8 -m training.train \
  --gaussian_h5_dir data_v9_n20k/h5s --renders_dir data_v9/renders \
  --save_dir checkpoints_v14 \
  --batch_size 1 --resolution 512 \
  --pe_type rope --augment_rotation \
  --phase1_epochs 10 --phase1_lr 1e-3 \
  --phase2_epochs 20 --phase2_lr 5e-5 \
  --save_interval 1 --keep_last_n 3 \
  --val_h5_dir data_v9_n20k/h5s_val --val_renders_dir data_v9/renders_val \
  --log_loss_weight 1.0 --lpips_loss_weight 0.0 \
  --num_workers 8 $RESUME_ARG
