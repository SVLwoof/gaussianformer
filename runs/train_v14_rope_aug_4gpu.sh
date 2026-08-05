#!/bin/zsh
#SBATCH --time=168:00:00
#SBATCH -c 48
#SBATCH --mem=360GB
#SBATCH --output=runs/train_v14_rope_aug_4gpu_%j.out
#SBATCH --job-name=gformer_v14_4g
#SBATCH --gres=gg:g4:4

# V14 on 4 GPUs, NON-KILLABLE (fits lab quota; 8-GPU non-killable is AssocGrpGRES-blocked).
# Switched here after the 8-GPU killable run kept getting preempted faster than an epoch
# (khan-02 gave 0 epochs/allocation), making no net progress past phase1_epoch_3.
# Guaranteed continuous: no preemption, predictable finish (~145 min/epoch, ~65 h).
#
# CONTINUITY: bs=2 x 4 GPUs = effective batch 8 -- IDENTICAL to the 8 x bs1 checkpoints
# already on disk -- so phase1_lr 1e-3 / phase2_lr 5e-5 transfer with no rescaling and the
# resumed optimizer/scheduler state stays consistent. Resume-aware -> continues from
# phase1_epoch_3 (epoch-based cosine scheduler is unaffected by the GPU-count change;
# steps/epoch is the same 4667 since samples/step = 8 either way).
#
# Same V14 design: pe_type=rope + RoMa rotation aug, save_interval=1 + keep_last_n=3,
# atomic checkpoints. bs=2 N=20k ~40 GB peak -> fits firefoot l40s (45/47.7 GB),
# epona a40 (45 GB), cyril a6000 (48 GB).
#
#   sbatch runs/train_v14_rope_aug_4gpu.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Resume-aware launch (zsh ARRAY -- zsh doesn't word-split an unquoted scalar, so a
# string "--resume X" would reach argparse as one token and crash; (Nom) = null-glob +
# mtime-newest-first, files[1] = newest or empty).
files=( checkpoints_v14/phase2_epoch_*.pt(Nom) )
[ ${#files} -eq 0 ] && files=( checkpoints_v14/phase1_epoch_*.pt(Nom) )
LATEST=${files[1]}
RESUME_ARG=()
if [ -n "$LATEST" ]; then
  RESUME_ARG=(--resume "$LATEST")
  echo "RESUME-AWARE: found $LATEST -> --resume $LATEST"
else
  echo "RESUME-AWARE: no checkpoint found -> fresh run from RenderFormer transfer"
fi

uv run --frozen torchrun --standalone --nproc_per_node=4 -m training.train \
  --gaussian_h5_dir data_v9_n20k/h5s --renders_dir data_v9/renders \
  --save_dir checkpoints_v14 \
  --batch_size 2 --resolution 512 \
  --pe_type rope --augment_rotation \
  --phase1_epochs 10 --phase1_lr 1e-3 \
  --phase2_epochs 20 --phase2_lr 5e-5 \
  --save_interval 1 --keep_last_n 3 \
  --val_h5_dir data_v9_n20k/h5s_val --val_renders_dir data_v9/renders_val \
  --log_loss_weight 1.0 --lpips_loss_weight 0.0 \
  --num_workers 8 $RESUME_ARG
