#!/bin/zsh
#SBATCH --time=24:00:00
#SBATCH -c 32
#SBATCH --mem=64GB
#SBATCH --output=runs/nsweep_geombias_%j.out
#SBATCH --job-name=nswgeom
#SBATCH --gres=gg:g4:8
#SBATCH --killable
#SBATCH --requeue
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08,khan-01,khan-02

# GEOMETRY-BIASED CROSS-ATTENTION PROBE at N=100 (proposal #1 from the codec arc).
# The view transformer's RoPE gives every patch token the SAME position (the camera
# origin), so cross-attention logits carry no per-patch geometry -- "which Gaussians
# are on my ray" must be inferred from features alone. This run adds a zero-init-gated
# ray/Gaussian alignment bias to those logits (--geom_bias). Equivalence at gate=0
# verified bit-exact vs the same seed, so epoch 0 IS v18_256-ep30.
# Data/init/epochs = the N=100 sweep baseline (margin 13.40 dB), but on 8 GPUs (user
# call, 2026-08-16): global batch 8 vs the baseline's 4, so 300 epochs = same data
# passes in HALF the wall-clock but 15k optimizer steps instead of 30k (lr unchanged).
# Read the margin with that caveat. Biased cross-attn layers run SDPA, not flash-attn.
#   PROBE=1 sbatch ... -> 1-epoch VRAM/step-time probe (no checkpoint)
#   sbatch data_v10/train_nsweep_geombias.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.local/bin:$PATH"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_ext_sm${ARCH:-unknown}"
mkdir -p "$TORCH_EXTENSIONS_DIR"

SEED=checkpoints_v18_256/phase2_epoch_30.pt
SAVE=checkpoints_nsweep_n100_geombias

PROBE_ARGS=()
if [ -n "$PROBE" ]; then
  SAVE=tmp/probe_geombias
  PROBE_ARGS=(--phase2_epochs 1 --save_interval 999)
  echo "GEOMBIAS PROBE: 1 epoch, node=$(hostname) sm_${ARCH}"
  echo "gpus: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | paste -sd'; ')"
else
  echo "GEOMBIAS: N=100 baseline schedule + --geom_bias | node=$(hostname) sm_${ARCH}"
fi

RESUME_ARG=()
p2=( ${SAVE}/phase2_epoch_*.pt(Nom) )
if [ -z "$PROBE" ] && [ ${#p2} -gt 0 ]; then
  RESUME_ARG=(--resume ${p2[1]}); echo "RESUME from ${p2[1]}"
else
  [ -f $SEED ] || { echo "FATAL: missing $SEED"; exit 1; }
  RESUME_ARG=(--init_from $SEED); echo "INIT from $SEED"
fi

uv run --no-sync torchrun --standalone --nproc_per_node=8 -m training.train \
  --gaussian_h5_dir data_v10/nsweep/n100_h5 \
  --renders_dir     data_v10/nsweep/n100_renders \
  --val_h5_dir      data_v10/nsweep/val100_h5 \
  --val_renders_dir data_v10/nsweep/val100_renders \
  --save_dir $SAVE --batch_size 1 --resolution 512 \
  --pe_type rope --augment_rotation --views_per_epoch 4 \
  --geom_bias \
  --phase2_epochs 300 --phase2_lr 5e-5 \
  --save_interval 20 --keep_last_n 2 \
  --log_loss_weight 0.5 --lpips_loss_weight 0.5 \
  --num_workers 8 $PROBE_ARGS $RESUME_ARG
rc=$?
exit $rc
