#!/bin/zsh
#SBATCH --time=20:00:00
#SBATCH -c 32
#SBATCH --mem=64GB
#SBATCH --output=runs/codec_so_%j.out
#SBATCH --job-name=codecso
#SBATCH --gres=gg:g4:8
#SBATCH --account=sagieb
#SBATCH --requeue
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08,khan-01,khan-02,firefoot-13,firefoot-04

# One scale-out object's codec training (2026-08-18 locked 10-object list). Cold start
# from the generalist v18_256 seed (unlike the tomato lineage) -- whether that suffices
# IS part of the scale-out question. 9000 views (uniform+grazing), 8xbs1, ~30k-step
# cosine cycles; save every 9 epochs (keep 3) so the rec-GT crossing point can be
# located at sub-cycle granularity, esp. for the SIMPLE objects (apple/vase).
#   sbatch --export=SCENE=scene_0959 data_v10/train_codec_scaleout.sh          # cycle 1
#   sbatch --export=SCENE=scene_0959,CYCLE2=1 data_v10/train_codec_scaleout.sh # cycle 2

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

SCENE=${SCENE:?SCENE required, e.g. scene_0959}
GH5=experiments/overfit/data/codec_scaleout/$SCENE/h5s
REN=experiments/overfit/data/codec_scaleout/$SCENE/renders
SEED=checkpoints_v18_256/phase2_epoch_30.pt
SAVE=checkpoints_codec_so_${SCENE}${CYCLE2:+_r2}
if [ -n "$CYCLE2" ] && [ ! -d checkpoints_codec_so_${SCENE}_r2 ]; then
  SEED_OVERRIDE=checkpoints_codec_so_${SCENE}/phase2_epoch_27.pt
fi
EPOCHS=${EPOCHS_OVR:-27}; SAVE_INT=9
echo "CODEC-SO $SCENE${CYCLE2:+ cycle2}: $EPOCHS epochs (1125 steps/epoch, 8xbs1), node=$(hostname) sm_${ARCH}"

RESUME_ARG=()
p2=( ${SAVE}/phase2_epoch_*.pt(Nom) )
if [ ${#p2} -gt 0 ]; then
  RESUME_ARG=(--resume ${p2[1]}); echo "RESUME from ${p2[1]}"
else
  SEED_USE=${SEED_OVERRIDE:-$SEED}
  [ -f $SEED_USE ] || { echo "FATAL: missing $SEED_USE"; exit 1; }
  RESUME_ARG=(--init_from $SEED_USE); echo "INIT from $SEED_USE"
fi

uv run --no-sync torchrun --standalone --nproc_per_node=8 -m training.train \
  --gaussian_h5_dir $GH5 --renders_dir $REN \
  --val_h5_dir data_v10/nsweep/val100_h5 --val_renders_dir data_v10/nsweep/val100_renders \
  --save_dir $SAVE --batch_size 1 --resolution 512 \
  --pe_type rope --augment_rotation \
  --phase2_epochs $EPOCHS --phase2_lr 5e-5 \
  --save_interval $SAVE_INT --keep_last_n 3 \
  --log_loss_weight 0.5 --lpips_loss_weight 0.5 \
  --num_workers 3 $RESUME_ARG
