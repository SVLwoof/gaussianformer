#!/bin/zsh
#SBATCH --time=20:00:00
#SBATCH -c 16
#SBATCH --mem=64GB
#SBATCH --output=runs/tomato_codec4_%j.out
#SBATCH --job-name=tomcod4
#SBATCH --gres=gg:g4:4
#SBATCH --account=sagieb
#SBATCH --requeue
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08,khan-01,khan-02

# CODEC v4: 3x the view budget (4500 randomized views, r 1.05-2.55), continuing the cycle
# ladder from codec3_r2 (the current best: rand -1.62 / close -0.06 / far -1.96 vs rec-GT).
# Cycle budget matches codec3: ~30k steps (4500 views / 4 ranks = 1125 steps/epoch, 27
# epochs = 30,375 steps). Recipe otherwise identical to the sweep baseline: 512 + LPIPS 0.5,
# aug ON, cosine 5e-5, 4xbs1.
#   sbatch --export=NONE data_v10/train_tomato_codec4.sh              # cycle 1 (seed codec3_r2)
#   sbatch --export=CYCLE2=1 data_v10/train_tomato_codec4.sh          # cycle 2 (seed cycle 1)

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

GH5=experiments/overfit/data/tomatoes_codec4/h5s
REN=experiments/overfit/data/tomatoes_codec4/renders
SEED=checkpoints_tomato_codec3_r2/phase2_epoch_80.pt
SAVE=checkpoints_tomato_codec4${CYCLE2:+_r2}
if [ -n "$CYCLE2" ] && [ ! -d checkpoints_tomato_codec4_r2 ]; then
  SEED_OVERRIDE=checkpoints_tomato_codec4/phase2_epoch_27.pt
fi
EPOCHS=${EPOCHS_OVR:-27}; SAVE_INT=3
echo "CODEC4${CYCLE2:+ cycle2}: h5=$GH5 | $EPOCHS epochs (1125 steps/epoch), node=$(hostname) sm_${ARCH}"

RESUME_ARG=()
p2=( ${SAVE}/phase2_epoch_*.pt(Nom) )
if [ ${#p2} -gt 0 ]; then
  RESUME_ARG=(--resume ${p2[1]}); echo "RESUME from ${p2[1]}"
else
  SEED_USE=${SEED_OVERRIDE:-$SEED}
  [ -f $SEED_USE ] || { echo "FATAL: missing $SEED_USE"; exit 1; }
  RESUME_ARG=(--init_from $SEED_USE); echo "INIT from $SEED_USE"
fi

uv run --no-sync torchrun --standalone --nproc_per_node=4 -m training.train \
  --gaussian_h5_dir $GH5 --renders_dir $REN \
  --val_h5_dir data_v10/nsweep/val100_h5 --val_renders_dir data_v10/nsweep/val100_renders \
  --save_dir $SAVE --batch_size 1 --resolution 512 \
  --pe_type rope --augment_rotation \
  --phase2_epochs $EPOCHS --phase2_lr 5e-5 \
  --save_interval $SAVE_INT --keep_last_n 2 \
  --log_loss_weight 0.5 --lpips_loss_weight 0.5 \
  --num_workers 4 $RESUME_ARG
