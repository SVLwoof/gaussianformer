#!/bin/zsh
#SBATCH --time=20:00:00
#SBATCH -c 16
#SBATCH --mem=64GB
#SBATCH --output=runs/nsweep_n1_%j.out
#SBATCH --job-name=nswn1
#SBATCH --gres=gg:g4:4
#SBATCH --account=sagieb
#SBATCH --requeue
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08,khan-01,khan-02

# N=1 CONTROLS -- the missing sweep point, measured under the CURRENT regime instead of borrowed
# from the June probe (different script, different seed, aug off). Two arms:
#   ARM=objav    scene_0387 (first object of the nested n10 subset -> extends 1 c 10 c 100 c 1000)
#   ARM=tomatoes the June probe's real-scan object, via the same symlink dirs it used
# Recipe = EXACTLY the sweep baseline: init v18_256 ep30, 512+LPIPS0.5, aug ON, wd 0.01,
# views_per_epoch 4, 4xbs1, 30k optimizer steps (= 30k epochs at 1 step/epoch; save/val sparse).
# Readout: margin vs ceiling on the object itself, alongside 7.57 (N=10) / 13.40 (N=100).
# If N=1 under the CURRENT recipe cannot approach the ceiling, the June anchor was doing hidden
# work (aug-off/seed) and the sweep's leftmost point moves.
#   sbatch --export=ARM=objav data_v10/train_nsweep_n1.sh
#   sbatch --export=ARM=tomatoes data_v10/train_nsweep_n1.sh

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

ARM=${ARM:?ARM required (objav|tomatoes)}
if [ "$ARM" = "objav" ]; then
  GH5=data_v10/nsweep/n1_h5; REN=data_v10/nsweep/n1_renders
else
  GH5=experiments/overfit/data/n10_codec/h5s; REN=experiments/overfit/data/n10_codec/renders
fi
SEED=checkpoints_v18_256/phase2_epoch_30.pt
SAVE=checkpoints_n10_codec${CYCLE2:+_r2}
if [ -n "$CYCLE2" ] && [ ! -d checkpoints_n10_codec_r2 ]; then
  SEED_OVERRIDE=checkpoints_n10_codec/phase2_epoch_80.pt
fi
EPOCHS=${EPOCHS_OVR:-80}; SAVE_INT=8
echo "N1 $ARM: h5=$GH5 | $EPOCHS epochs (1 step/epoch), node=$(hostname) sm_${ARCH}"

RESUME_ARG=()
p2=( ${SAVE}/phase2_epoch_*.pt(Nom) )
if [ ${#p2} -gt 0 ]; then
  RESUME_ARG=(--resume ${p2[1]}); echo "RESUME from ${p2[1]}"
else
  [ -f $SEED ] || { echo "FATAL: missing $SEED"; exit 1; }
  RESUME_ARG=(--init_from ${SEED_OVERRIDE:-$SEED}); echo "INIT from $SEED"
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
