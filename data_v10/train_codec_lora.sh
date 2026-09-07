#!/bin/zsh
#SBATCH --time=40:00:00
#SBATCH -c 16
#SBATCH --mem=48GB
#SBATCH --output=runs/codec_lora_%j.out
#SBATCH --job-name=codeclora
#SBATCH --gres=gg:g4:4
#SBATCH --killable
#SBATCH --account=killable-cs
#SBATCH --requeue
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08,khan-01,khan-02,firefoot-13,firefoot-04,firefoot-16

# LoRA feasibility (2026-09-07): same data / cycle recipe as train_codec_scaleout.sh
# (9000 views, 27 epochs, eff. batch 8, 512px, LPIPS 0.5, rot-aug), but only rank-RANK
# adapters on the attention projections train; the v18_256 base stays frozen. Checkpoints
# hold the adapter only, so the run dir stays in the MB range.
#   sbatch --export=SCENE=gopro,RANK=4 data_v10/train_codec_lora.sh
# LR: LoRA convention is a few x the full-FT rate; 2e-4 = 4x the 5e-5 codec recipe.

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.local/bin:$PATH"
EXT_TAG=
if grep -q '^13' /etc/debian_version 2>/dev/null; then
  export UV_PROJECT_ENVIRONMENT=/cs/labs/sagieb/shahaf_levy/venvs/gf-deb13
  export UV_CACHE_DIR=/cs/labs/sagieb/shahaf_levy/uv_cache_deb13
  export LD_LIBRARY_PATH=/etc/lib64/nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  EXT_TAG=deb13_
fi
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
[ -n "$ARCH" ] || ARCH=$(uv run --no-sync python -c "import torch;print('%d%d'%torch.cuda.get_device_capability())" 2>/dev/null)
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_ext_${EXT_TAG}sm${ARCH:-unknown}"
mkdir -p "$TORCH_EXTENSIONS_DIR"

SCENE=${SCENE:?SCENE required, e.g. gopro}
RANK=${RANK:?RANK required, e.g. 4}
GH5=experiments/overfit/data/codec_scaleout/$SCENE/h5s
REN=experiments/overfit/data/codec_scaleout/$SCENE/renders
SEED=${SEED_OVR:-checkpoints_v18_256/phase2_epoch_30.pt}
SAVE=${SAVE_OVR:-checkpoints_codec_lora_${SCENE}_r${RANK}}
NPROC=${NPROC:-4}
GRAD_ACCUM=${GRAD_ACCUM:-$((8/NPROC))}
EPOCHS=${EPOCHS_OVR:-27}; SAVE_INT=${SAVE_INT_OVR:-3}; LR=${LR_OVR:-2e-4}
echo "CODEC-LORA $SCENE r=$RANK lr=$LR: $EPOCHS epochs, ${NPROC}xbs1 x accum${GRAD_ACCUM}, node=$(hostname) sm_${ARCH}"

RESUME_ARG=()
p2=( ${SAVE}/phase2_epoch_*.pt(Nom) )
if [ ${#p2} -gt 0 ]; then
  RESUME_ARG=(--resume ${p2[1]}); echo "RESUME from ${p2[1]}"
else
  [ -f $SEED ] || { echo "FATAL: missing $SEED"; exit 1; }
  RESUME_ARG=(--init_from $SEED); echo "INIT from $SEED"
fi

uv run --no-sync torchrun --standalone --nproc_per_node=$NPROC -m training.train \
  --gaussian_h5_dir $GH5 --renders_dir $REN \
  --val_h5_dir data_v10/nsweep/val100_h5 --val_renders_dir data_v10/nsweep/val100_renders \
  --save_dir $SAVE --batch_size 1 --grad_accum $GRAD_ACCUM --resolution 512 \
  --pe_type rope --augment_rotation \
  --phase2_epochs $EPOCHS --phase2_lr $LR \
  --lora_rank $RANK \
  --save_interval $SAVE_INT --keep_last_n 2 \
  --log_loss_weight 0.5 --lpips_loss_weight 0.5 \
  --num_workers 3 $RESUME_ARG
