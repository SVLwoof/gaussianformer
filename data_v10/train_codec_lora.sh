#!/bin/zsh
#SBATCH --time=40:00:00
#SBATCH -c 16
#SBATCH --mem=48GB
#SBATCH --output=runs/codec_lora_%j.out
#SBATCH --job-name=codeclora
#SBATCH --gres=gg:g4:4
#SBATCH --requeue
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08,khan-01,khan-02,firefoot-13,firefoot-04,firefoot-16

# LoRA adapter on the frozen v18_256 base for one scale-out object (training/train_lora.py).
# Same data + recipe as train_codec_scaleout.sh (9000 views, 512px, LPIPS 0.5, rot-aug,
# effective batch 8 by default); only the adapter trains, checkpoints are MBs.
#   sbatch --killable --account=killable-cs --export=SCENE=gopro,LORA_RANK=4 data_v10/train_codec_lora.sh   # or -A sagieb (quota)
# Knobs (env): LORA_RANK (req) LORA_ALPHA LORA_TARGETS LORA_DROPOUT LR WD GRAD_ACCUM EPOCHS SAVE_INT SAVE_DIR NPROC SEED

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
LORA_RANK=${LORA_RANK:?LORA_RANK required, e.g. 4}
GH5=experiments/overfit/data/codec_scaleout/$SCENE/h5s
REN=experiments/overfit/data/codec_scaleout/$SCENE/renders
SEED=${SEED:-checkpoints_v18_256/phase2_epoch_30.pt}
SAVE_DIR=${SAVE_DIR:-checkpoints_lora_${SCENE}_r${LORA_RANK}}
NPROC=${NPROC:-4}
GRAD_ACCUM=${GRAD_ACCUM:-$((8/NPROC))}
EPOCHS=${EPOCHS:-27}; SAVE_INT=${SAVE_INT:-3}; LR=${LR:-2e-4}; WD=${WD:-0}
[ -f $SEED ] || { echo "FATAL: missing $SEED"; exit 1; }
OPT=()
[ -n "$LORA_ALPHA" ] && OPT+=(--alpha $LORA_ALPHA)
[ -n "$LORA_TARGETS" ] && OPT+=(--targets "$LORA_TARGETS")
[ -n "$LORA_DROPOUT" ] && OPT+=(--dropout $LORA_DROPOUT)
# MODEL_CFG: ;-separated GaussianFormerConfig overrides of the BASE (e.g. "proj_rope_2d=true");
# FG: fg-weighted loss background weight (e.g. 0.05). Both optional.
[ -n "$MODEL_CFG" ] && OPT+=(--model_cfg ${(s:;:)MODEL_CFG})
[ -n "$FG" ] && OPT+=(--fg_bg_weight $FG)
echo "CODEC-LORA $SCENE r=$LORA_RANK lr=$LR wd=$WD: $EPOCHS epochs, ${NPROC}xbs1 x accum${GRAD_ACCUM}, save=$SAVE_DIR, node=$(hostname) sm_${ARCH}"

uv run --no-sync torchrun --standalone --nproc_per_node=$NPROC -m training.train_lora \
  --gaussian_h5_dir $GH5 --renders_dir $REN --init_from $SEED --save_dir $SAVE_DIR \
  --rank $LORA_RANK --lr $LR --weight_decay $WD "${OPT[@]}" \
  --epochs $EPOCHS --save_interval $SAVE_INT --keep_last_n 2 \
  --batch_size 1 --grad_accum $GRAD_ACCUM --resolution 512 --augment_rotation \
  --log_loss_weight 0.5 --lpips_loss_weight 0.5 --num_workers 3
rc=$?
# torchrun has returned non-zero (exit 7, no traceback) after a fully successful run, which
# cancelled the afterok eval. The adapter file is the success criterion.
[ -f "$SAVE_DIR/lora_final.pt" ] && exit 0
exit $rc
