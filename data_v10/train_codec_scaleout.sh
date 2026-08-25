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
# cosine cycles; save every 3 epochs (keep 3) -- cadence must beat the preemption
# window (~1h), and 3 divides 27 so the final epoch gets a numbered checkpoint.
#   sbatch --export=SCENE=scene_0959 data_v10/train_codec_scaleout.sh          # cycle 1
#   sbatch --export=SCENE=scene_0959,CYCLE2=1 data_v10/train_codec_scaleout.sh # cycle 2

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.local/bin:$PATH"
# debian13 nodes (cluster upgrade 2026-10-05; test via --reservation=5787): isolated venv + caches,
# driver libs live in /etc/lib64/nvidia (stale ld.so.cache), nvidia-smi absent -> arch via torch.
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

SCENE=${SCENE:?SCENE required, e.g. scene_0959}
GH5=experiments/overfit/data/codec_scaleout/$SCENE/h5s
REN=experiments/overfit/data/codec_scaleout/$SCENE/renders
SEED=checkpoints_v18_256/phase2_epoch_30.pt
# Cycle N (CYCLE=N; CYCLE2=1 kept as an alias for N=2) trains in _rN and seeds from cycle N-1's
# final. Seed only when there is nothing to resume from. Testing the DIR here (not its
# contents) cold-started 0262/0772 c2 from v18 after a preemption that had mkdir'd the r2
# dir but not yet saved a checkpoint (2026-08-22).
CYCLE=${CYCLE:-${CYCLE2:+2}}; CYCLE=${CYCLE:-1}
cyc_suffix() { [ "$1" -ge 2 ] && echo "_r$1"; return 0; }
SAVE=${SAVE_OVR:-checkpoints_codec_so_${SCENE}$(cyc_suffix $CYCLE)}
if [ "$CYCLE" -ge 2 ]; then
  SEED_OVERRIDE=checkpoints_codec_so_${SCENE}$(cyc_suffix $((CYCLE-1)))/phase2_epoch_${C1_FINAL:-27}.pt
fi
NPROC=${NPROC:-8}
# Effective batch is pinned to 8 samples/step: fewer GPUs -> more grad accumulation.
# Submit 4-GPU variant with: sbatch --gres=gg:g4:4 -c 16 --export=SCENE=...,NPROC=4 <script>
GRAD_ACCUM=${GRAD_ACCUM:-$((8/NPROC))}
EPOCHS=${EPOCHS_OVR:-27}; SAVE_INT=${SAVE_INT_OVR:-3}
echo "CODEC-SO $SCENE cycle$CYCLE: $EPOCHS epochs, ${NPROC}xbs1 x accum${GRAD_ACCUM} ($((9000/NPROC/GRAD_ACCUM)) steps/epoch), node=$(hostname) sm_${ARCH}"

RESUME_ARG=()
p2=( ${SAVE}/phase2_epoch_*.pt(Nom) )
if [ ${#p2} -gt 0 ]; then
  RESUME_ARG=(--resume ${p2[1]}); echo "RESUME from ${p2[1]}"
else
  SEED_USE=${SEED_OVERRIDE:-$SEED}
  [ -f $SEED_USE ] || { echo "FATAL: missing $SEED_USE"; exit 1; }
  RESUME_ARG=(--init_from $SEED_USE); echo "INIT from $SEED_USE"
fi

uv run --no-sync torchrun --standalone --nproc_per_node=$NPROC -m training.train \
  --gaussian_h5_dir $GH5 --renders_dir $REN \
  --val_h5_dir data_v10/nsweep/val100_h5 --val_renders_dir data_v10/nsweep/val100_renders \
  --save_dir $SAVE --batch_size 1 --grad_accum $GRAD_ACCUM --resolution 512 \
  --pe_type rope --augment_rotation \
  --phase2_epochs $EPOCHS --phase2_lr 5e-5 \
  --save_interval $SAVE_INT --keep_last_n 2 \
  --log_loss_weight 0.5 --lpips_loss_weight 0.5 \
  --num_workers 3 $RESUME_ARG
