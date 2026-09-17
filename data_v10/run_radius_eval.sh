#!/bin/zsh
#SBATCH --time=3:00:00
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH --job-name=radeval
#SBATCH --output=runs/radeval_%j.out
#SBATCH --gres=gg:g4:1
#SBATCH --requeue
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08
# Close-range held-out readout (plan 2026-09-17 #3): heldout300 at orbit radius RADIUS with GT
# rendered from the full splats (data_v10/multi_radius_datagen.py --eval_radius).
#   sbatch --killable --account=killable-cs --export=TAG=probe_p2r_fg,CKPT=...,EXTRA="--model_cfg;proj_rope_2d=true" data_v10/run_radius_eval.sh
source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.local/bin:$PATH"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
export TORCH_EXTENSIONS_DIR="${TMPDIR:-/tmp}/torch_ext_job${SLURM_JOB_ID}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
SHARED="$HOME/.cache/torch_ext_sm${ARCH:-unknown}"
[ -f "$SHARED/gsplat_cuda/gsplat_cuda.so" ] && cp -r "$SHARED/gsplat_cuda" "$TORCH_EXTENSIONS_DIR/" 2>/dev/null
trap 'rm -rf "$TORCH_EXTENSIONS_DIR"' EXIT
: ${TAG:?} ${CKPT:?}
RADIUS=${RADIUS:-1.15}; RTAG=${RADIUS//./}
RENDERS=${RENDERS:-data_v10/nsweep/heldout300_r${RTAG}_renders}
EXTRA=${EXTRA//;/ }
[ -f "$RENDERS/scene_0007_view_0.png" ] || { echo "FATAL: no GT renders in $RENDERS"; exit 1; }
uv run --no-sync python -m data_v10.ceiling_eval \
  --split val --scenes_file data_v10/nsweep/heldout300_scenes.json --radius $RADIUS --renders_dir $RENDERS \
  --out data_v10/ceiling/${TAG}_heldout_r${RTAG}.jsonl \
  --model_tag ${TAG}_heldout_r${RTAG} --views 0,4,7,11 --ckpt "$CKPT" ${=EXTRA}
rc=$?; echo "DONE_RADIUS_EVAL rc=$rc"; exit $rc
