#!/bin/zsh
#SBATCH --time=2:00:00
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH --job-name=ceiling
#SBATCH --output=runs/ceiling_%A_%a.out
#SBATCH --gres=gg:g4:1
#SBATCH --account=sagieb
#SBATCH --killable
#SBATCH --requeue
#SBATCH --exclude=cyril-01

# rec-GT ceiling vs V17, per (scene, view), as a SLURM ARRAY -- each task strides the slice's
# canonical scene list, so SLURM schedules the width instead of us hand-packing shards.
#
# --killable: must NOT contend with V18 stage A's non-preemptible quota. ceiling_eval.py streams to
# JSONL and skips completed (scene, view) rows, so a preempted task resumes where it stopped --
# required, not a nicety, on a preemptible node.
#
# No --mail-type: requeue on killable spams. gsplat's JIT cache is warm (cu128 + cu130 .so both
# present), so array tasks can start in parallel without racing the ninja build.
#
#   sbatch --array=0-11 --export=SPLIT=val,NSHARDS=12 data_v10/run_ceiling_eval.sh
# Submitted from inside claude_node's allocation, so --export must be an explicit list: inheriting
# ALL leaks the parent job's GPU context and tasks die on "invalid device ordinal".

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.local/bin:$PATH"   # clean --export drops uv from PATH

# Isolate the gsplat JIT build cache PER GPU ARCHITECTURE. torch keys its extension dir on
# python+CUDA version only (py312_cu128), NOT on compute capability -- so an a40 (sm_86), an l40s
# (sm_89) and an rtxpro6000 (sm_120) all rebuild 'gsplat_cuda' into the SAME directory and clobber
# each other mid-build. That is what killed tasks with "Error building extension 'gsplat_cuda'"
# even though a warm .so was present. Keying on compute_cap keeps reuse within an arch.
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_ext_sm${ARCH:-unknown}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
echo "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR (sm_${ARCH})"

SPLIT=${SPLIT:?SPLIT required (train|val|unseen2x)}
NSHARDS=${NSHARDS:?NSHARDS required (must equal the --array width)}
SHARD=${SLURM_ARRAY_TASK_ID:-0}
# 4 matched views rather than all 14: views within an object are highly correlated, so the
# statistical unit is the OBJECT -- budget is better spent on object count. The same 4 cameras are
# used for every slice, which is what keeps train/test comparable (a view MISMATCH between slices
# would confound camera hardness with memorisation; a lower matched count does not).
VIEWS=${VIEWS:-0,4,7,11}
TAG=${TAG:-v17}
CKPT=${CKPT:-checkpoints_v17_512lp/phase2_epoch_36.pt}
SCENES=data_v10/ceiling_scenes/${SPLIT}.json
OUT=data_v10/ceiling/${TAG}_${SPLIT}_${SHARD}of${NSHARDS}.jsonl
LIMIT_ARG=()
[ -n "$LIMIT" ] && LIMIT_ARG=(--limit $LIMIT)

mkdir -p data_v10/ceiling
# --no-sync, NOT --frozen. `uv run --frozen` still RECONCILES the venv on every invocation, and
# uv.lock records flash-attn as 2.8.3 while the installed wheel normalises to
# 2.8.3+cu12torch2.9cxx11abitrue -- so uv uninstalls and reinstalls it every single run. With an
# array of 45 jobs sharing one NFS .venv that is a race: tasks importing gsplat during another
# task's reinstall window died on "cannot import name 'csrc'", and tasks that started mid-window
# silently fell back from FlashAttention to SDPA. --no-sync leaves the environment untouched.
echo "split=$SPLIT tag=$TAG shard=$SHARD/$NSHARDS views=$VIEWS ckpt=$CKPT node=$(hostname) out=$OUT"
uv run --no-sync python -m data_v10.ceiling_eval \
  --split "$SPLIT" --scenes_file "$SCENES" --out "$OUT" --model_tag "$TAG" \
  --shard "$SHARD" --nshards "$NSHARDS" --views "$VIEWS" --ckpt "$CKPT" $LIMIT_ARG
