#!/bin/zsh
#SBATCH --time=3:00:00
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH --job-name=nsweval
#SBATCH --output=runs/nsweval_%j.out
#SBATCH --gres=gg:g4:1
#SBATCH --account=sagieb
#SBATCH --killable
#SBATCH --requeue
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08

# Evaluate one N-sweep checkpoint against the rec-GT ceiling, on TWO scene sets:
#   TRAIN  -- the run's own training objects. This is the FIT measure and the whole point:
#             if a model cannot reach the ceiling on 10 objects it has seen thousands of times,
#             scale is not the bottleneck.
#   HELDOUT -- a common 300-object val set, disjoint from the val100 used for in-training
#             validation, so generalisation is not read off objects the run was monitored on.
#
#   N=10 sbatch --export=N=10 data_v10/run_nsweep_eval.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.local/bin:$PATH"

# Per-JOB extension dir, not just per-arch. Per-arch stops an a40 and an l40s clobbering each
# other, but two jobs on the SAME arch still race the same gsplat_cuda build dir -- which is
# exactly what killed the first eval attempt (two sm_89 nodes at once). A private dir costs one
# ~5 min compile per job and cannot race anything.
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
SHARED="$HOME/.cache/torch_ext_sm${ARCH:-unknown}"
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_ext_job${SLURM_JOB_ID}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
# Seed from the shared per-arch cache if it holds a complete build, to skip the compile.
if [ -f "$SHARED/gsplat_cuda/gsplat_cuda.so" ]; then
  cp -r "$SHARED/gsplat_cuda" "$TORCH_EXTENSIONS_DIR/" 2>/dev/null && echo "seeded JIT cache from $SHARED"
fi
trap 'rm -rf "$TORCH_EXTENSIONS_DIR"' EXIT

N=${N:?N required}
TAG=${TAG:-nsweep_n${N}}   # override for variants (ctrl, fgloss, scratch) -- rows and resume
                           # state are keyed by output file, so variants MUST NOT share one
CKPT=${CKPT:-$(ls -t checkpoints_nsweep_n${N}/phase2_epoch_*.pt 2>/dev/null | head -1)}
[ -z "$CKPT" ] && { echo "FATAL: no checkpoint for N=$N"; exit 1; }
echo "N=$N ckpt=$CKPT node=$(hostname) sm_${ARCH}"

# NOTE: do NOT end this script on a bare `echo` -- a trailing echo returns 0 and masks a failed
# python run, which is how the first attempt reported COMPLETED 0:0 while writing zero rows.
rc=0

# Own training objects -- the fit measure.
uv run --no-sync python -m data_v10.ceiling_eval \
  --split train --scenes_file data_v10/nsweep/n${N}_scenes.json \
  --out data_v10/ceiling/${TAG}_train.jsonl \
  --model_tag ${TAG}_train --views 0,4,7,11 --ckpt "$CKPT" || rc=1

# Common held-out set -- generalisation.
uv run --no-sync python -m data_v10.ceiling_eval \
  --split val --scenes_file data_v10/nsweep/heldout300_scenes.json \
  --out data_v10/ceiling/${TAG}_heldout.jsonl \
  --model_tag ${TAG}_heldout --views 0,4,7,11 --ckpt "$CKPT" || rc=1

echo "DONE_NSWEEP_EVAL rc=$rc"
exit $rc
