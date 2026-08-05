#!/bin/zsh
#SBATCH --time=1:00:00
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH --job-name=ceilstrips
#SBATCH --output=runs/ceilstrips_%j.out
#SBATCH --gres=gg:g4:1
#SBATCH --account=sagieb
#SBATCH --killable
#SBATCH --requeue
#SBATCH --exclude=cyril-01

# GT | rec-GT | V17 strips for the tail cases ceiling_report surfaces.
#   sbatch --export=CASES=data_v10/ceiling/cases.json,OUT=meeting_material/ceiling_cases \
#     data_v10/run_ceiling_strips.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.local/bin:$PATH"

# Per-arch JIT cache -- see run_ceiling_eval.sh for why this is not optional.
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_ext_sm${ARCH:-unknown}"
mkdir -p "$TORCH_EXTENSIONS_DIR"

CASES=${CASES:-data_v10/ceiling/cases.json}
OUT=${OUT:-meeting_material/ceiling_cases}
CKPT=${CKPT:-checkpoints_v17_512lp/phase2_epoch_36.pt}

echo "cases=$CASES out=$OUT ckpt=$CKPT node=$(hostname) sm_${ARCH}"
uv run --no-sync python -m data_v10.ceiling_strips --cases "$CASES" --out "$OUT" --ckpt "$CKPT"
