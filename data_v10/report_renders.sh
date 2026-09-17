#!/bin/zsh
#SBATCH --time=3:00:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --output=runs/report_renders_%j.out
#SBATCH --job-name=rptrender
#SBATCH --gres=gg:g4:1
#SBATCH --requeue
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08,khan-01,khan-02
# Render sweep for the supervisor report. ARGS = everything after report_renders.py.
#   sbatch --killable --account=killable-cs --export=ARGS="ladder --scenes 7 23 --out docs/report/renders" data_v10/report_renders.sh
source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_ext_sm${ARCH:-unknown}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
: ${ARGS:?}
PYTHONPATH=. uv run --no-sync python data_v10/report_renders.py ${(z)ARGS}
