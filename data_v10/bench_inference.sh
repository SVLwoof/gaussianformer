#!/bin/zsh
#SBATCH --time=1:00:00
#SBATCH -c 4
#SBATCH --mem=24GB
#SBATCH --job-name=bench_inf
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue
#SBATCH --exclude=cyril-01,firefoot-01,khan-01,khan-02,firefoot-13
# Output goes to $HOME (lab share may be full). sbatch --output=$HOME/bench_%j.out ...
source /etc/profile.d/huji-lmod.sh; module load nvidia; module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_ext_sm${ARCH:-unknown}"
: ${SCENE:?} ${CKPT:?}
PYTHONPATH=. uv run --no-sync python data_v10/bench_inference.py
