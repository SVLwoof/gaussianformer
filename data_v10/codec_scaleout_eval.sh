#!/bin/zsh
#SBATCH --time=2:00:00
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH --output=runs/codec_so_eval_%j.out
#SBATCH --job-name=csoeval
#SBATCH --gres=gg:g4:1
#SBATCH --requeue
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08,khan-01,khan-02

# Scale-out codec verdict for one (SCENE, CKPT). TAG optional.
#   sbatch --killable --account=killable-cs --export=SCENE=...,CKPT=... data_v10/codec_scaleout_eval.sh   # or -A sagieb

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
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

: ${SCENE:?} ${CKPT:?}
PYTHONPATH=. SCENE=$SCENE CKPT=$CKPT TAG=${TAG:-} uv run --no-sync python data_external/codec_scaleout_eval.py
