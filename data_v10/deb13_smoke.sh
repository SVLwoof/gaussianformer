#!/bin/zsh
#SBATCH --time=3:00:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --output=runs/deb13_smoke_%j.out
#SBATCH --job-name=deb13smoke
#SBATCH --gres=gg:g4:1
#SBATCH --killable
#SBATCH --requeue
#SBATCH --reservation=5787
# Debian13 (upgrade 2026-10-05) smoke test. Builds an ISOLATED venv + caches; never touches .venv.
source /etc/profile.d/huji-lmod.sh 2>/dev/null
module load nvidia; module load cuda; echo "modules: $(module -t list 2>&1 | tr "\n" " ")"; echo "nvcc: $(which nvcc) CUDA_HOME=$CUDA_HOME"
cd /cs/labs/sagieb/shahaf_levy/gaussianformer
export PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT=/cs/labs/sagieb/shahaf_levy/venvs/gf-deb13
export UV_CACHE_DIR=/cs/labs/sagieb/shahaf_levy/uv_cache_deb13
# debian13 nodes ship the driver libs in /etc/lib64/nvidia but ld.so.cache is stale and nvidia-smi is absent.
export LD_LIBRARY_PATH=/etc/lib64/nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
uv --version && uv sync --frozen || { echo "FATAL: uv sync failed"; exit 1; }
ARCH=$(uv run --no-sync python -c "import torch;print('%d%d'%torch.cuda.get_device_capability())" 2>/dev/null || echo unknown)
export TORCH_EXTENSIONS_DIR=$HOME/.cache/torch_ext_deb13_sm${ARCH}
echo "node=$(hostname) debian $(cat /etc/debian_version) sm_${ARCH} driver=$(head -1 /proc/driver/nvidia/version)"; echo "glibc: $(ldd --version|head -1)"
uv run --no-sync python - <<'PY'
import torch, sys; print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda, "ok", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
try:
    import flash_attn; print("flash_attn", flash_attn.__version__)
except Exception as e: print("flash_attn MISSING/BROKEN:", e)
import gsplat; print("gsplat", gsplat.__version__)
from gsplat import rasterization; print("gsplat CUDA ext loaded")
PY
[ $? -eq 0 ] || { echo "FATAL: import check failed"; exit 1; }
PYTHONPATH=. SCENE=octopus CKPT=checkpoints_codec_so_octopus_r2/phase2_epoch_27.pt TAG=deb13_smoke_octopus_r2 \
  uv run --no-sync python data_external/codec_scaleout_eval.py && echo "SMOKE OK" || { echo "FATAL: eval failed"; exit 1; }
