#!/bin/zsh
#SBATCH --time=1:00:00
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH --gres=gg:g4:1
#SBATCH --output=runs/diag_val_%j.out
#SBATCH --job-name=v16_diag_val
#SBATCH --account=sagieb
#SBATCH --killable
#SBATCH --requeue

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

print "=== A/B: V16 vs V15 phase1_epoch_5 val log-L1 (lower=better; V15 trained to ~0.0037) ==="
print "--- V16 phase1_epoch_5 ---"
uv run --frozen python -m data_v10.diag_val checkpoints_v16_256/phase1_epoch_5.pt 400 256
print "--- V15 phase1_epoch_5 (reference; should be ~0.0037) ---"
uv run --frozen python -m data_v10.diag_val checkpoints_v15_256/phase1_epoch_5.pt 400 256
print "=== DIAG DONE ==="
