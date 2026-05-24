#!/bin/zsh
#SBATCH --time=04:00:00
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --output=runs/regen_data_n20k_%j.out
#SBATCH --job-name=regen_n20k
#SBATCH --gres=gg:g4:1

# Re-prune the Objaverse_Splats sources to N=20,000 Gaussians per scene
# (vs the existing N=5,000 in data_v9). Writes new H5s to data_v9_n20k/.
# GT renders are FULL-scene rasterizations and target_n-independent, so we
# pass --skip_renders and reuse data_v9/renders/ at training time.
#
# Runs train + val sequentially in one job. Train ~ 1-1.5h, val ~10 min.
#
# Submit:
#   sbatch runs/regen_data_n20k.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1

echo "=== train split: target_n=20000 -> data_v9_n20k/h5s/ ==="
uv run --frozen python -m data_v9.process_objaverse \
  --object_list data_v9/object_list_train.json \
  --out_dir data_v9_n20k \
  --split train \
  --target_n 20000 \
  --skip_renders

echo ""
echo "=== val split: target_n=20000 -> data_v9_n20k/h5s_val/ ==="
uv run --frozen python -m data_v9.process_objaverse \
  --object_list data_v9/object_list_val.json \
  --out_dir data_v9_n20k \
  --split val \
  --target_n 20000 \
  --skip_renders
