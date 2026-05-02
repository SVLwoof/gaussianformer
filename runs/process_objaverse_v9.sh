#!/bin/zsh
#SBATCH --time=24:00:00
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --output=runs/process_objaverse_v9_%j.out
#SBATCH --job-name=v9_process
#SBATCH --mail-type=END,FAIL,BEGIN
# #SBATCH --mail-user=your-email@example.com
#SBATCH --gres=gg:g4:1

# V9 Phase A: process Objaverse_Splats objects into H5 + GT renders for the
# L1+LPIPS fine-tune. Both train (3000) and val (200) splits run in this
# allocation; each chunk zip is downloaded once (~3.3 GB) and reused.

# Cluster's lmod isn't sourced into non-interactive zsh by default — pull it
# in explicitly so `module load` actually resolves (otherwise the loads
# silently fail and we drift onto whatever CUDA the venv happens to bundle).
source "${LMOD_INIT:-/etc/profile.d/lmod.sh}"
module load nvidia
module load cuda
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1

# Step 1: build object lists from the metadata CSV (cheap, ~1 min).
python -m data_v9.build_object_list \
  --out_dir data_v9 \
  --n_train 3000 --n_val 200 \
  --min_psnr 32.0 --max_lpips 0.06 --num_gs 50000 \
  --n_chunks 12 --seed 0

# Step 2: process train split (downloads needed chunks on first use).
python -m data_v9.process_objaverse \
  --object_list data_v9/object_list_train.json \
  --out_dir data_v9 --split train \
  --target_n 5000 --n_views 14 --resolution 512

# Step 3: process val split (chunks are cached from train run).
python -m data_v9.process_objaverse \
  --object_list data_v9/object_list_val.json \
  --out_dir data_v9 --split val \
  --target_n 5000 --n_views 14 --resolution 512
