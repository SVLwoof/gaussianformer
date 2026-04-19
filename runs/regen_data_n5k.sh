#!/bin/zsh
#SBATCH --time=12:00:00
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH --output=runs/regen_data_n5k_%j.out
#SBATCH --job-name=regen_n5k
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@example.com
#SBATCH --gres=gg:g4:1

# Regenerate the v2 dataset with max_gaussians=5000 (up from 3000) into a
# parallel set of dirs so the n=3000 baseline stays intact. To revert, just
# point training back at data_v2/{objects,h5s,h5s_val,renders,renders_val}.

module load nvidia
module load cuda
cd /path/to/gaussianformer
source .venv/bin/activate
export PYTHONUNBUFFERED=1

N=5000

python data_v2/process_objects.py \
  --raw_dir data_v2/raw_objects \
  --output_dir data_v2/objects_n5k \
  --max_gaussians $N

python data_v2/compose_scenes.py \
  --objects_dir data_v2/objects_n5k \
  --output_dir data_v2/h5s_n5k \
  --num_scenes 1000 \
  --seed 42

python data_v2/compose_scenes.py \
  --objects_dir data_v2/objects_n5k \
  --output_dir data_v2/h5s_n5k_val \
  --num_scenes 100 \
  --seed 1

python data_v2/render_gt.py \
  --h5_dir data_v2/h5s_n5k \
  --output_dir data_v2/renders_n5k

python data_v2/render_gt.py \
  --h5_dir data_v2/h5s_n5k_val \
  --output_dir data_v2/renders_n5k_val
