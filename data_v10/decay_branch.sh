#!/bin/zsh
#SBATCH --time=8:00:00
#SBATCH -c 32
#SBATCH --mem=96GB
#SBATCH --output=runs/decay_%j.out
#SBATCH --job-name=gf_decay
#SBATCH --gres=gg:g4:8
#SBATCH --requeue
#SBATCH --exclude=cyril-01,epona-01,epona-02,khan-01,khan-02,firefoot-01,firefoot-08
# Decay branch off a stable checkpoint of the full-data run (training/train_full.py): same data and recipe, LR
# cosine-decayed to the floor over DECAY steps, so the evaluated model is properly cooled. Resumes automatically.
#   sbatch -A sagieb --export=PATH,HOME,USER,FROM=checkpoints_full_p4/milestone_step_94000.pt data_v10/decay_branch.sh
source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.local/bin:$PATH"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_ext_sm${ARCH:-unknown}"

: ${FROM:?FROM=<stable checkpoint>}
DECAY=${DECAY:-10000}
AT=${${FROM:t:r}##*_}                       # step of the source checkpoint
SAVE=${SAVE:-checkpoints_full_p4_d${AT}}
echo "DECAY branch from=$FROM (step $AT) decay=$DECAY save=$SAVE node=$(hostname) sm_${ARCH}"
uv run --no-sync torchrun --standalone --nproc_per_node=8 -m training.train_full \
  --gaussian_h5_dir data_v10/h5s_20k_rec_r3 --renders_dir data_v10/renders_r3 \
  --val_h5_dir data_v10/nsweep/val100_h5 --val_renders_dir data_v10/nsweep/val100_renders --fg_bg_weight 0.05 \
  --save_dir $SAVE --resume_from $FROM --resolution 512 \
  --steps $(( AT + DECAY )) --decay_steps $DECAY --save_every 2500
