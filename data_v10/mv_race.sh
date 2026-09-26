#!/bin/zsh
#SBATCH --time=16:00:00
#SBATCH -c 16
#SBATCH --mem=64GB
#SBATCH --output=runs/mvrace_%x_%j.out
#SBATCH --gres=gg:g4:4
#SBATCH --requeue
#SBATCH --exclude=cyril-01,epona-01,epona-02,khan-01,khan-02,firefoot-01,firefoot-08
# Multi-view race at N=1000 (2026-09-26): train_full from the full run's stage-R checkpoint on 1,000 objects x 28 views,
# K views per object per step, STEPS steps with the last 20% decayed. Arms: A K=1 15k, B K=4 15k, C K=4 6k.
#   sbatch -A sagieb -J mv_A --export=PATH,HOME,USER,K=1,STEPS=15000 data_v10/mv_race.sh
source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.local/bin:$PATH" UV_PROJECT_ENVIRONMENT=/cs/labs/sagieb/shahaf_levy/gaussianformer/.venv
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_ext_sm${ARCH:-unknown}"
: ${K:?} ${STEPS:?}
SAVE=/cs/labs/sagieb/shahaf_levy/gaussianformer/checkpoints_mvrace_k${K}_s${STEPS}
echo "MVRACE K=$K steps=$STEPS save=$SAVE node=$(hostname) sm_${ARCH}"
uv run --no-sync torchrun --standalone --nproc_per_node=4 -m training.train_full \
  --gaussian_h5_dir data_v10/nsweep/n1000_r3_h5 --renders_dir data_v10/renders_r3 \
  --val_h5_dir data_v10/nsweep/val100_h5 --val_renders_dir data_v10/nsweep/val100_renders --fg_bg_weight 0.05 \
  --save_dir $SAVE --init_from /cs/labs/sagieb/shahaf_levy/gaussianformer/checkpoints_full_p4_r/full_step_3000.pt \
  --resolution 512 --steps $STEPS --decay_steps $(( STEPS / 5 )) --save_every $(( STEPS / 5 )) --views_per_object $K
