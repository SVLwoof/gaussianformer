#!/bin/zsh
#SBATCH --time=18:00:00
#SBATCH -c 16
#SBATCH --mem=64GB
#SBATCH --output=runs/nsweep_expand_%j.out
#SBATCH --job-name=nswexpand
#SBATCH --gres=gg:g4:4
#SBATCH --account=sagieb
#SBATCH --requeue
#SBATCH --killable
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08,khan-01,khan-02

# V19 DE-RISK PROBE: warm depth-EXPANSION (enc 12->18, view 6->9, 194.9M->286.9M) at N=100.
# Init = identity-preserving insertion (make_expanded_ckpt.py; equivalence verified bit-exact,
# so epoch 0 IS v18_256-ep30 -- the 0.092 attractor is unreachable). Schedule = EXACTLY the
# N=100 sweep baseline. The capacity story predicts train-fit margin < 13.40 dB, mirroring the
# shrink result (114.6M -> 17.23). If it lands materially under 13.40, depth expansion is
# validated as the V19 lever BEFORE committing the full-data run. Guaranteed quota; 287M @ bs1
# @512 is untested on 45G -- watch the first step for OOM (fallback: smaller expansion).
#   arms: sbatch --export=ENC=18,VIEW=6 ...  |  sbatch --export=ENC=12,VIEW=9 ...

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.local/bin:$PATH"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_ext_sm${ARCH:-unknown}"
mkdir -p "$TORCH_EXTENSIONS_DIR"

ENC=${ENC:?}
VIEW=${VIEW:?}
SEED=checkpoints_nsweep_n100_base_r2/phase2_epoch_300.pt
SAVE=checkpoints_nsweep_n100_base_r3
echo "BASE-R2: fresh cosine on trained BASELINE (control for the restart gain)"

RESUME_ARG=()
p2=( ${SAVE}/phase2_epoch_*.pt(Nom) )
if [ ${#p2} -gt 0 ]; then
  RESUME_ARG=(--resume ${p2[1]}); echo "RESUME from ${p2[1]}"
else
  [ -f $SEED ] || { echo "FATAL: missing $SEED"; exit 1; }
  RESUME_ARG=(--init_from $SEED); echo "INIT from $SEED"
fi

uv run --no-sync torchrun --standalone --nproc_per_node=4 -m training.train \
  --gaussian_h5_dir data_v10/nsweep/n100_h5 --renders_dir data_v10/nsweep/n100_renders \
  --val_h5_dir data_v10/nsweep/val100_h5 --val_renders_dir data_v10/nsweep/val100_renders \
  --save_dir $SAVE --batch_size 1 --resolution 512 \
  --pe_type rope --augment_rotation --views_per_epoch 4 \
  \
  --phase2_epochs 300 --phase2_lr 5e-5 \
  --save_interval 20 --keep_last_n 2 \
  --log_loss_weight 0.5 --lpips_loss_weight 0.5 \
  --num_workers 8 $RESUME_ARG
