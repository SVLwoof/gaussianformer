#!/bin/zsh
#SBATCH --time=12:00:00
#SBATCH -c 16
#SBATCH --mem=64GB
#SBATCH --output=runs/nsweep_logscale_%j.out
#SBATCH --job-name=nswlogscale
#SBATCH --gres=gg:g4:4
#SBATCH --account=sagieb
#SBATCH --killable
#SBATCH --requeue
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08,khan-01,khan-02

# N=100 MEMORIZATION CONTROL (aug OFF, wd=0) -- the second control, at the N where the sweep's
# fit visibly broke (13.40 dB on its own 100 objects under the standard recipe).
# Same nested n100 subset, same init, same 30k-step budget, same 4xbs1 layout as the sweep run,
# so THIS pair differs in exactly (aug, wd). Guaranteed quota (Sagie 4), non-preemptible.
#   sbatch data_v10/train_nsweep_logscale.sh

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

SEED=checkpoints_v18_256/phase2_epoch_30.pt
SAVE_R=checkpoints_nsweep_n100_logscale_256
SAVE_B=checkpoints_nsweep_n100_logscale
echo "LOGSCALE: log10(scale)+3 input | stage R 256x100 -> stage B 512x300 | node=$(hostname)"

# --- Stage R: 256 log-L1 recovery from the pruned init ---
if [ ! -f $SAVE_R/phase2_epoch_100.pt ]; then
  RES_R=()
  pR=( ${SAVE_R}/phase2_epoch_*.pt(Nom) )
  if [ ${#pR} -gt 0 ]; then RES_R=(--resume ${pR[1]}); echo "stage R RESUME from ${pR[1]}";
  else RES_R=(--init_from $SEED); fi
  uv run --no-sync torchrun --standalone --nproc_per_node=4 -m training.train \
    --gaussian_h5_dir data_v10/nsweep/n100_h5 --renders_dir data_v10/nsweep/n100_renders \
    --val_h5_dir data_v10/nsweep/val100_h5 --val_renders_dir data_v10/nsweep/val100_renders \
    --save_dir $SAVE_R --batch_size 1 --resolution 256 \
    --pe_type rope --augment_rotation --views_per_epoch 4 \
    --log_scale_input \
    --phase2_epochs 100 --phase2_lr 5e-5 \
    --save_interval 10 --keep_last_n 2 \
    --log_loss_weight 1.0 --lpips_loss_weight 0.0 \
    --num_workers 8 $RES_R || exit 1
fi

# --- Stage B: the standard N=100 schedule ---
RES_B=()
pB=( ${SAVE_B}/phase2_epoch_*.pt(Nom) )
if [ ${#pB} -gt 0 ]; then RES_B=(--resume ${pB[1]}); echo "stage B RESUME from ${pB[1]}";
else RES_B=(--init_from $SAVE_R/phase2_epoch_100.pt); echo "stage B INIT from stage R"; fi
uv run --no-sync torchrun --standalone --nproc_per_node=4 -m training.train \
  --gaussian_h5_dir data_v10/nsweep/n100_h5 --renders_dir data_v10/nsweep/n100_renders \
  --val_h5_dir data_v10/nsweep/val100_h5 --val_renders_dir data_v10/nsweep/val100_renders \
  --save_dir $SAVE_B --batch_size 1 --resolution 512 \
  --pe_type rope --augment_rotation --views_per_epoch 4 \
  --log_scale_input \
  --phase2_epochs 300 --phase2_lr 5e-5 \
  --save_interval 20 --keep_last_n 2 \
  --log_loss_weight 0.5 --lpips_loss_weight 0.5 \
  --num_workers 8 $RES_B
