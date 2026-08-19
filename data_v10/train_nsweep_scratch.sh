#!/bin/zsh
#SBATCH --time=12:00:00
#SBATCH -c 16
#SBATCH --mem=64GB
#SBATCH --output=runs/nsweep_scratch_%j.out
# (D is stamped into SAVE + the log line)
#SBATCH --job-name=nswscratch
#SBATCH --gres=gg:g4:4
#SBATCH --account=sagieb
#SBATCH --killable
#SBATCH --requeue
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08

# N=100 MEMORIZATION CONTROL (aug OFF, wd=0) -- the second control, at the N where the sweep's
# fit visibly broke (13.40 dB on its own 100 objects under the standard recipe).
# Same nested n100 subset, same init, same 30k-step budget, same 4xbs1 layout as the sweep run,
# so THIS pair differs in exactly (aug, wd). Guaranteed quota (Sagie 4), non-preemptible.
#   sbatch data_v10/train_nsweep_scratch.sh

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

D=${D:?D required (768|384)}
SAVE=checkpoints_nsweep_n100_scratch_d${D}
EPOCHS=600   # 60k steps: from-scratch needs more than the warm 30k
echo "SCRATCH d=$D | 4 GPUs, $EPOCHS epochs, node=$(hostname)"

RESUME_ARG=()
p2=( ${SAVE}/phase2_epoch_*.pt(Nom) )
[ ${#p2} -gt 0 ] && { RESUME_ARG=(--resume ${p2[1]}); echo "RESUME from ${p2[1]}"; }

uv run --no-sync torchrun --standalone --nproc_per_node=4 -m training.train \
  --gaussian_h5_dir data_v10/nsweep/n100_h5 \
  --renders_dir     data_v10/nsweep/n100_renders \
  --val_h5_dir      data_v10/nsweep/val100_h5 \
  --val_renders_dir data_v10/nsweep/val100_renders \
  --save_dir $SAVE \
  --batch_size 1 --resolution 512 \
  --pe_type rope --augment_rotation --views_per_epoch 4 \
  --from_scratch --latent_dim $D \
  --phase2_epochs $EPOCHS --phase2_lr 5e-5 \
  --save_interval 20 --keep_last_n 2 \
  --log_loss_weight 0.5 --lpips_loss_weight 0.5 \
  --num_workers 8 $RESUME_ARG
