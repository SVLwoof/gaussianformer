#!/bin/zsh
#SBATCH --time=24:00:00
#SBATCH -c 16
#SBATCH --mem=64GB
#SBATCH --output=runs/nsweep_scratch2_%j.out
#SBATCH --job-name=nswscr2
#SBATCH --gres=gg:g4:4
#SBATCH --account=sagieb
#SBATCH --killable
#SBATCH --requeue
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08,khan-01,khan-02

# WIDTH-CAPACITY PAIR, TAKE 2 -- with a matched 256 warmup stage for both arms.
#
# Take 1 (train_nsweep_scratch.sh) put from-scratch models straight into 512+LPIPS and BOTH arms
# plateaued at LPIPS ~0.093 for 200+ epochs (d384 burned 27k steps flat) -- the V16-phase1
# failure mode: from-scratch never takes off in the deep-supervision 512+LPIPS regime. Every run
# that ever worked went through a low-res log-L1 stage first. So, per curriculum:
#   stage A: 256, log-L1 only, from scratch, 200 epochs (~20k steps)
#   stage B: 512, log 0.5 + LPIPS 0.5, init from stage A, 400 epochs (~40k steps)
# IDENTICAL treatment for d=768 and d=384, so the width comparison stays clean.
# Readout unchanged: train-fit margin vs ceiling on the same n100 objects.
#   sbatch --export=D=768 data_v10/train_nsweep_scratch2.sh
#   sbatch --export=D=384 data_v10/train_nsweep_scratch2.sh

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
SAVE_A=checkpoints_nsweep_n100_scratch2_d${D}_256
SAVE_B=checkpoints_nsweep_n100_scratch2_d${D}
echo "SCRATCH2 d=$D | stage A 256x200ep -> stage B 512x400ep | node=$(hostname) sm_${ARCH}"

# --- Stage A: 256 log-L1 from scratch (skipped once its final ckpt exists) ---
if [ ! -f $SAVE_A/phase2_epoch_200.pt ]; then
  RES_A=()
  pA=( ${SAVE_A}/phase2_epoch_*.pt(Nom) )
  [ ${#pA} -gt 0 ] && { RES_A=(--resume ${pA[1]}); echo "stage A RESUME from ${pA[1]}"; }
  uv run --no-sync torchrun --standalone --nproc_per_node=4 -m training.train \
    --gaussian_h5_dir data_v10/nsweep/n100_h5 --renders_dir data_v10/nsweep/n100_renders \
    --val_h5_dir data_v10/nsweep/val100_h5 --val_renders_dir data_v10/nsweep/val100_renders \
    --save_dir $SAVE_A --batch_size 1 --resolution 256 \
    --pe_type rope --views_per_epoch 4 \
    --from_scratch --latent_dim $D \
    --phase2_epochs 200 --phase2_lr 5e-5 \
    --save_interval 20 --keep_last_n 2 \
    --log_loss_weight 1.0 --lpips_loss_weight 0.0 \
    --num_workers 8 $RES_A || exit 1
fi

# --- Stage B: 512 + LPIPS, init from stage A ---
RES_B=()
pB=( ${SAVE_B}/phase2_epoch_*.pt(Nom) )
if [ ${#pB} -gt 0 ]; then
  RES_B=(--resume ${pB[1]}); echo "stage B RESUME from ${pB[1]}"
else
  RES_B=(--init_from $SAVE_A/phase2_epoch_200.pt); echo "stage B INIT from stage A"
fi
uv run --no-sync torchrun --standalone --nproc_per_node=4 -m training.train \
  --gaussian_h5_dir data_v10/nsweep/n100_h5 --renders_dir data_v10/nsweep/n100_renders \
  --val_h5_dir data_v10/nsweep/val100_h5 --val_renders_dir data_v10/nsweep/val100_renders \
  --save_dir $SAVE_B --batch_size 1 --resolution 512 \
  --pe_type rope --views_per_epoch 4 \
  --latent_dim $D \
  --phase2_epochs 400 --phase2_lr 5e-5 \
  --save_interval 20 --keep_last_n 2 \
  --log_loss_weight 0.5 --lpips_loss_weight 0.5 \
  --num_workers 8 $RES_B
