#!/bin/zsh
#SBATCH --time=00:40:00
#SBATCH -c 32
#SBATCH --mem=200GB
#SBATCH --output=runs/probe_v18_512_bs2_%j.out
#SBATCH --job-name=gformer_probe512
#SBATCH --gres=gg:g4:4
#SBATCH --killable
#SBATCH --requeue
#SBATCH --exclude=khan-01,khan-02,cyril-01

# VRAM probe for V18 stage B: does --batch_size 2 --resolution 512 WITH LPIPS fit?
#
# Why this needs to be a real training step, not a forward pass:
#   compute_loss() calls LPIPS-VGG INSIDE the autocast block and inside the autograd graph
#   (training/train.py:178), so VGG16's feature-pyramid activations over a 512x512 pred AND
#   target are retained for backward. That backward pass is the memory peak. An lpips_w=0 or
#   no_grad probe under-reports it badly.
#
# Why --exclude: the gg:g4 gres spans HETEROGENEOUS VRAM -- l40s/a40 45G, a6000 48G, but
#   khan's rtxpro6000 is 96G. Stage B requests plain gg:g4:4 and could land anywhere, so the
#   binding case is 45G. Probing on khan would be a false pass. Excluded here on purpose.
#
# Short by construction: --max_samples 400 (=50 steps/rank at 4x bs2) and --phase2_epochs 1.
# --save_interval 999 so the single epoch never writes a checkpoint. No val dirs -- the val
# loop runs under no_grad and is strictly cheaper than the train step we're measuring.
#
#   sbatch data_v10/probe_v18_512_bs2.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True   # must match stage B
export PATH="$HOME/.local/bin:$PATH"                 # submitted with a clean env; uv lives here

echo "node: $(hostname)  gpus: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | paste -sd'; ')"

GH5=data_v10/h5s_20k_rec; REN=data_v10/renders

seed=( checkpoints_v18_256/phase2_epoch_*.pt(Nom) )
if [ ${#seed} -eq 0 ]; then echo "FATAL: no checkpoints_v18_256/phase2_epoch_*.pt to init from"; exit 1; fi
echo "INIT from ${seed[1]}"

# Poll real VRAM use in the background -- catches fragmentation//allocator overhead that
# torch.cuda.max_memory_allocated() would hide. Killed when training exits.
( while true; do
    echo "[vram $(date +%T)] $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | paste -sd' | ')"
    sleep 15
  done ) &
POLLER=$!
trap "kill $POLLER 2>/dev/null" EXIT

uv run --frozen torchrun --standalone --nproc_per_node=4 -m training.train \
  --gaussian_h5_dir $GH5 --renders_dir $REN \
  --save_dir tmp/probe_512bs2 \
  --batch_size 2 --resolution 512 \
  --pe_type rope --augment_rotation --views_per_epoch 4 \
  --max_samples 400 \
  --phase1_epochs 5 --phase1_lr 1e-3 \
  --phase2_epochs 1 --phase2_lr 5e-5 \
  --save_interval 999 \
  --log_loss_weight 0.5 --lpips_loss_weight 0.5 \
  --num_workers 8 --init_from ${seed[1]}

echo "EXIT CODE: $?"
