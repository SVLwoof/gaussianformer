#!/bin/zsh
#SBATCH --time=1:30:00
#SBATCH -c 16
#SBATCH --mem=64GB
#SBATCH --job-name=loopbench
#SBATCH --output=runs/loopbench_%j.out
#SBATCH --gres=gg:g4:2
#SBATCH --account=sagieb
#SBATCH --killable
#SBATCH --requeue
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08

# OLD vs NEW training-loop benchmark, both variants on THIS SAME node back to back so the
# comparison can't be poisoned by the heterogeneous GPU pool (identical configs measured 3.5x
# apart across node types -- never compare timings across nodes).
#
# OLD = git HEAD (per-step .item() syncs, blocking H2D copies, uint8-roundtrip resize),
#       run from a temporary git worktree, same venv (--project).
# NEW = the working tree.
# Config mirrors production stage B: 512px, LPIPS 0.5, aug, views_per_epoch 4, bs1, 2 ranks,
# n100 subset -> 200 steps/rank/epoch, 3 epochs each. Epoch 1 is warmup; compare epochs 2-3.

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
REPO=/cs/labs/sagieb/shahaf_levy/gaussianformer
cd $REPO
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.local/bin:$PATH"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_ext_sm${ARCH:-unknown}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
echo "bench node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

WT=/tmp/gf_head_$SLURM_JOB_ID
git worktree add --detach $WT HEAD || exit 1
trap "cd $REPO; git worktree remove --force $WT; rm -rf /tmp/bench_ckpt_$SLURM_JOB_ID" EXIT

run_variant() {
  local name=$1 dir=$2
  echo "=== VARIANT $name (cwd=$dir) ==="
  cd $dir
  uv run --project $REPO --no-sync torchrun --standalone --nproc_per_node=2 -m training.train \
    --gaussian_h5_dir $REPO/data_v10/nsweep/n100_h5 \
    --renders_dir     $REPO/data_v10/nsweep/n100_renders \
    --save_dir /tmp/bench_ckpt_$SLURM_JOB_ID/$name \
    --batch_size 1 --resolution 512 \
    --pe_type rope --augment_rotation --views_per_epoch 4 \
    --phase2_epochs 3 --phase2_lr 5e-5 --save_interval 999 \
    --log_loss_weight 0.5 --lpips_loss_weight 0.5 --num_workers 8 \
    --init_from $REPO/checkpoints_v18_256/phase2_epoch_30.pt \
    2>&1 | grep -E "^\[phase2\] Epoch|Warm-start" | sed "s/^/[$name] /"
  cd $REPO
}

run_variant OLD $WT
run_variant NEW $REPO
echo BENCH_DONE
