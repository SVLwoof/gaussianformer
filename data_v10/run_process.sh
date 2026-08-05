#!/bin/zsh
#SBATCH --time=24:00:00
#SBATCH -c 16
#SBATCH --mem=64GB
#SBATCH --job-name=v10_proc
#SBATCH --output=runs/v10_proc_%x_%j.out
#SBATCH --gres=gg:g4:1
#SBATCH --requeue

# V10 full-splat data gen. Real zsh script (NOT --wrap) so module load cuda works -> gsplat.
# Queue policy is set PER SUBMIT (not baked in): add `--account=sagieb` to run under the Sagie
# entitlement (guaranteed, non-preemptible, 4-GPU group cap), or `--killable` for preemptible slots.
# Pass OBJLIST, SPLIT, OUTDIR via explicit --export (NOT --export=ALL -- leaks parent SLURM GPU
# context from the claude_node -> 'invalid device ordinal'). Resume-friendly (skips done scenes).
#   sbatch --export=ALL,OBJLIST=data_v10/object_list_TEST.json,SPLIT=train,OUTDIR=data_v10_test data_v10/run_process.sh

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PATH="$HOME/.local/bin:$PATH"   # uv lives here; needed under a clean --export (no ALL)
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
# NOTE: this cluster DOES cgroup-isolate the gg:g4 gres -- each job sees only its GPU as device 0,
# so SLURM's CUDA_VISIBLE_DEVICES=0 is correct. Do NOT set CVD=$SLURM_JOB_GPUS (that is the global
# physical index -> "No CUDA GPUs are available" inside the cgroup).
echo "node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"

: ${OBJLIST:?set OBJLIST}
: ${SPLIT:=train}
: ${OUTDIR:=data_v10}

uv run --frozen python -m data_v10.process_full \
  --object_list $OBJLIST --split $SPLIT --out_dir $OUTDIR
