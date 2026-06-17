#!/bin/zsh
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gg:g4:6
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=0:25:00
#SBATCH --output=runs/probe_mn_%j.out
#SBATCH --job-name=v16_mnprobe
#SBATCH --account=sagieb
#SBATCH --killable
#SBATCH --requeue

# Multi-node throughput probe: 2 nodes x 6 GPUs = 12, vs the known 8-GPU single-node baseline
# (6.2 samples/s at 256 bs=1). Measures whether inter-node DDP all-reduce scales at bs=1 or is
# comms-bound. Runs ~2 phase-2 epochs on the small subset (no staging needed). If 12 gives
# >~8 samples/s it scales (>1.3x); if ~6 or less, multi-node is a wash.

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
export MASTER_PORT=29517
SUB=data_v10/probe_subset
LOG=/tmp/probe_mn.log
print "nodes: $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')  master=$MASTER_ADDR"

( srun --ntasks-per-node=1 bash -c "
    cd '$SLURM_SUBMIT_DIR'
    uv run --frozen torchrun --nnodes=\$SLURM_NNODES --nproc_per_node=6 \
      --node_rank=\$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
      -m training.train --gaussian_h5_dir $SUB/h5s --renders_dir $SUB/renders \
      --save_dir /tmp/probe_mn --batch_size 1 --resolution 256 \
      --pe_type rope --augment_rotation \
      --phase1_epochs 0 --phase2_epochs 2 --save_interval 9999 \
      --log_loss_weight 1.0 --lpips_loss_weight 0.0 --num_workers 8
  " ) 2>&1 | while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%N)" "$l"; done | tee $LOG

print "\n==== MULTINODE RESULT (256 bs=1, 12 GPUs / 2 nodes) ===="
python3 - "$LOG" <<'PY'
import sys,re
ts=[(float(m.group(1)),int(m.group(2))) for line in open(sys.argv[1],errors='ignore')
    if (m:=re.match(r'(\d+\.\d+).*\[phase2\] step (\d+)',line))]
W=12  # world size
if len(ts)>=4:
    (t0,s0),(t1,s1)=ts[2],ts[-1]   # drop first 2 logged pts (warmup)
    sps=(s1-s0)/(t1-t0) if t1>t0 else 0
    print(f"12-GPU: {sps:.2f} step/s  {sps*W:.1f} samples/s   (baseline 8-GPU = 6.2 samples/s)")
    print(f"scaling vs 8-GPU: {sps*W/6.2:.2f}x   (ideal 12/8 = 1.50x)")
else:
    print(f"INCONCLUSIVE: only {len(ts)} step points - check {sys.argv[1]} and the .out for NCCL/rendezvous errors")
PY
print "==== DONE ===="
