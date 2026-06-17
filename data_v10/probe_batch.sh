#!/bin/zsh
#SBATCH --time=2:00:00
#SBATCH -c 32
#SBATCH --mem=64GB
#SBATCH --output=runs/probe_batch_%j.out
#SBATCH --job-name=v16_probe
#SBATCH --gres=gg:g4:8
#SBATCH --account=sagieb
#SBATCH --killable
#SBATCH --requeue

# V16 batch/throughput probe. For each (resolution, batch_size): run ~1 phase-2 epoch (full
# backward + Adam = the VRAM-critical phase) on a small symlinked subset, full views. Capture
# peak per-GPU VRAM + steady-state step time. CRITICAL: between configs, hard-kill all workers
# and WAIT for GPU memory to drain (the previous probe leaked memory across configs -> false
# OOMs), verifying drainage before the next config.

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

SUB=data_v10/probe_subset
CONFIGS=("256 1" "256 2" "256 4" "512 1" "512 2")

maxmem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1; }
drain() {  # hard-kill workers, wait until GPU mem < 2000MB (or give up after ~60s)
  pkill -9 -f "training.train" 2>/dev/null; pkill -9 -f "torchrun" 2>/dev/null
  pkill -9 -f "torch.distributed" 2>/dev/null
  for k in {1..30}; do m=$(maxmem); [ "${m:-9999}" -lt 2000 ] && break; sleep 2; done
  print "  [drain] max GPU mem now ${m}MB"
}

print "==== V16 BATCH PROBE (clean) ===="
drain
for cfg in $CONFIGS; do
  res=${cfg% *}; bs=${cfg#* }
  log=/tmp/probe_${res}_${bs}.log; vram=/tmp/vram_${res}_${bs}.txt
  rm -rf /tmp/probe_ckpt 2>/dev/null; rm -f $log $vram 2>/dev/null
  print "\n---- res=$res bs=$bs (eff=$((bs*8))) ----"
  ( while true; do maxmem; sleep 1; done ) > $vram &
  samp=$!
  ( timeout --signal=KILL 300 uv run --frozen torchrun --standalone --nproc_per_node=8 -m training.train \
      --gaussian_h5_dir $SUB/h5s --renders_dir $SUB/renders \
      --save_dir /tmp/probe_ckpt --batch_size $bs --resolution $res \
      --pe_type rope --augment_rotation \
      --phase1_epochs 0 --phase2_epochs 1 --save_interval 9999 \
      --log_loss_weight 1.0 --lpips_loss_weight 0.0 --num_workers 8 2>&1 ) \
    | while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%N)" "$l"; done > $log
  kill $samp 2>/dev/null
  peak=$(sort -n $vram 2>/dev/null | tail -1)
  oom=$(grep -ciE "out of memory|CUDA error" $log)
  python3 - "$log" "$res" "$bs" "$peak" "$oom" <<'PY'
import sys,re
log,res,bs,peak,oom=sys.argv[1:6]
ts=[(float(m.group(1)),int(m.group(2))) for line in open(log,errors='ignore')
    if (m:=re.match(r'(\d+\.\d+).*\[phase2\] step (\d+)',line))]
if int(oom)>0:
    print(f"RESULT res={res} bs={bs}: OOM  peakVRAM={peak}MB")
elif len(ts)>=3:
    (t0,s0),(t1,s1)=ts[1],ts[-1]
    sps=(s1-s0)/(t1-t0) if t1>t0 else 0
    print(f"RESULT res={res} bs={bs}: OK  peakVRAM={peak}MB  {sps:.2f} step/s  {sps*int(bs)*8:.1f} samples/s  ({len(ts)} pts)")
else:
    print(f"RESULT res={res} bs={bs}: INCONCLUSIVE peakVRAM={peak}MB pts={len(ts)}")
PY
  drain
done
print "\n==== PROBE DONE ===="
