#!/bin/zsh
#SBATCH --time=2:00:00
#SBATCH -c 32
#SBATCH --mem=64GB
#SBATCH --output=runs/probe_khan_%j.out
#SBATCH --job-name=v16_probe_khan
#SBATCH --gres=gg:g4:8
#SBATCH --nodelist=khan-01
#SBATCH --account=sagieb
#SBATCH --killable
#SBATCH --requeue

# Batch/throughput probe on a khan node (RTX Pro 6000, 96 GB) — faster GPU + 2x VRAM vs the
# a40 baseline (256 bs=1 = 6.2 samples/s). Tests: (1) how much faster khan is per-GPU,
# (2) whether bigger batch finally helps throughput on a faster GPU (it didn't on the
# saturated a40), (3) max batch in 96 GB. Same harness: clean drain between configs.

source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

SUB=data_v10/probe_subset
CONFIGS=("256 1" "256 2" "256 4" "256 8" "512 1" "512 2" "512 4")

maxmem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1; }
drain() {
  pkill -9 -f "training.train" 2>/dev/null; pkill -9 -f "torchrun" 2>/dev/null
  pkill -9 -f "torch.distributed" 2>/dev/null
  for k in {1..30}; do m=$(maxmem); [ "${m:-9999}" -lt 2000 ] && break; sleep 2; done
  print "  [drain] max GPU mem now ${m}MB"
}

print "==== V16 BATCH PROBE on $(hostname) (RTX Pro 6000, 96GB) ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
drain
for cfg in $CONFIGS; do
  res=${cfg% *}; bs=${cfg#* }
  log=/tmp/khan_${res}_${bs}.log; vram=/tmp/khanv_${res}_${bs}.txt
  rm -rf /tmp/probe_ckpt 2>/dev/null; rm -f $log $vram 2>/dev/null
  print "\n---- res=$res bs=$bs (eff=$((bs*8))) ----"
  ( while true; do maxmem; sleep 1; done ) > $vram &
  samp=$!
  ( timeout --signal=KILL 300 uv run --frozen torchrun --standalone --nproc_per_node=8 -m training.train \
      --gaussian_h5_dir $SUB/h5s --renders_dir $SUB/renders \
      --save_dir /tmp/probe_ckpt --batch_size $bs --resolution $res \
      --pe_type rope --augment_rotation \
      --phase1_epochs 0 --phase2_epochs 2 --save_interval 9999 \
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
    samp=sps*int(bs)*8
    print(f"RESULT res={res} bs={bs}: OK  peakVRAM={peak}MB  {sps:.2f} step/s  {samp:.1f} samples/s  ({samp/6.2:.2f}x vs a40-bs1)")
else:
    print(f"RESULT res={res} bs={bs}: INCONCLUSIVE peakVRAM={peak}MB pts={len(ts)}")
PY
  drain
done
print "\n==== KHAN PROBE DONE ===="
