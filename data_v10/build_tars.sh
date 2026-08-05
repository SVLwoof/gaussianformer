#!/bin/zsh
#SBATCH --time=6:00:00
#SBATCH -c 32
#SBATCH --mem=16GB
#SBATCH --output=runs/build_tars_%j.out
#SBATCH --job-name=v15_build_tars
#SBATCH --account=sagieb
#SBATCH --killable
#SBATCH --requeue

# One-time: pack the recovered-20k h5s + GT renders into a few big tars on NFS, so each V15
# training job stages by extracting a handful of large sequential files instead of cp-ing
# ~214k tiny files (the small-file NFS metadata tax was ~1.8 MB/s -> ~4.5 h/job). Reading the
# loose files once here is the same slow cost, but PARALLELISED across shards (overlapping NFS
# round-trips beats one D-state reader) and paid ONCE -- reused by all 3 stages + every requeue.
# Resume-safe: skips any shard tar that already exists (killable/requeue friendly).
#
#   sbatch data_v10/build_tars.sh

source /etc/profile.d/huji-lmod.sh
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

SRC=data_v10
OUT=data_v10/tars
LISTS=$OUT/.lists
mkdir -p $OUT $LISTS

# Build N shard-tars of a flat directory in parallel, by splitting its file list.
# args: <src_subdir> <out_prefix> <n_shards>
build_sharded() {
  local sub=$1 prefix=$2 n=$3
  local listfile=$LISTS/${prefix}.all
  [ -f "$listfile" ] || ( cd $SRC/$sub && ls -U > "$OLDPWD/$listfile" )
  split -d -n l/$n "$listfile" $LISTS/${prefix}_   # even split into n parts (00..)
  local pids=()
  for part in $LISTS/${prefix}_[0-9]*; do
    local idx=${part##*_}
    local tarf=$OUT/${prefix}_${idx}.tar
    if [ -s "$tarf" ]; then echo "skip existing $tarf"; continue; fi
    ( tar -cf "$tarf.partial" -C $SRC/$sub -T "$part" && mv "$tarf.partial" "$tarf" \
        && echo "done $tarf ($(du -h "$tarf"|cut -f1))" ) &
    pids+=($!)
  done
  wait $pids
}

echo "=== building tars at $(date) ==="
# renders is the long pole (187k tiny files) -> 16 shards; h5s (13.4k files) -> 8.
build_sharded renders        renders     16
build_sharded h5s_20k_rec    h5s          8
build_sharded renders_val    renders_val  2
build_sharded h5s_20k_rec_val h5s_val     1
echo "=== all tars built at $(date) ==="
ls -lh $OUT/*.tar
echo "total: $(du -sh $OUT | cut -f1)"
