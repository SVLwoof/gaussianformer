#!/bin/zsh
# Submit one N=10 architecture probe: train (probe_n10.sh) -> eval (run_nsweep_eval.sh, afterok).
#   data_v10/submit_probe.sh TAG "MODEL_CFG" STAGE_R ACCOUNT
#   e.g. data_v10/submit_probe.sh p3_rope32 "rope_dim=32 rope_pos_scale=4" 3000 sagieb
#        data_v10/submit_probe.sh baseline "" 0 killable
set -e
TAG=${1:?TAG}; CFG=${2:-}; STAGE_R=${3:-0}; ACCT=${4:-killable}
cd "$(dirname "$0")/.."
if [ "$ACCT" = killable ]; then ACC=(--killable --account=killable-cs); else ACC=(--account=$ACCT); fi
train=$(sbatch "${ACC[@]}" --export=TAG=$TAG,MODEL_CFG="$CFG",STAGE_R=$STAGE_R data_v10/probe_n10.sh | awk '{print $NF}')
EXTRA=""; [ -n "$CFG" ] && EXTRA="--model_cfg $CFG"
N_EPOCHS=$(( 30000 / 10 ))
ev=$(sbatch "${ACC[@]}" --dependency=afterok:$train \
     --export=N=10,TAG=probe_$TAG,CKPT=checkpoints_probe_$TAG/phase2_epoch_$N_EPOCHS.pt,EXTRA="$EXTRA" \
     data_v10/run_nsweep_eval.sh | awk '{print $NF}')
echo "$TAG: train $train -> eval $ev ($ACCT) cfg=[$CFG] stage_r=$STAGE_R"
