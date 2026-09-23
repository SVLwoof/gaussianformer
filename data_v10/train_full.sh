#!/bin/zsh
#SBATCH --time=7-00:00:00
#SBATCH -c 32
#SBATCH --mem=96GB
#SBATCH --output=runs/full_%j.out
#SBATCH --job-name=gf_full
#SBATCH --gres=gg:g4:8
#SBATCH --requeue
#SBATCH --exclude=cyril-01,firefoot-01,firefoot-08,khan-01,khan-02
# Full-data run of the settled recipe (2026-09-23): 512/4 windowed + P2 + fg 0.05 + multi-radius views (28/object)
# + LPIPS 0.5, warmup-stable-decay LR (training/train_full.py). Resume-safe end to end: resubmit the same command.
#   stage R  256 px, log-L1, STAGE_R steps (default 3000, cosine-decayed) from the v18_256 seed -- the probes' recovery
#   main     512 px, STEPS optimizer steps (default 375000 = 112 views/object at 8 GPUs), stable LR (DECAY=0) ->
#            extendable; one milestone per pass over every view (MILESTONE steps), rolling checkpoints otherwise.
# Freeze: touch $SAVE/FREEZE (${SAVE}_r/FREEZE during stage R) -> checkpoint + clean exit within 20 steps; resubmit.
#   sbatch -A sagieb data_v10/train_full.sh
#   smoke: sbatch -A sagieb --time=1:00:00 --export=PATH,HOME,USER,SAVE=checkpoints_full_smoke,STAGE_R=40,STEPS=200,SAVE_EVERY=100 data_v10/train_full.sh
source /etc/profile.d/huji-lmod.sh
module load nvidia
module load cuda
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.local/bin:$PATH"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_ext_sm${ARCH:-unknown}"

SAVE=${SAVE:-checkpoints_full_p4}
STAGE_R=${STAGE_R:-3000}; STEPS=${STEPS:-375000}; DECAY=${DECAY:-0}
SAVE_EVERY=${SAVE_EVERY:-2000}; MILESTONE=${MILESTONE:-94000}
SEED=${SEED:-checkpoints_v18_256/phase2_epoch_30.pt}
CFG=(proj_rope_2d=true patch_size=4 ray_embed_patch=8 xattn_window=8 view_bf16=true view_grad_checkpoint=true)
DATA=(--gaussian_h5_dir data_v10/h5s_20k_rec_r3 --renders_dir data_v10/renders_r3
      --val_h5_dir data_v10/nsweep/val100_h5 --val_renders_dir data_v10/nsweep/val100_renders --fg_bg_weight 0.05)
RUN=(uv run --no-sync torchrun --standalone --nproc_per_node=8 -m training.train_full)
echo "FULL save=$SAVE stage_r=$STAGE_R steps=$STEPS decay=$DECAY node=$(hostname) sm_${ARCH}"

if [ ! -f ${SAVE}_r/full_step_${STAGE_R}.pt ]; then
  "${RUN[@]}" "${DATA[@]}" --save_dir ${SAVE}_r --init_from $SEED --model_cfg "${CFG[@]}" --resolution 256 \
    --steps $STAGE_R --warmup_steps $(( STAGE_R / 30 + 1 )) --decay_steps $(( STAGE_R / 3 )) \
    --log_loss_weight 1.0 --lpips_loss_weight 0.0 --save_every $(( STAGE_R / 3 )) || exit 1
fi
"${RUN[@]}" "${DATA[@]}" --save_dir $SAVE --init_from ${SAVE}_r/full_step_${STAGE_R}.pt --resolution 512 \
  --steps $STEPS --decay_steps $DECAY --save_every $SAVE_EVERY --milestone_every $MILESTONE
