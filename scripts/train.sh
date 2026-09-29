#!/usr/bin/env bash
# The four training stages of the released model on one node with 8 GPUs (batch 8, one view per GPU).
# Stage 4 takes about 4.5 days on 8 L40S. Every stage resumes from its newest checkpoint when rerun.
# BASE=shahafvl/gaussianformer-256px skips stages 1-2 and starts from the published stage-2 checkpoint.
set -euo pipefail
RUN=(torchrun --standalone --nproc_per_node="${GPUS:-8}" -m training.train
     --data data/train --val_data data/val --val_objects data/splits/val100.json)
FINE=(proj_rope_2d=true xattn_window=8 patch_size=4 ray_embed_patch=8 view_bf16=true view_grad_checkpoint=true)
BASE=${BASE:-checkpoints/2_base/step_402000.pt}

if [[ $BASE == checkpoints/* ]]; then
  # 1. Gaussian input head on the frozen RenderFormer backbone, 256 px, log-L1.
  "${RUN[@]}" --init renderformer --head_only --save_dir checkpoints/1_head --resolution 256 \
      --steps 33500 --decay_steps 33500 --lr 1e-3 --lpips_weight 0 --bg_weight 1
  # 2. Whole model, 256 px, log-L1.
  "${RUN[@]}" --init checkpoints/1_head/step_33500.pt --save_dir checkpoints/2_base --resolution 256 \
      --steps 402000 --decay_steps 402000 --lpips_weight 0 --bg_weight 1
fi

# 3. Projected 2-D RoPE, windowed cross-attention and 4 px patches: a short recovery at 256 px.
"${RUN[@]}" --init "$BASE" --model_cfg "${FINE[@]}" --save_dir checkpoints/3_recovery \
    --resolution 256 --steps 3000 --warmup_steps 101 --decay_steps 1000 --save_every 1000 --lpips_weight 0

# 4. 512 px with log-L1 + LPIPS: four passes over every view, cosine decay over the last 10k steps.
"${RUN[@]}" --init checkpoints/3_recovery/step_3000.pt --save_dir checkpoints/4_main --steps 385000 --decay_steps 10000
