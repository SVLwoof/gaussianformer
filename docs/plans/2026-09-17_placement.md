# Plan: the placement bottleneck (post-meeting, 2026-09-17)

Decision (Shahaf + Sagie, 17 Sep): drop the canvas / residual line. Keep the model a pure
transformer renderer and attack the reason it under-fits: **the view stage has no channel for
where inside a patch a Gaussian lands.** Context tokens are view-independent; the only
view-dependent signal per Gaussian is its attention weight. A patch learns *which* Gaussians
are its own, never *where* they sit in its 8x8 pixels. Evidence: error spectrum peaks at 8 px,
14 dB under-fit on train with a 0.55 dB train/test gap, capacity and RoPE-bandwidth arms flat,
P2 (better "which") +1.4 dB, canvas (pixel-aligned "where") +3 dB held-out.

## 0. Repository hygiene (first)

- Tag `v19-campaign` = the state every number in the two reports was produced from.
- Branch `v19/cleanup` off it: remove the null / rejected arms from the model code so the
  next work starts from a small surface. Removed: `canvas_residual`, `deblock_kernel`,
  `proj_bias`, `proj_feat`, `rope_hf_scale`, `rope_pos_scale`, `ray_rope_2d`
  (+ their plumbing, `deblock_zoom.py`). Kept: `proj_rope_2d` (positive), `canvas_cond`
  (needed to load the reported checkpoints; no new work on it), `geom_bias` (kept in Aug by
  decision), `fg_bg_weight`.
- One PR `v19/cleanup -> main` supersedes the eight stale August PRs (#6-#13), all of which
  are ancestors of this branch. Run `/codereview` on that PR, then merge.
- Checkpoints of removed arms proposed for deletion (not executed without a go):
  `checkpoints_probe_p1res_*`, `checkpoints_probe_p4_deblock9`, `checkpoints_probe_p2_bias_feat`,
  `checkpoints_probe_p2_full`, `checkpoints_probe_p3_*`, `checkpoints_probe_p2r_{dim32,scale1,both}`
  (~25 GB). Holds unchanged.
- New work on `v20/placement` off `v19/cleanup`.

## 1. Offset-aware values ("value RoPE") -- the direct test

`value_rope_2d=true`: in the view stage's cross-attention, rotate each Gaussian's value vector
by its projected patch coordinate (u, v) and un-rotate the attention output by the query's patch
centre. The aggregate becomes `sum_k w_k R(uv_k - uv_q) v_k`: each Gaussian's offset from the
patch is carried as phase into the token, so the decoder receives placement, not just membership.
32 head channels (16 log-spaced frequencies per axis, 1 to 7 rad/patch, wavelengths 0.9 to 6
patches), per-head zero-initialised mixing gate so the checkpoint is bit-exact at load
(warm-safe, no recovery stage). Reuses P2's projection.

Arm: `p2r_fg_vrope` = the P2+fg recipe (`proj_rope_2d=true --fg_bg_weight 0.05`, seed V18,
stage R 3000, 30k steps) plus `value_rope_2d=true`. Control: `p2r_fg` (fit 3.35 / held-out
19.39). Readout: the usual fit / heldout300 margins. Pass = fit moves by dB, not tenths.
Cost: one 4-GPU sagieb run, ~10 h.

## 2. Finer ray grid (patch 4)

`patch_size=4` with `ray_embed_patch=8`: each 4x4 ray patch is nearest-upsampled to 8x8 before
the pretrained ray-map Linear, so the checkpoint warm-starts exactly; the DPT head already
scales with `patch_size`. Halves the Gaussians per token per axis.

Stage 1 (cheap, decisive): at 256 px, `256/4` (4096 tokens, the per-token load of today's
512/8) against a `256/8` control (1024 tokens) with the same schedule; evaluation at 256
(`ceiling_eval --res 256`). ~2.5 h each on 4 GPUs. If the load hypothesis is right this is a
multi-dB fit gain. Stage 2 only if stage 1 passes: `512/4` (16k ray tokens, ~4x view-stage
cost, ~30 h) -- and it is the point where the quadratic scene encoder must go sparse
(k-NN attention over Gaussians) to afford anything further.

## 3. Camera-distance augmentation

All training views sit on one orbit radius (1.7). The only wins were at close range (r 1.15),
which the model has never seen. Rebuild the full 50k splats of the ten N=10 objects (deleted
17 Aug; `data_v10/process_full.py` restricted to those ten), render GT at r = 1.15 and 2.45
(the codec close / far radii), write `n10_h5_r3` with 42 views per object. Held-out close-range
readout: heldout300 GT at r = 1.15 from `full_h5s_val` (exists), `ceiling_eval --radius 1.15`.
Arm `p2r_fg_aug` vs `p2r_fg` on (a) the standard readout, (b) the r = 1.15 readout, (c) the
slipper zero-shot close range. ~10 h after data is ready.

## Not doing now

Feature splatting (a rasterizer in the loop); RoPE bandwidth; capacity growth; cycles.

## Schedule

Day 0 (today): cleanup PR, value-RoPE + patch code, launch 1 and both 256 runs of 2, start the
full-splat rebuild for 3. Day 1: verdicts on 1 and 2-stage-1; launch 3 and, if warranted,
2-stage-2. Day 2: verdicts on 3 (and 2-stage-2 on day 3).
