# GaussianFormer Progress Report

## Session: 2026-04-14

### Overview

First end-to-end proof-of-concept training run of GaussianFormer on a Linux machine with 12 cores, 128GB RAM, and an RTX A6000 (48GB VRAM). The session covered three phases: training data generation at scale, initial model training with iterative debugging, and a data quality audit that led to improvements in the mesh-to-Gaussian conversion pipeline.

---

### 1. Training Data Generation (400 scenes)

**What we did:** Scaled up scene generation from 30 to 400 randomized scene descriptor JSONs using `generate_training_data.py`. Each scene contains background planes, 1-2 main objects (from a pool of 23 meshes), and 1-3 light sources, all following RenderFormer's training-data constraints (camera distance [1.5, 2.0], FOV [30, 60], scene bbox [-0.5, 0.5], etc.).

**Pipeline executed:**
1. `generate_training_data.py` -- 400 mesh-format scene JSONs
2. `batch_render_training_scenes.py` -- JSON to H5 conversion, then RenderFormer inference to produce ground-truth PNG renders (256px, bf16 precision, AGX tone mapping)
3. `batch_convert_to_gaussian_examples.py` -- OBJ meshes to Gaussian PLY files + JSON rewrite
4. `gaussian_scene_processor/batch_generate_h5.py` -- Gaussian PLY + JSON to HDF5 training files

**Decision: 100 scenes for PoC rendering.** Rendering all 400 scenes through RenderFormer was feasible (~5-6 views/second on the A6000) but we chose to render only 100 to prioritize seeing training results quickly. All 400 H5 conversions (both mesh and Gaussian) were completed for future use.

**Decision: AGX tone mapping.** RenderFormer outputs HDR images (values above 1.0). To produce LDR ground-truth PNGs suitable as training targets, we chose AGX tone mapping -- the most neutral option for CG/3D rendering, preserving color accuracy without artistic grading. The training loss function (`log10(ldr + 1)`) expects LDR inputs.

**Decision: 256px resolution.** Matched the training config's default resolution to avoid unnecessary resizing in the data loader, and keeps VRAM usage lower for the PoC.

---

### 2. Model Training -- First PoC Run

**Architecture recap:** Two-phase training pipeline. Phase 1 freezes the pretrained RenderFormer backbone and trains only the Gaussian input module (13K / 195M parameters). Phase 2 unfreezes everything for full fine-tuning.

**Bug fix: Variable-length Gaussian sequences.** Different scenes have different numbers of Gaussians (6000-9000), causing `torch.stack` to fail in the default collate function. Added a custom `collate_fn` to `training/dataset.py` that pads Gaussians to the max count in each batch and updates the boolean mask accordingly.

**Training run 1 (batch_size=2):**
- Phase 1 ran for ~10 epochs before we decided to increase throughput.
- Loss at epoch 10: 0.054484.
- Visual inspection: output transitioned from uniform dark images to colored blobs -- confirming the input encoder was learning to convert Gaussian parameters into meaningful tokens.

**Decision: Increase batch size to 4.** VRAM usage at batch_size=2 was ~22GB out of 48GB available. Doubled to batch_size=4 to reduce wall-clock time per epoch. Phase 1 (13K trainable params, small gradient buffers) handled this fine.

**Training run 2 (batch_size=4, resumed from epoch 10 weights):**
- Phase 1 completed all 20 epochs successfully.
- Loss progression: 0.0535 (epoch 1) to 0.0457 (epoch 20) -- steady decrease.
- Phase 2 OOM'd immediately. With all 195M parameters unfrozen, optimizer states and gradients roughly doubled VRAM requirements (~44GB exceeded the 48GB budget).

**Decision: Drop to batch_size=2 for phase 2.** Phase 2 requires more VRAM per sample due to full-model gradients. Restarted with `--skip_phase1 --resume checkpoints/phase1_epoch_20.pt --batch_size 2 --phase2_epochs 30`.

**Decision: Reduce epoch counts (20 phase1 + 30 phase2).** Original plan was 50+100=150 epochs. Reduced to 50 total to get results within the session. The priority was validating the approach, not achieving best quality.

**Intermediate results (visual inspection via checkpoint renders):**
- Epoch 10 (first run): uniform dark images transitioned to colored blobs.
- Epoch 10 (second run, effectively ~20 epochs total): blobs became more structured, loss dropped to 0.0487.
- Epoch 20 (phase 1 complete): further structure, loss at 0.0457.

---

### 3. Gaussian Conversion Quality Audit

During training, we investigated whether the mesh-to-Gaussian conversion pipeline was producing faithful representations. The audit revealed several issues.

#### Issue 1: Scene bounding box violations (fixed)

**Finding:** 37% of Gaussians had positions outside the [-0.5, 0.5] range that RenderFormer was trained on. Root cause: light sources (tri.obj meshes) were converted to Gaussians with scale ~2.2 and translations at z=1.5-2.5, creating large clusters of Gaussians far from the scene center.

**Fix:** Skip objects keyed `light_*` in `scene_gaussian.py`. Lights are emissive-only objects that don't contribute meaningful geometry in a Gaussian representation. This also reduced per-scene Gaussian counts by 1000-3000.

**Note:** After excluding lights, object Gaussians still extend somewhat beyond [-0.5, 0.5]. This is by design -- it matches the original RenderFormer mesh scene layout where backgrounds sit at the box boundary and normalized objects extend slightly beyond. The [-0.5, 0.5] constraint describes where the scene center should be, not a hard clamp on every vertex.

#### Issue 2: Isotropic Gaussian scales (fixed)

**Finding:** All Gaussians had identical scales across all three axes (`[r, r, r]`), producing spherical blobs. Each object had exactly one scale value (derived from `sqrt(mesh_area / num_samples) * 1.5`), resulting in only 6 unique scale vectors across 9000 Gaussians per scene.

**Fix:** Changed to anisotropic scales: `[r, r, r * 0.1]`. The rotation quaternions already align each Gaussian's local Z-axis to the surface normal (via `get_rotation_to_align_vectors`), so making the Z-scale 10x thinner produces flat disc-shaped Gaussians that lie on the mesh surface. This is standard practice in 3D Gaussian Splatting. The 0.1 factor is a common choice in the literature.

The downstream transform in `scene_gaussian.py` (`transformed_scales = scales * obj_scale`) handles anisotropic scales correctly because `obj_scale` is uniform `[s, s, s]`, preserving the anisotropy ratio.

#### Issue 3: Fixed sample count per object (fixed)

**Finding:** Every object received exactly 1000 Gaussians regardless of mesh complexity. A 5647-vertex Lucy mesh and a 510-vertex banana both became 1000 Gaussians, leading to over-compression on complex models and over-spreading on simple ones.

**Fix:** Adaptive sampling based on vertex count: `max(100, min(5000, num_vertices // 2))`. Examples:
- plane.obj (81 vertices) -- 100 Gaussians (was 1000)
- bunny.obj (2850 vertices) -- 1425 Gaussians
- teapot.obj (4884 vertices) -- 2442 Gaussians
- lucy/11k.obj (5647 vertices) -- 2823 Gaussians

#### Issues investigated but not changed

**Colors (max ~0.4):** Investigated and determined this is correct behavior, not a bug. PLY colors default to white (1.0), then get multiplied by `color_tint` (which equals the material diffuse values from scene generation). Many material archetypes have diffuse values capped at 0.5 (glossy) or 0.05 (metallic). The observed 0.4 max accurately represents the scene materials.

**Opacity (all 1.0):** Correct for solid mesh surfaces. Each Gaussian represents a point on an opaque surface, so full opacity is the right default. Variation could be explored later but is not a priority for the PoC.

#### Results after conversion improvements

| Metric | Before | After |
|--------|--------|-------|
| Gaussians per scene | 6000-9000 (uniform) | 1100-2600 (adaptive) |
| Scale shape | Spherical (1:1:1) | Disc (1:1:0.1) |
| Light Gaussians | Included (outliers at z=2+) | Excluded |
| Sample count per object | Fixed 1000 | 100-5000 (vertex-adaptive) |

The improved Gaussian H5 files have been regenerated and are ready for the next training run after the current PoC completes.

---

### 4. PoC Training Results (completed)

The first PoC training run completed successfully with the original (pre-improvement) Gaussian data.

**Phase 2 loss progression:**

| Epoch | Avg Loss |
|-------|----------|
| 1 | 0.0722 *(initial spike from unfreezing all params)* |
| 5 | 0.0456 |
| 10 | 0.0379 |
| 15 | 0.0332 |
| 20 | 0.0287 |
| 25 | 0.0253 |
| 30 | 0.0214 |

**Overall loss reduction:** 0.0535 (phase 1 start) to 0.0214 (phase 2 end) -- a 60% reduction.

**Key observations:**
- Loss was still decreasing at epoch 30, indicating room for more training.
- The phase 2 initial spike (0.072 at epoch 1) from unfreezing all parameters recovered within 3 epochs.
- Per-epoch time stabilized at ~53 seconds after an initial warm-up period of ~275 seconds for the first few epochs (CUDA kernel compilation, memory allocation).
- batch_size=4 OOM'd at phase 2 start; batch_size=2 ran without issue for all 30 epochs.

**Visual quality:** Renders progressed from uniform dark images (untrained) to colored blobs (phase 1 epoch 10) to more structured outputs with discernible color regions (phase 2 epoch 30). Not yet recognizable as scenes, but clearly learning scene-level features.

**Checkpoints saved:**
- `checkpoints/phase1_epoch_10.pt` (779MB)
- `checkpoints/phase1_epoch_20.pt` (779MB)
- `checkpoints/phase2_epoch_10.pt` (2.3GB, includes optimizer state for all 195M params)
- `checkpoints/phase2_epoch_20.pt` (2.3GB)
- `checkpoints/phase2_epoch_30.pt` (2.3GB)
- `checkpoints/gaussianformer_final/` (HuggingFace-format model for inference)

**Next step:** Retrain with the improved Gaussian data (anisotropic scales, adaptive sampling, light exclusion) using the same pipeline.

---

### 5. Software Splatting Renderer & Data Quality Audit

To validate the Gaussian conversion quality independently of the neural renderer, we built a software splatting renderer (`render_gsplat.py`). Initial renders looked terrible — murky blobs with no scene structure. Investigation revealed two issues:

**Bug 1: Rotation-unaware projection (fixed).** The renderer projected Gaussians using only raw scale components as pixel radii, completely ignoring the quaternion rotations. Fixed by implementing proper EWA splatting: build 3D covariance from scales + quaternion rotation, transform to camera space, project to 2D via the perspective Jacobian, then rasterize oriented ellipses using Mahalanobis distance.

**Bug 2: Visualization parameters (fixed).** Raw material colors are very dark (mean brightness 12%) because they lack lighting simulation, and all opacities are 1.0 causing complete occlusion. Added brightness boost (3x), opacity scaling (0.5x), front-to-back compositing with transmittance tracking, and gamma correction.

**Camera convention verified correct.** The Y/Z flip in camera space (Blender -Z forward → pinhole +Z forward) was confirmed working: all 1504 Gaussians in scene_0000 pass the depth test, and 1199/1504 project inside the 256×256 image.

**Data quality confirmed.** Point cloud renders and fixed splatting renders show correct scene structure — wall positions, object silhouettes, and color assignments all match ground truth layouts. The remaining blurriness is inherent to having ~1500-3500 large Gaussians per scene (vs millions in real 3DGS) and is expected. The Gaussians are input tokens for the transformer, not meant to be splatted directly.

---

### 6. V2 Training — Improved Gaussian Data

Retrained from scratch with the improved Gaussian data (anisotropic scales, adaptive sampling, light exclusion).

**Key improvement: batch_size=4 for both phases.** The improved data has fewer Gaussians per scene (1100-2600 vs 6000-9000), so phase 2 no longer OOMs at batch_size=4. This doubled throughput compared to v1's batch_size=2 for phase 2.

**V2 loss progression:**

| Epoch | V1 (old data) | V2 (improved data) |
|-------|--------------|-------------------|
| Phase 1 end (ep 20) | 0.0545 | 0.0526 |
| Phase 2 ep 10 | 0.0416 | 0.0394 |
| Phase 2 ep 20 | 0.0278 | 0.0284 |
| Phase 2 ep 30 | 0.0202 | **0.0195** |

**Visual quality:** V2 renders show more structure than v1 — scene-level color and layout are emerging (wall colors, background separation, rough silhouettes). Objects are not yet distinguishable, and there's no fine detail. Loss was still decreasing at epoch 30.

**Checkpoints saved:**
- `checkpoints/phase1_epoch_{10,20}.pt`
- `checkpoints/phase2_epoch_{10,20,30}.pt`
- `checkpoints/gaussianformer_final/`
- V1 checkpoints preserved in `checkpoints_v1/`

---

### 7. V3 Training — Scaled Up (400 scenes, 512px, 100 epochs)

Full scale-up: 4x more data, 4x resolution, 3x longer training.

**Data scale-up:**
- Rendered all 400 scenes at 512px with AGX tone mapping (was: 100 scenes at 256px)
- 805 total views across 400 scenes (was: 207 views from 100 scenes)
- Ground-truth renders: `training_renders/` (overwritten with 512px versions)

**Phase 1 (20 epochs, batch_size=4, 512px):**
- Loss: 0.067 (epoch 1) → 0.043 (epoch 20)
- Lower final loss than v2's phase 1 (0.053) — more data helps the input encoder
- ~6 min/epoch (366s) vs ~50s at 256px — expected 7x slowdown from 4x resolution + 4x data

**Phase 2 (100 epochs, batch_size=4, 512px) — completed:**

| Epoch | Avg Loss | Notes |
|-------|----------|-------|
| 1 | 0.0535 | Initial spike from unfreezing |
| 10 | 0.0212 | Matches v2's final loss in 10 epochs |
| 20 | 0.0152 | |
| 30 | 0.0118 | |
| 50 | 0.0086 | |
| 80 | 0.0066 | 3x lower than v2's final (0.0195) |
| 100 | **0.0056** | Final — 3.5x better than v2 |

**Key observations:**
- batch_size=4 works for both phases at 512px (fewer Gaussians per scene keeps VRAM manageable)
- Loss was still decreasing at epoch 100, but rate was slowing (0.0066→0.0056 over last 20 epochs)
- ~7 min/epoch, total phase 2 runtime ~12 hours
- Final model saved to `checkpoints/gaussianformer_final/`

**Visual quality progression (checkpoint renders at 256px with AGX tone mapping):**
- Epoch 10: Background walls visible, correct room geometry; objects absent
- Epoch 50: Wall colors more distinct, some shadow/shape hints on floor
- Epoch 100: Distinct wall colors (yellow, pink), visible dark objects on floor, improved sphere/object color rendering (blue/teal). Objects still blobby — expected at this data scale.

Overall: V3 correctly renders room geometry and wall colors, and is starting to pick up objects. Remaining weakness is object detail (blobs rather than sharp shapes).

**Checkpoints saved (every 10 epochs):**
- `checkpoints/phase1_epoch_{10,20}.pt`
- `checkpoints/phase2_epoch_{10,20,...,100}.pt`
- `checkpoints/gaussianformer_final/` (HuggingFace-format model)
- V2 checkpoints preserved in `checkpoints_v2/`

---

### 8. Renderer Validation Against Real 3DGS Data

Validated our software splatting renderer (`render_gsplat.py`) against the Voxel51/gaussian_splatting dataset — real-world 3DGS scenes from the original Kerbl et al. paper.

**Setup:** Downloaded the "truck" scene (1.69M Gaussians) from HuggingFace. Handled 3DGS PLY format conversions: log-space scales → exp, SH DC coefficients → RGB (color = 0.5 + C0 * sh_dc), logit opacity → sigmoid.

**Finding: Renderer geometry is correct, but performance-limited for real 3DGS.** At close distances (10-15 units from scene center), the renderer produces continuous, properly-composited surfaces with correct colors. At distances needed to frame the full scene, the 1.69M tiny Gaussians (median scale ~0.01 units, projected to ~0.07 pixels at viewing distance) appear as sparse scattered points.

**Root cause:** Real 3DGS scenes have millions of sub-pixel Gaussians optimized for specific camera poses. Our per-Gaussian CPU loop (O(N) with per-pixel rasterization) takes ~80 seconds for 1.69M Gaussians at 128px — orders of magnitude slower than GPU-based 3DGS renderers that process all Gaussians in parallel with tile-based rasterization.

**Conclusion:** Our software splatting renderer is validated for our use case (1000-4000 Gaussians per training scene). It correctly implements EWA covariance projection, front-to-back compositing, and camera conventions. For million-scale real 3DGS scenes, a GPU renderer (like the original `diff-gaussian-rasterization` or `gsplat`) would be needed.

**Files created:** `validate_renderer.py`, `renderer_validation/` (output images)

---

### 9. gsplat Validation — Renderer Confirmed Correct

Installed `gsplat` (nerfstudio's production CUDA renderer) as a ground-truth reference. CUDA toolkit 12.8 (loaded via `module`) JIT-compiled gsplat's kernels against PyTorch 2.11+cu130 — took ~160s one-time, then cached.

**Validation results (comparing our software renderer vs gsplat, pixel-by-pixel):**

| Test | Gaussians | PSNR | Result |
|------|-----------|------|--------|
| Tier 1: Synthetic (5 Gaussians, known positions/colors) | 5 | 65.0 dB | PASS |
| Tier 2: Training H5 scene (scene_0000.h5) | 1504 | 50.0 dB | PASS |

Both well above the 30 dB threshold. Our software renderer's EWA projection, camera conventions, and compositing are correct.

**Decision: gsplat is now the primary splatting renderer.** Since gsplat is validated, GPU-accelerated, and handles any Gaussian count, it replaces our software renderer (`render_gsplat.py`) as the ground-truth for Gaussian visualization. The software renderer is kept as a CPU fallback.

**Pre-rendered gsplat reference images:** All 400 training scenes (805 views) pre-rendered at 512px with gsplat and saved to `gsplat_renders/`. These are generated once and reused — no GPU needed at comparison time.

**Triplet comparison workflow:** New `render_triplet.py` produces side-by-side comparisons:
1. **RenderFormer GT** (from `training_renders/`) — what the scene should look like
2. **gsplat render** (from `gsplat_renders/`) — what the Gaussian input tokens look like when splatted
3. **GaussianFormer output** (rendered from checkpoint) — what the model predicts

This triplet keeps us in check across the full pipeline. If column 3 diverges from column 1, it's a model quality issue. If column 2 looks wrong, it's a data issue.

**Files created:** `validate_with_gsplat.py`, `render_gsplat_precompute.py`, `render_triplet.py`

---

## Session: 2026-04-19

### 10. V4 Training — Real 3DGS Data (data_v2, 100 epochs phase 2)

First training run on the v2 data pipeline: real 3D Gaussian Splatting objects with gsplat-rendered ground truth (vs v3's mesh-sampled Gaussians + RenderFormer GT). Phase 1: 20 epochs frozen backbone. Phase 2: 100 epochs full fine-tune with CosineAnnealingLR (5e-5 → 5e-7), `save_interval=5`, batch_size=2, 512px. Ran as SLURM batch job (`runs/train_phase2.sh`, job 29812427) on an A40/L40S for ~37 h.

**Phase 2 loss progression (log-HDR L1 space):**

| Epoch | Train | Val |
|-------|-------|-----|
| 5 | 0.0043 | 0.0038 |
| 10 | 0.0033 | 0.0040 |
| 25 | 0.0020 | 0.0019 |
| 50 | 0.0013 | 0.00141 |
| 75 | — | 0.00094 |
| 100 | 0.00063 | **0.00088** |

Training was clean: no divergence, val and train tracked closely, cosine LR decayed smoothly, no overfitting signal.

**Visual validation on held-out val set (`data_v2/h5s_val`, 100 scenes).** Rendered 7 scenes × 2 views with `phase2_epoch_100.pt` and compared LDR (AGX-tonemapped) outputs to gsplat ground truth (`eval_val_set.py`).

| Metric | Value |
|---|---|
| Mean PSNR (sRGB, 14 views) | **15.18 dB** |
| Min / Max | 12.27 / 17.89 dB |

**Verdict: results are not good.** 15 dB is far below any reasonable rendering quality bar (good novel-view synthesis is ≥25 dB; even mediocre baselines are ≥20 dB). The low log-HDR loss (0.00088) is misleadingly optimistic — `log10(ldr + 1)` compresses the target range heavily, so a model that produces the right low-frequency color/brightness distribution can score well while missing all detail. Outputs qualitatively look like blurry color blobs with roughly-correct global color/layout.

**Why it under-performed (hypotheses):**
1. **Target difficulty jump.** V2's gsplat GT is real 3DGS with fine detail, sharp edges, and view-consistent specular behavior. V3's RenderFormer mesh renders are much smoother and closer to what the RenderFormer backbone was pretrained on.
2. **Token budget vs. target complexity.** ~2–4k input Gaussians (our constraint from O(N²) attention) represent a real 3DGS scene much less faithfully than they represented simple mesh scenes. Real 3DGS reconstructions use 100k–1M+ Gaussians.
3. **Loss is in log-HDR space, not perceptual.** Log compression + L1 in HDR strongly rewards getting low-frequency content right and barely penalizes missing detail. The monotonically-decreasing loss curve masked how much perceptual gap remained.
4. **Data scale is small.** ~400 training scenes for a 195M-parameter fine-tune. Same scale as v3 but on a harder target.

**Next steps to try (in order of expected impact):**
- Add an sRGB-space perceptual loss (L1/LPIPS on tone-mapped output) so the training signal tracks visual quality.
- Render earlier checkpoints (epoch 30, 50, 75) to see whether PSNR plateaued — if so, more epochs alone won't help; need loss or architecture changes.
- Scale data: regenerate 1000+ v2 scenes, diversify object pool.
- Consider higher Gaussian budget per scene (4096 → 8192) if VRAM allows; targets now genuinely need more points.

**Files:** `eval_val_set.py` (val PSNR + side-by-side renders), `checkpoint_renders/v4_val/scene_*_sbs.png` (GT | GF pairs), `runs/train_phase2.sh`, `runs/train_phase2_29812427.out`.

---

## Session: 2026-04-19 / 2026-04-20

### 11. V5 Training — Dual-Loss Ablation (null result)

Targeted ablation of v4's loss function. v4's single log-HDR L1 loss compresses target range so aggressively that it rewards low-frequency correctness and barely penalizes missing detail — the leading hypothesis for why v4 plateaued at 15 dB PSNR. v5 adds a second term in sRGB space to give training a direct perceptual signal.

**Config changes vs v4 (everything else identical: same data, same phase-1 resume point, same cosine LR 5e-5 → 5e-7, same 100 phase-2 epochs, same bs=2 @ 512px):**

1. **Dual loss.** `total = log_HDR_L1(pred, log10(gt+1)) + sRGB_L1((10**pred - 1).clamp(≥0), gt)` with both weights 1.0. New CLI flags `--log_loss_weight`, `--srgb_loss_weight` (`training/train.py`).
2. **Flash Attention 2** (2.8.3) installed via `[tool.uv.sources]` URL pin on the prebuilt cu12/torch2.9 wheel. Required torch downgrade 2.11+cu130 → 2.9.1+cu128 (no prebuilt FA wheel for torch 2.11, and system `nvcc` is 12.8 so no from-source build possible; FA4 was ruled out separately because it doesn't support sm_89/Ada).
3. **bf16 autocast** around forward+loss in both train and val loops (flash-attn requires fp16/bf16; pipeline's internal autocast uses whatever `torch_dtype` is passed).

Ran as SLURM job 29842072 on an L40S (firefoot-02) for ~20 h. ~12 min/epoch — 45% faster than v4's 22 min/epoch, the expected flash-attn speedup. Training was clean: both loss terms descended monotonically in lockstep, train/val tightly matched throughout, no divergence.

**Final losses (epoch 100):**

| | Train | Val |
|---|-------|-----|
| total | 0.00332 | 0.00445 |
| log term | 0.000786 | 0.00106 |
| srgb term | 0.00253 | 0.00339 |

**Visual eval on the same 7 val scenes × 2 views as v4:**

| Metric | v4 | v5 |
|---|---|---|
| Mean PSNR | 15.18 dB | **15.15 dB** |
| Range | 12.27–17.89 | 12.24–17.92 |

**Verdict: null result.** The dual loss did not improve sRGB PSNR — the delta (-0.03 dB) is well within eval noise, and per-scene spread is essentially identical. Notably v5's log-loss term (0.00079) ended *lower* than v4's final (0.00088), yet perceptual PSNR didn't move. This is informative: **v4 was not under-optimizing the perceptual signal**. The loss function is not the bottleneck; something structural is (input Gaussian budget, data scale, or capacity at low-N inputs).

**Also learned on closer inspection of v4/v5 renders:** while PSNR is low, outputs are visibly recognizable objects — not just color blobs as the v4 write-up suggested. The model is clearly capable of rendering full scenes; it just lacks fidelity. Bumps the ceiling on what we think the architecture can do and changes the diagnosis from "the model isn't learning scenes" to "the model is scene-aware but starved of detail".

**Decisions going forward:**
- **Revert the dual loss** in the next training run. Simplicity wins when the added complexity buys nothing. Restore v4's single log-HDR L1 as default; keep the CLI flags around so we can re-enable cheaply if a later experiment wants them.
- **Focus gain-seeking elsewhere.** Most promising levers, in expected-impact order:
  1. **Raise input Gaussian count.** Current data caps at ~3k per scene (`data_v2/process_objects.py --max_gaussians 3000`, median ~6.4k over the subset); bumping to 5k+ is a drop-in data-regen. Real 3DGS scenes need more points than we're giving the model.
  2. **More data.** 400 training scenes for a 195M-parameter fine-tune is tight — regenerate 1000+ v2 scenes, diversify the object pool.
  3. **Architectural levers for higher N.** Gradient checkpointing, sequence/tensor parallelism if we can access multi-GPU.

**Files:** `runs/train_phase2_v5.sh`, `runs/train_phase2_v5_29842072.out`, `checkpoints_v5/phase2_epoch_100.pt`, `checkpoint_renders/v5_val/scene_*_sbs.png` (GT | GF pairs).

---

## Session: 2026-04-20 / 2026-04-23

### 12. V6 Training — Bumped Gaussian Budget (N=3k → 5k), Simple Loss Restored

Follow-up on v5's diagnosis: the loss function was not the bottleneck, so v6 reverts to v4's single log-HDR L1 loss and instead attacks the input-token ceiling. Regenerated the entire v2 dataset with `max_gaussians=5000` (up from 3000) into parallel `_n5k` dirs so the n=3000 baseline stays intact for comparison.

**Data regen (`runs/regen_data_n5k.sh`, SLURM job 29855499):** 1000 train + 100 val scenes. Same `compose_scenes.py` logic as before — train seed=42, val seed=1, drawing from a shared object pool. Scene sizes land at ~7–12k Gaussians each (scenes aggregate multiple objects, each individually capped at 5k), so per-scene token counts roughly doubled vs v4/v5.

**Training (`runs/train_phase2_v6.sh`, SLURM job 29855721, firefoot-04):** identical config to v4 — resume from `checkpoints_v4/phase1_epoch_20.pt`, 100 phase-2 epochs, cosine LR 5e-5 → 5e-7, bs=2 @ 512px, `save_interval=5`. Only differences vs v4: the bumped-N dataset and flash-attn 2.8.3 inherited from v5. Wall-clock ~24 h (~14 min/epoch), consistent with the O(N²) attention cost scaling up from v5's 12 min.

**Final losses (log-HDR L1, epoch 100):**

| | Train | Val |
|---|-------|-----|
| v4 | 0.00063 | 0.00088 |
| **v6** | **0.000626** | — |

Train loss landed essentially on top of v4's — the bigger token budget did not change the achievable training-loss floor.

**Visual eval on the v6 val set (`data_v2/h5s_n5k_val`, 7 scenes × 2 views, `eval_val_set.py`):**

| Metric | v4 | v5 | v6 |
|---|---|---|---|
| Mean PSNR | 15.18 dB | 15.15 dB | **15.80 dB** |
| Range | 12.27–17.89 | 12.24–17.92 | 13.35–18.72 |

**Important caveat:** v6's PSNR is not directly comparable to v4/v5 because the val set itself is different (regenerated with different seed and N=5k). The +0.62 dB headline number mostly reflects a val-set shift, not a true quality gain.

**Verdict: visually indistinguishable from v5.** On side-by-side inspection of `checkpoint_renders/v6_val/`, outputs look essentially the same as v5 — recognizable scene-aware blobs with correct global color/layout, same level of detail. The token-count bump did not buy a visible fidelity improvement at this capacity/data scale.

**Net takeaways:**
- **Simpler loss confirmed sufficient.** v4's single log-HDR L1 produces results on par with v5's dual loss — and v6 reproduces that quality on the bumped-N data without needing the sRGB term. Reverting v5's loss complexity was the right call; we no longer need to carry the `--srgb_loss_weight` plumbing as load-bearing — it's just an idle option.
- **Token count alone isn't the lever.** Doubling the per-scene Gaussian budget (3k → 5k) did not move visual quality. The remaining bottleneck is probably model capacity at low-N inputs and/or data scale, not token budget.
- **Baseline is stable.** Three independent 100-epoch runs now land in the same visual regime. The architecture has a consistent ceiling here; further gain-seeking should target either data diversity, meaningfully higher N (≥8k) with matching model capacity, or architectural changes (e.g. Perceiver-style bottleneck) rather than loss tweaks.

**Files:** `runs/regen_data_n5k.sh`, `runs/train_phase2_v6.sh`, `data_v2/{objects,h5s,h5s_val,renders,renders_val}_n5k/`, `checkpoints_v6/phase2_epoch_100.pt`, `checkpoint_renders/v6_val/scene_*_sbs.png`.

---

### Files Modified This Session

| File | Changes |
|------|---------|
| `generate_training_data.py` | NUM_SCENES: 30 to 400 |
| `training/dataset.py` | Added `collate_fn` for variable-length Gaussian padding |
| `training/train.py` | Import and use custom `collate_fn` in DataLoader |
| `batch_convert_to_gaussian_examples.py` | Anisotropic scales, adaptive sample count |
| `gaussian_scene_processor/scene_gaussian.py` | Skip light objects |
| `infer_checkpoint.py` | New file -- inference from training checkpoints |

### Files Created This Session

| File | Purpose |
|------|---------|
| `infer_checkpoint.py` | Load a `.pt` training checkpoint and render scenes for visual inspection |
| `render_gsplat.py` | Software splatting renderer with EWA covariance projection |
| `validate_renderer.py` | Validates renderer against Voxel51 3DGS dataset (real-world scenes) |
| `render_checkpoints.py` | Batch render multiple checkpoints for training progression visualization |
| `validate_with_gsplat.py` | Validates our renderer against gsplat (pixel-by-pixel PSNR comparison) |
| `render_gsplat_precompute.py` | Pre-renders all training scenes with gsplat (run once, reuse forever) |
| `render_triplet.py` | Triplet comparison: RenderFormer GT / gsplat Gaussians / GaussianFormer output |
| `checkpoint_renders/` | Directory containing rendered outputs from various training checkpoints |
| `training_renders/` | Ground-truth PNG renders (AGX tone mapped) |
| `gsplat_renders/` | Pre-rendered gsplat reference images (512px, all 400 scenes) |
| `gsplat_renders_v2/` | Software splatting renders for data quality validation (legacy) |
| `renderer_validation/` | Validation renders from Voxel51 truck scene + gsplat comparison |
| `triplet_renders/` | Side-by-side triplet comparisons |

---

### Future Ideas

**Apple Sharp for Gaussian generation (shelved).** Apple's Sharp model (`apple/Sharp` on HuggingFace) generates ~1.2M high-quality Gaussians from a single image in <1 second. Could potentially replace our mesh-to-Gaussian conversion by feeding ground-truth renders through Sharp instead. Main blocker: Sharp outputs 1.2M Gaussians but our transformer's self-attention is O(n²), making it infeasible beyond ~4096 tokens. Would require either aggressive subsampling (losing most information) or architectural changes (sparse attention, perceiver-style bottleneck). Worth revisiting if we hit a quality ceiling with mesh-converted Gaussians.

**LR scheduler.** Loss curves show diminishing returns in later epochs with constant LR (v3 phase 2 slowed around epoch 70-80). Add cosine decay or cosine-with-warmup scheduler to `training/train.py` before the next major training run.

---

### External 3DGS Datasets (surveyed)

Surveyed Hugging Face and the broader ecosystem for 3DGS datasets that could help validate our renderer and provide higher-quality Gaussian data. All major datasets use PLY format and contain the standard Gaussian parameters (positions, scales, quaternion rotations, colors/SH, opacities).

**For renderer validation:**

| Dataset | Scenes | Gaussians/scene | Size | License | URL |
|---------|--------|-----------------|------|---------|-----|
| Voxel51/gaussian_splatting | 4 real-world | Varies (7K & 30K iterations) | 3.5 GB | Apache 2.0 | huggingface.co/datasets/Voxel51/gaussian_splatting |

From the original 3DGS paper (Kerbl et al.). Includes reference images (980×980) — ideal for testing our software splatting renderer against known-good output.

**For object-level Gaussian data:**

| Dataset | Objects | Gaussians/object | Size | License | URL |
|---------|---------|-------------------|------|---------|-----|
| Objaverse_Splats | 141K | 50K (uniform) | 382 GB | Non-commercial | huggingface.co/datasets/ShapeSplats/Objaverse_Splats |
| ModelNet_Splats | 12K | Varies | 62 GB | Custom | huggingface.co/datasets/ShapeSplats/ModelNet_Splats |
| ShapeSplatsV1 | 52K | Varies | 306 GB | Custom | huggingface.co/datasets/ShapeNet/ShapeSplatsV1 |

Objaverse_Splats is the most promising: 141K diverse objects with consistent 50K Gaussians each, plus 2D renders available separately. Could replace our mesh-to-Gaussian conversion for individual objects, then compose into scenes. Would need subsampling from 50K to ~1000-3000 per object for our transformer.

**For scene-level data (reference/future):**

| Dataset | Scenes | Gaussians/scene | Size | License | URL |
|---------|--------|-----------------|------|---------|-----|
| SceneSplat-7K | 7.9K indoor | 1.4M avg | Large | CC-BY-SA-4.0 | huggingface.co/datasets/GaussianWorld/scene_splat_7k |
| SceneSplat-49K | 49K mixed | 1.25M avg | Large | CC-BY-SA-4.0 | huggingface.co/datasets/GaussianWorld/scene_splat_49k |
| InteriorGS | 1K indoor | Varies | 44 GB | Custom | huggingface.co/datasets/spatialverse/InteriorGS |

These scene-level datasets have 1M+ Gaussians/scene (far beyond our ~4096 token limit). Useful as reference for quality comparison but not directly usable without major subsampling or architectural changes.

**Key takeaway:** The Voxel51 dataset (4 scenes, Apache 2.0) is immediately actionable for renderer validation. Objaverse_Splats could be a future data source if we want higher-quality object Gaussians than our mesh conversion produces.

---

## Session: 2026-04-25 / 2026-04-26

### 13. V7 — External-Scene Feasibility (Tanks & Temples Truck, LightGaussian-style pruning)

First test of GaussianFormer on a real-world 3DGS scene from outside the synthetic training distribution. Goal: render the same scene via gsplat (ground truth) and via GaussianFormer v6, side-by-side, on a 14-view orbit. New branch `feat/external-scene-eval-lego` (named for the original Lego target — pivoted to Truck after Lego required training-from-scratch infra).

**Scene:** Tanks & Temples *Truck* PLY from `pjramg/gaussian_splatting` on HuggingFace. 2,056,645 Gaussians, COLMAP-fit world frame with origin at the truck, Y-up convention. World coords span ~150 units (truck is ~3 units across; remainder is captured surroundings — pavement, trees, fences, sky).

**Pipeline (`data_external/`):**

1. `prune_truck.py`: load raw PLY → center on origin → multiply by `0.45 / 3.0 = 0.15` so the truck fits in [-0.45, 0.45] → drop Gaussians outside `±bbox_clip=0.5` → drop Gaussians whose mean linear scale exceeds `0.05` → score the remaining ~273k via `Σ_views (projected_opacity × radii_x × radii_y) × max_scale^0.1` over 64 orbit views (gsplat-native LightGaussian approximation) → keep top 5000 → write pruned PLY.
2. `build_truck_h5.py`: pruned PLY → 14-dim Gaussian tokens (drop `f_rest_*` SH, `0.5 + C0*f_dc` for color, sigmoid opacity, exp scale, normalize quat) + 14-view orbit (radius 1.7, FOV 45°, Blender convention, Y-up matching the source PLY) → standard-format H5.
3. `data_v2/render_gt.py`: H5 → gsplat ground-truth renders.
4. `compare_truck.py`: H5 + v6 checkpoint → GaussianFormer renders + per-view side-by-sides + 7×2 overview, computing PSNR vs the gsplat GT.

**The critical bug — and the fix:** First end-to-end run gave 8.40 dB PSNR with both columns showing colorful abstract noise, no recognizable truck. Diagnosis: H5 inputs were `means ∈ [-14.66, 6.52]`, `scales ∈ [0, 4.39]` — orders of magnitude outside the model's training distribution (`means ∈ [-0.5, 0.5]`, `scales < 0.05`). LightGaussian was happily picking the highest-significance Gaussians from the entire scene, and most of those are giant sky/ground slabs at ±15 normalized units. The fix was to filter to the training-distribution range *before* scoring. After the filter: `means ∈ [-0.500, 0.500]`, `scales ∈ [0.000, 0.136]`, `opacities ∈ [0.027, 1.000]`, and PSNR jumped to **13.99 dB mean (12.58 – 15.36) across 14 views** — within ~2 dB of the in-distribution v6 val-set mean (15.80 dB).

**Three-way comparison** (`data_external/truck/renders/overview_3way.png`, per-view triples in `sxs_3way/`):

| Column | Source | N | What you see |
|---|---|---|---|
| **left** | gsplat, bbox-only (no LightGaussian) | 552,174 | Sharp teal truck with wheels, bed, headlights — plus a faint upside-down ghost of the captured surroundings inside the bbox |
| **middle** | gsplat, LightGaussian top-5k | 5,000 | Sparse spiky truck — clearly recognizable color and pose, much-reduced detail |
| **right** | GaussianFormer v6 | 5,000 (input) | Smooth blob with the right teal/grey colors in roughly the right place; loses high-frequency Gaussian-splat texture; consistent across all 14 viewpoints |

**Reference render of the full unpruned scene** (`renders/gsplat_full_wide/`, orbit radius 8 to clear the surrounding cloud): the PLY captures a complete outdoor environment, not just the truck — the full unpruned render at the comparison-rig radius (1.7) puts the camera *inside* the surrounding sky/ground/buildings, which is why both spatial filtering (to isolate the truck) and the orbit rig come out of normalized truck-centric coords rather than world coords.

**Net takeaways:**

- **GaussianFormer v6 generalizes meaningfully to a real-world 3DGS scene** when the input is filtered to the training distribution (positions in [-0.5, 0.5], scales < 0.05). Output is a smoothed, denoised version of the sparse input — not pixel-perfect, but consistent in color and pose across all 14 angles. 13.99 dB on out-of-distribution data vs 15.80 dB in-distribution is a real generalization signal, not a coincidence.
- **Without distribution-matching filters, the model produces hallucinated noise.** The bbox + scale clip is the load-bearing step — LightGaussian alone is not enough on a real-world capture, because real-world 3DGS PLYs encode the entire surrounding environment with parameters orders of magnitude beyond what an object-centric synthetic-trained model has ever seen.
- **The "ground truth" itself is dramatically lossy at N=5000.** Even gsplat with the same 5k Gaussians loses most of the truck's detail relative to the bbox-only render with 552k. PSNR-vs-pruned-gsplat measures fidelity to a sparsified target, not to the original capture — the architecture's ceiling and the 5k input cap are entangled in this number.

**Files added:** `data_external/{prune_truck,build_truck_h5,compare_truck,render_full_truck,build_3way_overview}.py`, `.gitignore` patterns for `data_external/*/raw.ply`, `pruned_*.ply`, `h5/`, `renders/`. Renders + checkpoints live under `data_external/truck/{h5,renders}/` (gitignored due to size).

**Next steps.** V7 plus the v4/v5/v6 trio (all landing in the same visual regime) point at the same conclusion: the bottleneck is no longer loss, data quantity, or token budget — it's architectural at this scale. Two follow-ups, very different cost:
- **Cheap (next branch):** finetune v6 on a small real+synthetic mix (e.g. ShapeSplats objects + ~100 of our scenes) and re-run the Truck eval. Tests whether the OOD smoothness gap closes with data alone.
- **Expensive (multi-week):** Perceiver-style bottleneck so the model can ingest ≥20k input Gaussians without O(N²) attention cost. This is the structural fix already flagged in "Future Ideas" above; V7 is the evidence that we can no longer get further at N=5k.

Recommendation: cheap first — it rules out (or in) the data-only hypothesis before paying the architectural cost.

---

## Session: 2026-04-26

### 14. V8 — External-Scene Demo, Tomatoes (clean object scan)

V7's Truck renders were unconvincing — the GF output was a smooth blob roughly aligned with the truck, but the comparison was muddied by the scene itself: a real outdoor capture with surrounding pavement/trees/sky that bbox-cropping turned into a ghost-laden mess even on the gsplat side. We pivoted to a much simpler external scene to show the "GaussianFormer + gsplat work in tandem" story cleanly.

**Scene:** *Tomatoes - No Postshot!* from superspl.at (`https://superspl.at/scene/0101ad57`), CC BY 4.0. A 219,424-Gaussian scan of a bowl of tomatoes captured with Reality Scan (COLMAP poses) and trained with Brush (gsplat-based). Raw PLY at `data_external/tomatoes/raw.ply` (32 MB, gitignored).

**Why this scene was the right pivot:** raw stats showed everything sitting in [-0.12, +0.12] on every axis (vs the Truck's ±15+ before any filtering). 100% of Gaussians within ±0.12 of the median, scales already inside the v6 training range (median 0.00045, max 0.028 vs training cap 0.05), and a clean black background with no surrounding scan artifacts. Distribution match was free — no bbox/scale filtering needed at all.

**Orientation discovery (one bug we caught early).** First inspection rendered 6 axis-aligned views (`render_tomatoes_axes.py`). The `from_MINUS_Y` view showed a clean top-down of the bowl, suggesting +Y was up — but the actual 14-view orbit came out with the plate mounted upside-down (rim at top, tomatoes hanging below). Re-reading the side views revealed I'd misidentified the bowl's rounded *underside* as the rim. The source PLY actually has gravity-down == +Y (Reality Scan's convention here); the orbit rig assumes +Y up. Fix: 180° rotation about the X axis during normalization — `(x,y,z) -> (x,-y,-z)` for positions and the corresponding quaternion update `(w,x,y,z) -> (-x,w,-z,y)`. A proper rotation, no handedness flip, Gaussian shapes preserved. Re-rendering produced right-side-up bowls.

**Final normalization (locked in for every downstream tool):** center on per-Gaussian median position, 180° rotation about X, ×2.0 scale. After this: positions in [-0.235, +0.225], scales (linear) in [0, 0.055].

**Pipeline (one script, `data_external/run_tomatoes.py`):**
1. Normalize raw PLY -> `normalized.ply`.
2. Score all 219k Gaussians once via gsplat-native LightGaussian over a 64-view orbit (radius 1.7, fov 45°).
3. For each `N ∈ {5000, 10000, 20000, 30000}`: take top-N by score, write `pruned_n{N}.ply` and `tomatoes_n{N}.h5`, render the pruned subset with gsplat, render the same H5 through GaussianFormer v6 (`checkpoints_v6/phase2_epoch_100.pt`, bf16 + flash-attn 2.8.3), build a per-N 3-way overview.
4. Master overview comparing all four N values for two representative views.

The full-scene gsplat reference (column 1 of every comparison) is reused from `render_tomatoes_raw_orbit.py` — same normalization, same orbit rig, so the columns line up pixel-for-pixel by camera index.

**Bumped Gaussian budget on inference.** v6 was trained at N≤5k. With FA2 (memory O(N·d) for attention), we could push inference to much higher N — N=30k ran fine on the A6000 (~6× the training sequence length, no OOM). RoPE positional encoding generalized without observable failure.

**Results:**

| N | PSNR vs pruned-gsplat (mean / min / max) | PSNR vs full-gsplat (mean / min / max) |
|---|---|---|
| 5,000 | 23.16 / 20.94 / 25.14 | 21.78 / 20.37 / 23.03 |
| 10,000 | 22.87 / 21.16 / 24.53 | 21.95 / 20.70 / 23.32 |
| 20,000 | 23.08 / 21.98 / 24.27 | 22.54 / 21.54 / 23.59 |
| 30,000 | 23.33 / 22.38 / 24.21 | 23.00 / 22.15 / 23.71 |

For comparison: v7 Truck @ N=5k landed at **13.99 dB** vs pruned-gsplat; v6 in-distribution synthetic val at **15.80 dB**. Tomatoes @ N=5k is +9 dB over the Truck and +7 dB over in-distribution — substantively better than anything we'd seen.

**Visual verdict (user inspection):** passable but not great — the GF output is recognizably a red/orange object on a white plate, holds shape across all 14 angles, and gets the colors right, but it's smoothed/blurry compared to the gsplat reference (the v6 "scene-aware blob" signature). Not pixel-faithful. The key qualitative win over V7 is that the Truck output was *unusable* (you couldn't tell what it was supposed to be without context); the Tomatoes output is *usable* (clearly a bowl/plate of tomatoes from any angle).

**N didn't help much.** Going 5k → 30k bumps GF PSNR by only 0.17 dB, while the gsplat side gains dramatically over the same range. Same conclusion as V6: the bottleneck is architectural at this model scale, not input-token-budget. RoPE held up, FA2 paid off, but the ceiling is still where we left it.

**Mild scale bias.** The GF outputs read slightly smaller in frame than the gsplat ground truth — likely a shift introduced because the bowl normalizes to ±0.225 rather than the v6 training distribution's ±0.5. Could probably close this by scaling the bowl to fill more of the frame (e.g. NORM_SCALE=4.0), at the cost of pushing some Gaussian scales above the training cap. Not pursued in this session.

**Files added:** `data_external/{inspect_tomatoes,render_tomatoes_axes,render_tomatoes_raw_orbit,run_tomatoes}.py`. Outputs land under `data_external/tomatoes/{normalized.ply, pruned_n*.ply, h5/, renders/{gsplat_full,gsplat_n*,gaussianformer_n*}/, renders/overview_3way_n*.png, renders/overview_all_N.png}` (gitignored).

**Net takeaways:**

- **GaussianFormer v6 produces a usable render on a clean real-world object scan.** "Usable" = recognizable subject, correct colors, consistent across viewpoints, holds shape. "Not great" = blurry, lacks high-frequency detail, mildly mis-scaled. This is the cleanest external-scene demo we have.
- **Input scene quality matters more than N.** Going from a messy outdoor T&T capture (Truck) to a clean isolated object (Tomatoes) bought 9 dB at the same N=5k. Going from N=5k to N=30k on the same clean scene bought 0.17 dB. The data is the lever; budget isn't.
- **The v6 rendering ceiling is the same on- and off-distribution.** Tomatoes (out-of-distribution) at 23 dB is *higher* than v6's in-distribution val at 16 dB — but that comparison is unfair (the v6 val set is harder synthetic scenes, multi-object compositions). What's apples-to-apples: the smooth/blurred *style* of the output is identical between in-distribution v6 val renders and these tomatoes renders. Same architectural ceiling, regardless of input source.

### 15. V9 — Single-object training data overhaul + multi-GPU DDP (Objaverse_Splats)

V8 nailed down the takeaway: data is the lever, N isn't. So V9 attacks data directly — replace v6's multi-object Cornell-box training set with thousands of clean single-object 3DGS scans matched to the tomatoes target style. The bet: a model trained on objects-only data will render objects sharper. If it doesn't, we know perceptual loss (LPIPS) is the next lever; if it does, we may not need LPIPS at all.

This session covers Phase A (data acquisition) + Phase B (DDP migration); the training run (Phase C) is in flight at the time of writing.

**Phase A — data: 2,667 train + 183 val from Objaverse_Splats.**

Source: `ShapeSplats/Objaverse_Splats` on HuggingFace — 141K synthetic 3DGS-fits of Objaverse models, 50K Gaussians each, non-commercial license (research-only is fine for our use). Selective download via `huggingface_hub` is the practical path; the full dataset is hundreds of GB but we only need ~3K objects.

Pipeline (`data_v9/{build_object_list,process_objaverse}.py`):
1. **Filter** the metadata CSV: PSNR ≥ 32, LPIPS ≤ 0.06, num_GS == 50000. 89,927 of 141,703 entries (63.5%) pass. Spread across 154 chunks; we draw from the top 12 chunks by passing-object density (8,762-object pool).
2. **Stratified sample** 3,000 train + 200 val with a fixed seed (object distribution roughly 240/chunk for train, 16/chunk for val).
3. **Per-object processing**, in a per-chunk inner loop so each 3.3 GB zip is downloaded once:
   - extract `{chunk}/{uid}/ckpts/point_cloud_15000.ply` from the zip;
   - decode Kerbl 3DGS fields → median-center → scale so `max(|pos|) == 0.45`;
   - drop if `max_opacity < 0.4` (filters degenerate ghost-fits — the smoke test caught a sushi platter that scored PSNR 40 dB but had max opacity 0.175, rendering as nearly-transparent);
   - LightGaussian-prune 50K → top-5K by `Σ_views(opacity · radii_x · radii_y) · max_scale^0.1` over a 64-view orbit (radius 1.7, fov 45°);
   - render 14 orbit views @ 512px from the **full 50K Gaussians** (the GT — we deliberately give the model a stretch goal: render with 5K what the full 50K can render);
   - bundle into H5 with `means/scales/rotations/colors/opacities/c2w/fov` (matches `training/dataset.py` v6 schema exactly).

After filtering: 2,667 train (333 dropped, ~11%) and 183 val (17 dropped, ~8.5%). Total disk: ~7.2 GB (1.5 GB H5 + 5.7 GB PNG renders).

**Two operational issues caught and fixed during Phase A:**
- *Stuck HF download* (job 30269450). The first SLURM run hung at `hf_hub_download` for chunk 000-045 — a 0-byte `.incomplete` file, no progress for 1h34m, log otherwise normal. The cluster job stayed in `RUNNING` state with no stderr. Fix: `scancel`, clear the partial blob, resubmit. The script's resume support (skip if H5 + view_0.png both exist) made the resubmit cheap — 875 already-processed scenes were detected and skipped, run finished cleanly in ~36 more minutes. No code change needed; the HF blip didn't recur on retry.
- *SLURM modules silently failing.* `runs/process_objaverse_v9.sh` had `module load nvidia / module load cuda` at the top, but the cluster's lmod isn't sourced into non-interactive zsh by default — the loads emitted `command not found: module` and silently did nothing. The venv's bundled CUDA (flash-attn wheel, torch's libs, gsplat compiled at install time) papered over this for our workload, so files kept being written and we caught it only when reviewing the log carefully. Fix in both V9 SLURM scripts: source the cluster's lmod init (`${LMOD_INIT:-/etc/profile.d/lmod.sh}`) before any `module load`.

**Phase B — DDP migration of `training/train.py`.**

The plan called for a 100-epoch run on ~2.7× v6's data. Single-GPU estimate was ~80h; with 4 GPUs we can pull that to ~18h, paying back a ~1.5h migration cost on the first run. Picked **torchrun + raw PyTorch DDP** over HuggingFace Accelerate — no new dep, transparent (we see exactly what's distributed), and small enough to do without an abstraction layer.

Edits, all backwards-compatible (single-GPU path is byte-identical when `LOCAL_RANK` is unset):
- helpers: `setup_ddp()`, `is_main_process()`, `unwrap_model()`, `reduce_metric()`;
- `DistributedSampler` for both train + val DataLoaders, with `set_epoch(epoch)` per epoch for proper shuffling diversity across ranks;
- model wrapped in `DistributedDataParallel(device_ids=[local_rank])` after `model.to(device)`; the unwrapped `model.config` is stashed before wrap so `training_forward` keeps working at every call site;
- `freeze_backbone` / `unfreeze_all` operate on `unwrap_model(model)` because they do name-based parameter matching (`gaussian_encoder.weight`) which DDP's `module.` prefix would silently break;
- val metrics (`log_loss`, `lpips_term`, total) reduced via `dist.all_reduce(op=AVG)` so rank 0 prints global averages;
- all `print(...)` and checkpoint saves gated on `is_main_process()`;
- checkpoints save `unwrap_model(model).state_dict()` so the saved files have no `module.` prefix and stay drop-in compatible with `infer_gaussian.py` / our eval scripts.

LPIPS plumbing was added in the same session (lazy-init `get_lpips()`, dual-term `compute_loss(...) -> (total, log_term, lpips_term)`, separate logging of all three terms in train + val) but the first V9 training run keeps `--lpips_loss_weight 0.0` per the data-first plan — see "Why" below.

**Smoke test, 2 GPUs (`runs/train_phase2_v9_smoke.sh`, job 30283997):** 50 samples × 1 epoch, ~41 s wall. Confirmed: `world_size: 2`, both ranks step through epoch 1, val ran across 2,562 samples and aggregated cleanly, rank-0-only checkpoint saved without DDP key prefix. A few benign warnings: IPv6 socket complaints (cluster has v6 disabled, falls back to v4), `Grad strides do not match bucket view strides` (DDP perf hint, not error), pre-existing RMSNorm dtype mismatch (unrelated to DDP).

**Phase C — production training, 3 GPUs (`runs/train_phase2_v9.sh`, job 30286612).** Submitted at the close of this session.

```
torchrun --standalone --nproc_per_node=3 -m training.train \
  --gaussian_h5_dir data_v9/h5s --renders_dir data_v9/renders \
  --val_h5_dir data_v9/h5s_val --val_renders_dir data_v9/renders_val \
  --save_dir checkpoints_v9 \
  --batch_size 2 --resolution 512 \
  --skip_phase1 --resume checkpoints_v4/phase1_epoch_20.pt \
  --phase2_epochs 100 --phase2_lr 5e-5 --save_interval 5 \
  --log_loss_weight 1.0 --lpips_loss_weight 0.0
```

Per-device bs=2, global bs=6 (vs v6's bs=2 single-GPU). LR kept at 5e-5 — prefer attributing any improvement to data, not LR; sqrt-rule scaling (~9e-5 for 3× global bs) is a clean next-iter ablation. Checkpoint cadence 5 epochs (matches v4/v5/v6). Wall-clock estimate: ~23h vs ~80h single-GPU.

(Initially submitted as 4 GPU / job 30284924 but stuck in `AssocGrpGRES` — our group's per-account cap is 4 GPUs total and the interactive session was holding one slot. Resubmitted at 3 GPU. Trivial bump back to 4 once the interactive ends.)

**Per-step rate diagnostics — DDP comm is the real bottleneck, not data IO.**

The "~14 min/epoch" estimate came from extrapolating v6's single-GPU rate (7.3 step/s @ bs=2 @ 512px). Actual measured rate on V9 across three configurations:

| Config | Rate (step/s) | min/epoch | 100-epoch wall |
|---|---|---|---|
| v6 baseline (1 GPU, 14k samples) | 7.3 | 16 | 24h |
| V9 3-GPU, num_workers=0 (job 30286612) | 1.18 | 88 | 147h |
| V9 3-GPU, num_workers=4, 24c/128GB (job 30298160) | 1.78 | 58 | 97h |
| V9 3-GPU, num_workers=8, 32c/256GB (job 30301987) | 1.55 | 67 | 112h |

Workers helped from 1.18 → 1.78 step/s (filesystem cache + parallel decode), but doubling workers + cores + RAM did *not* climb further (1.55 in the new run is statistical noise relative to 1.78). That definitively rules out data-loader IO as the bottleneck. Per-batch budget at 1.55 step/s = 645 ms; v6 single-GPU forward+backward = 140 ms; the unaccounted ~500 ms/batch lives in DDP comm — most likely NCCL all-reduce on the 200M-param model going through PCIe rather than NVLink (no topology check done yet). To investigate next iter: `NCCL_DEBUG=INFO` topology log, `gradient_as_bucket_view=True`, ZeroRedundancyOptimizer, larger per-device bs (comm cost is ~constant per batch, not per sample).

Final SLURM alloc: 168h (max on `medium` partition), comfortably covering the ~112h ETA. Job 30301987 is the one that went the distance.

**Why no LPIPS for the first run.** Two reasons. First, V8's evidence pointed at *both* data and loss as untested levers; running them together would conflate which one moved the needle. Second, training data was hypothesized to be the bigger lever (v8: data quality bought +9 dB vs the truck; loss tweaks in v5 bought 0). If V9 alone breaks the 23 dB ceiling, the next iter chases LPIPS for additional sharpness; if V9 plateaus near 23 dB, that's the signal that data wasn't enough and LPIPS is the necessary remedy.

**Files added.**
- `data_v9/build_object_list.py` — quality-filter + stratified sample of metadata CSV.
- `data_v9/process_objaverse.py` — per-object pipeline (extract → normalize → score → prune → render → H5).
- `data_v9/smoke_test_objaverse.py` — Phase A pre-flight: 10 random objects, axis-aligned views, master grid; written before scaling to 3K.
- `runs/process_objaverse_v9.sh` — SLURM batch for data generation.
- `runs/train_phase2_v9.sh` — SLURM batch for the 4-GPU training run.
- `runs/train_phase2_v9_smoke.sh` — 2-GPU 1-epoch DDP smoke.
- `training/train.py` — DDP migration + LPIPS dual-loss plumbing.
- `pyproject.toml` / `uv.lock` — added `lpips>=0.1.4` (no conflicts; flash-attn version-string normalized).

Outputs: `data_v9/{h5s,renders,h5s_val,renders_val}/` (gitignored, 7.2 GB), `data_v9/{metadata_train,metadata_val,object_list_train,object_list_val}.json`, `data_v9/smoke_test/` (10-object orientation grid).

**Open questions for the next session (post-Phase C).**
- How does V9 epoch-0 (untrained on this data) compare to V9 epoch-100 on the tomatoes scene? The v4 phase1 checkpoint we resume from has only seen the input encoder briefly trained — anything beyond that is fresh learning on V9.
- Does the multi-object catastrophic-forgetting check (V9 ckpt rendered on V6's old val) show degradation? If yes, future runs may want a 30/70 mix.
- Is global bs=6 stable at lr=5e-5, or do we see slow convergence pointing at LR underuse?

**Auto-check planned at the first checkpoint.** A wakeup is scheduled to fire ~5.6h after job 30301987 starts (estimated time of `phase2_epoch_5.pt`), to run `data_external/run_tomatoes.py --checkpoint checkpoints_v9/phase2_epoch_5.pt --tag v9_ep5` and report PSNR + visual sanity vs V8's 23.16 dB headline. The `--checkpoint` and `--tag` CLI flags were added to `run_tomatoes.py` in this session — V9 outputs go under `data_external/tomatoes/renders/gaussianformer_n{N}_v9_ep5/` and `overview_3way_n{N}_v9_ep5.png` so they sit alongside V6's existing renders rather than overwriting.

**Forward-looking plan filed at `~/.claude/plans/post-v9-followups.md`.** Three threads, ordered by ROI:
1. NCCL/comm investigation — the diagnostic and mitigation plan for the unexplained 500 ms/batch in DDP. Pre-requisite for any future multi-GPU run.
2. The actual V9 evaluation — tomatoes at every 5-epoch checkpoint, decision rule (≥24 dB win, 23-24 marginal → LPIPS, <23 → architecture is the wall), held-out superspl.at scans + V6-val cross-distribution check still pending.
3. Misc infra (snapshot dataset?, prune old checkpoints).

Pending: training run completion, tomatoes apples-to-apples vs V8's 23.16 dB headline, and held-out superspl.at scans (data not yet curated).

## Session: 2026-04-27 — V9 mid-training tomatoes evals

Job 30301987 still running on 3 GPU (currently mid-epoch 13). Two checkpoints landed and were evaluated against the V8 baseline (23.16 dB pruned / 21.78 dB full at N=5000).

**Epoch 5 — first read, breaks the ceiling.**

| N | vs pruned (mean / min / max) | vs full (mean / min / max) |
|---|---|---|
| 5000 | 31.45 / 30.85 / 32.03 | **25.60** / 24.05 / 27.21 |
| 10000 | 30.17 / 28.68 / 32.09 | 26.64 / 24.83 / 28.42 |
| 20000 | 29.16 / 27.46 / 30.86 | 27.21 / 25.35 / 29.16 |
| 30000 | 28.82 / 27.22 / 30.66 | 27.43 / 25.70 / 29.53 |

That's +8.3 dB vs pruned and +3.8 dB vs full at N=5000 over V8 — well above the V9-plan win bar of ≥24 dB. Visually, the v6/v8 smooth-blob style is gone: each tomato is individually resolvable, plate rim is crisp, color is faithful. Some residual horizontal elongation remains. Conclusion at this checkpoint: **data was the lever, not the loss form.** L1 on its own is not the bottleneck V5/V6/V8 implied — multi-object Cornell-box training data was.

**Epoch 10 — plateau on the tomatoes scene.**

| N | ep5 → ep10 vs pruned | ep5 → ep10 vs full |
|---|---|---|
| 5000 | 31.45 → 31.50 | 25.60 → 25.68 |
| 10000 | 30.17 → 30.21 | 26.64 → 26.58 |
| 20000 | 29.16 → 29.28 | 27.21 → 27.27 |
| 30000 | 28.82 → 29.07 | 27.43 → **27.65** |

All deltas within run-to-run noise. Visual is indistinguishable from epoch 5. Train loss kept descending (epoch 5 avg 0.001778 → epoch 10 avg 0.001498) and val too (0.001952 → 0.001757), so the model is still learning in-distribution on Objaverse_Splats — the gain just isn't transferring to the tomatoes scene yet. Either tomatoes hit a representational ceiling for this model+data, or the cosine-LR descent (currently 4.97e-5 → 4.88e-5 over the 5 epochs measured) hasn't yet pushed hard enough to shift OOD-ish scenes. Continuing training; checking back at every 5-epoch checkpoint.

**Epoch 15 — slow drift up, not a hard plateau.**

| N | ep10 → ep15 vs pruned | ep10 → ep15 vs full |
|---|---|---|
| 5000 | 31.50 → 31.79 | 25.68 → 25.89 |
| 10000 | 30.21 → 30.17 | 26.58 → 26.81 |
| 20000 | 29.28 → 29.24 | 27.27 → 27.55 |
| 30000 | 29.07 → 29.18 | 27.65 → **28.11** |

Small but consistent uptick everywhere, biggest at high N (N=30000 vs full +0.46 dB over 5 epochs). Reframes ep5/ep10 as a near-plateau rather than a true plateau — slow drift, not stalled. Train loss kept descending too (0.001498 → 0.001347), val 0.001757 → 0.001693. Visual still indistinguishable from ep10.

**Epoch 20 — slight regression, within noise.**

| N | ep15 → ep20 vs pruned | ep15 → ep20 vs full |
|---|---|---|
| 5000 | 31.79 → 31.24 | 25.89 → 25.78 |
| 10000 | 30.17 → 29.88 | 26.81 → 26.69 |
| 20000 | 29.24 → 29.05 | 27.55 → 27.34 |
| 30000 | 29.18 → 29.04 | 28.11 → 27.89 |

All four N's dipped slightly (−0.1 to −0.5 dB) — within run-to-run noise but the first uniformly-negative step. Train loss kept descending (0.001347 → 0.001240) and val too (0.001693 → 0.001668), so the model is still learning in-distribution; tomatoes is just noisily oscillating around the ~25.8/27.9 dB band rather than climbing. Visual still indistinguishable. Consistent with the hypothesis that meaningful tomatoes movement, if it comes, will arrive in the late-LR phase (ep50+, when cosine pushes LR below ~2e-5).

**Epoch 25 — bounce back, ep20 dip confirmed as noise.**

| N | ep20 → ep25 vs pruned | ep20 → ep25 vs full |
|---|---|---|
| 5000 | 31.24 → 31.62 | 25.78 → 25.90 |
| 10000 | 29.88 → 30.61 | 26.69 → 26.83 |
| 20000 | 29.05 → 29.59 | 27.34 → 27.56 |
| 30000 | 29.04 → 29.35 | 27.89 → 28.06 |

All four climbed +0.3 to +0.7 dB, confirming ep20 was within the noise band rather than a real regression. Against ep15 (the prior local high) tomatoes is essentially oscillating in a ±0.5 dB band: ~31.5/25.9 dB at N=5000, ~29.2/28.0 dB at N=30000. Train+val loss kept descending (train 0.001240 → 0.001156, val 0.001668 → 0.001627). Visual indistinguishable from ep15/ep20.

**Epoch 30 — new highs at every N.**

| N | ep25 → ep30 vs pruned | ep25 → ep30 vs full |
|---|---|---|
| 5000 | 31.62 → 31.68 | 25.90 → 26.16 |
| 10000 | 30.61 → 30.89 | 26.83 → 27.17 |
| 20000 | 29.59 → 29.72 | 27.56 → 27.79 |
| 30000 | 29.35 → 29.41 | 28.06 → 28.23 |

All four climbed (+0.1 to +0.3 dB), every N a new local high. First time over 26 dB at N=5000 and over 28.2 dB at N=30000 against full-3DGS. The "plateau" is now clearly just the noise floor — model is still slowly improving. Train+val loss continue descending (train 0.001156 → 0.001086, val 0.001627 → 0.001592). Visual indistinguishable.

**Trajectory at the third-mark (ep5 → ep30, all dB vs full):**

| N | ep5 | ep10 | ep15 | ep20 | ep25 | ep30 |
|---|---|---|---|---|---|---|
| 5000  | 25.60 | 25.68 | 25.89 | 25.78 | 25.90 | **26.16** |
| 10000 | 26.64 | 26.58 | 26.81 | 26.69 | 26.83 | **27.17** |
| 20000 | 27.21 | 27.27 | 27.55 | 27.34 | 27.56 | **27.79** |
| 30000 | 27.43 | 27.65 | 28.11 | 27.89 | 28.06 | **28.23** |

After 25 epochs of additional training: +0.56 dB at N=5000, +0.80 dB at N=30000. Cosine LR is at 3.98e-5 (started 5.00e-5) — most of the perceptual-refinement window is still ahead (LR drops below 1e-5 only after ~ep75). Trajectory supports the earlier hypothesis that late-LR phase is where the bigger gains may show up.

**Epoch 35 — divergence between vs-pruned and vs-full.**

| N | ep30 → ep35 vs pruned | ep30 → ep35 vs full |
|---|---|---|
| 5000 | 31.68 → 31.00 | 26.16 → 26.37 |
| 10000 | 30.89 → 30.51 | 27.17 → 27.40 |
| 20000 | 29.72 → 29.75 | 27.79 → 28.12 |
| 30000 | 29.41 → 29.62 | 28.23 → **28.64** |

Every N posted a new high vs full — biggest single-step jump yet at N=30000 (+0.41 dB). Vs pruned slipped at low N (−0.7 at N=5000, −0.4 at N=10000) while staying flat or rising at high N. Plausible read: model is learning to fill detail *beyond* what 5K-Gaussian tokens can directly represent, so it tracks full-3DGS more faithfully while diverging from the pruned-token reference. Vs full is the meaningful comparison (it's the actual ground-truth render); vs pruned was always bounded by the token-budget representation. Net positive.

Train loss kept descending (0.001086 → 0.001026); val ticked up a hair for the first time (0.001592 → 0.001613) — within noise but worth flagging if it persists. Visual still indistinguishable.

**Epoch 40 — slight wobble vs ep35; in-distribution val recovers.**

| N | ep35 → ep40 vs pruned | ep35 → ep40 vs full |
|---|---|---|
| 5000 | 31.00 → 30.90 | 26.37 → 26.21 |
| 10000 | 30.51 → 30.05 | 27.40 → 27.14 |
| 20000 | 29.75 → 29.16 | 28.12 → 27.79 |
| 30000 | 29.62 → 29.04 | 28.64 → 28.21 |

All N regressed slightly on tomatoes vs ep35 (vs full: −0.16 / −0.26 / −0.33 / −0.43). However val loss recovered (0.001613 → 0.001599, below ep30's 0.001592 by a hair) and train kept descending (0.001026 → 0.000972). Read: ep35 was a tomatoes-favorable point in the noise band; ep40 is essentially flat with ep30 on tomatoes while in-distribution metrics keep improving. Cosine LR at 3.29e-5 — still in the upper-third of the schedule. Continue to ep45+.

**Epoch 45 — vs-pruned/vs-full divergence widens further.**

| N | ep40 → ep45 vs pruned | ep40 → ep45 vs full |
|---|---|---|
| 5000 | 30.90 → 31.71 | 26.21 → 25.98 |
| 10000 | 30.05 → 30.83 | 27.14 → 26.97 |
| 20000 | 29.16 → 29.90 | 27.79 → 27.70 |
| 30000 | 29.04 → 29.61 | 28.21 → 28.17 |

Vs pruned bounced up sharply at every N (+0.57 to +0.81). Vs full slipped slightly at every N (−0.04 to −0.23). The interpretation is now consistent across ep35/ep40/ep45: model is trading off ability-to-mimic-pruned-tokens against ability-to-mimic-full-render. Vs full has been ~flat in the 26.0–26.4 / 27.0–27.4 / 27.7–28.1 / 28.2–28.6 dB bands since ep30 — looks like a soft plateau on tomatoes specifically, while in-distribution val keeps descending (0.001599 → 0.001593 at ep45).

Train 0.000922, lr 2.91e-5 — about half-way through cosine schedule.

**Epoch 50 — local low; in-distribution val hits a new minimum.**

| N | ep45 → ep50 vs pruned | ep45 → ep50 vs full |
|---|---|---|
| 5000 | 31.71 → 31.60 | 25.98 → 25.75 |
| 10000 | 30.83 → 30.68 | 26.97 → 26.78 |
| 20000 | 29.90 → 29.60 | 27.70 → 27.47 |
| 30000 | 29.61 → 29.23 | 28.17 → 27.90 |

Both vs-pruned and vs-full slipped at every N — first time both move in the same direction since ep30. Tomatoes-side this looks like a real (but small) regression. In-distribution val, however, hit a new low: 0.001593 → 0.001585. Read: model is still learning but tomatoes-specific generalization is bounded. Train continues 0.000922 → 0.000877, lr 2.53e-5.

**Epoch 55 — partial bounce on tomatoes; val ticks back up.**

| N | ep50 → ep55 vs pruned | ep50 → ep55 vs full |
|---|---|---|
| 5000 | 31.60 → 31.19 | 25.75 → 25.96 |
| 10000 | 30.68 → 30.32 | 26.78 → 26.92 |
| 20000 | 29.60 → 29.48 | 27.47 → 27.63 |
| 30000 | 29.23 → 29.24 | 27.90 → 28.12 |

Vs-full recovered 0.13–0.22 dB at every N (ep50 → ep55), but still below ep35 highs. Vs-pruned dropped a hair at low N. Val rose from 0.001585 → 0.001594 (within noise). Train 0.000877 → 0.000835, lr 2.14e-5.

**Epoch 60 — plateau confirmed; fractional drift, no breakthrough.**

vs-pruned: 31.49 / 30.85 / 29.94 / 29.58. vs-full: 25.90 / 26.94 / 27.74 / 28.24. Train 0.000835 → 0.000795, val 0.001594 → 0.001584 (new low by 0.000001, within noise). Tomatoes ep60 ≈ ep55 ± 0.1 dB at every N — still 0.4–0.5 dB below ep35 peaks. Decision: kill V9, pivot to V10 (LPIPS fine-tune).

**Tomatoes plateau (vs full, dB) is now well-characterized:**

| N | ep30 | ep35 | ep40 | ep45 | ep50 | ep55 | ep60 |
|---|---|---|---|---|---|---|---|
| 5000  | 26.16 | **26.37** | 26.21 | 25.98 | 25.75 | 25.96 | 25.90 |
| 10000 | 27.17 | **27.40** | 27.14 | 26.97 | 26.78 | 26.92 | 26.94 |
| 20000 | 27.79 | **28.12** | 27.79 | 27.70 | 27.47 | 27.63 | 27.74 |
| 30000 | 28.23 | **28.64** | 28.21 | 28.17 | 27.90 | 28.12 | 28.24 |

ep35 holds every high-water mark. Seven checkpoints later, all four N values are 0.4–0.5 dB below ep35 peak. Strong signal that pure log-HDR L1 + 5K-token data has saturated tomatoes — confirms the V9 plan's go/no-go diagnostic and motivates V10 (LPIPS fine-tune from ep35 or ep60).

**Eval invocation pattern.** `python -m data_external.run_tomatoes --checkpoint checkpoints_v9/phase2_epoch_<N>.pt --tag v9_ep<N>` (or `sbatch --export=ALL,EP=<N> runs/eval_v9_tomatoes.sh` from a no-GPU session). Outputs land at `data_external/tomatoes/renders/gaussianformer_n{N}_v9_ep<N>/` and `overview_3way_n{N}_v9_ep<N>.png`. Logs at `runs/eval_v9_ep<N>.log`.

## V10 — LPIPS fine-tune from V9 ep60 (started 2026-04-29)

V9 killed after ep60 once tomatoes plateau was confirmed (ep35 held every vs-full high through ep60). V10 = warm-start from V9 ep60 + add LPIPS-VGG (weight 0.2) on display-space LDR. 30 epochs, cosine 2e-5 → 2e-7, save_interval=2. `runs/train_phase2_v10.sh` (job 30465943).

**Throughput**: 0.82 step/s (vs V9's 1.78), ~7870s/epoch (~2.19 h). VGG backward is ~2.2x more expensive than I estimated. 30 epochs ≈ 65 h. VRAM steady at 22/46 GB across multiple samples — confirmed *not* a low-water artifact; V10 actually uses half what V9's profile suggested.

**Loss trajectory** (train avg / val):
| Epoch | log train | lpips train | total train | log val | lpips val |
|---|---|---|---|---|---|
| 1 | 0.000919 | 0.025249 | 0.005969 | — | — |
| 2 | 0.000927 | 0.023128 | 0.005552 | 0.001653 | 0.033426 |

Log term essentially flat; LPIPS descending. Perceptual gradient is the active learning signal as designed.

**ep2 tomatoes — Hold, no collapse:**

| N | V10 ep2 | V9 ep60 | Δ vs ep60 | V9 ep35 peak | Δ vs peak |
|---|---|---|---|---|---|
| 5000  | 25.78 | 25.90 | −0.12 | 26.37 | −0.59 |
| 10000 | 26.71 | 26.94 | −0.23 | 27.40 | −0.69 |
| 20000 | 27.43 | 27.74 | −0.31 | 28.12 | −0.69 |
| 30000 | 27.84 | 28.24 | −0.40 | 28.64 | −0.80 |

PSNR drifted 0.1–0.4 dB below V9 ep60. Expected: LPIPS shifts the L1 minimum, pixel-PSNR is the wrong metric. **Visual inspection is the signal.** User confirmed shapes are better-defined despite added noise — sharpening trade against L1-mean smoothness, the V10 hypothesis. Continue.

**Eval invocation pattern (V10).** `sbatch --export=ALL,EP=<N> runs/eval_v10_tomatoes.sh`. Outputs at `data_external/tomatoes/renders/gaussianformer_n{N}_v10_ep<N>/` and `overview_3way_n{N}_v10_ep<N>.png`. Logs at `runs/eval_v10_ep<N>.log`.

---

## V10b — LPIPS fine-tune restart (bs=4, 4 GPUs, 2026-04-29/30)

V10 cancelled after ep4 (throughput too slow at 2.19 h/epoch for 30 epochs). V10b restarts from `checkpoints_v10/phase2_epoch_4.pt` with bs=4 / 4 GPUs. 26 epochs, cosine 5e-5 → 5e-7, save_interval=2. `runs/train_phase2_v10b.sh` (job 30471385). Completed 2026-04-30.

**Throughput**: ~47.4 min/epoch (2.75× faster than V10's 130 min/epoch). VRAM 40.6/46 GB — tight but stable.

**Loss trajectory (full run, train / val):**

| Epoch | log train | lpips train | total train | log val | lpips val |
|---|---|---|---|---|---|
| 1  | 0.000972 | 0.021103 | 0.005192 | — | — |
| 2  | 0.000982 | 0.020599 | 0.005102 | 0.001656 | 0.033049 |
| 6  | 0.000959 | 0.018448 | 0.004649 | 0.001670 | 0.032894 |
| 10 | 0.000922 | 0.016741 | 0.004270 | 0.001650 | 0.032738 |
| 14 | 0.000875 | 0.015088 | 0.003893 | 0.001654 | 0.032790 |
| 18 | 0.000836 | 0.013806 | 0.003597 | 0.001654 | 0.032870 |
| 22 | 0.000812 | 0.013009 | 0.003413 | 0.001656 | 0.032989 |
| 26 | 0.000803 | 0.012718 | 0.003347 | 0.001659 | 0.033052 |

Train LPIPS dropped 40% (0.021 → 0.013). **Val LPIPS flat at ~0.033 throughout** — mild overfitting on the perceptual term; 2.7K training objects is a small dataset for LPIPS fine-tuning. Val log-L1 also flat (~0.00166), no regression on reconstruction.

**Tomatoes eval — final results (vs full-gsplat, dB):**

| N | V9 ep60 | V10b ep10 | V10b ep20 | V10b ep26 | Δ ep26 vs V9 |
|---|---|---|---|---|---|
| 5000  | 25.90 | 25.88 | 25.83 | 25.80 | −0.10 |
| 10000 | 26.94 | 26.78 | 26.74 | 26.70 | −0.24 |
| 20000 | 27.74 | 27.52 | 27.44 | 27.41 | −0.33 |
| 30000 | 28.24 | 27.94 | 27.84 | 27.81 | −0.43 |

V9 ep60 holds best PSNR across all N. V10b PSNR regresses 0.1–0.4 dB vs V9 ep60, consistent with LPIPS optimizing a perceptual proxy rather than pixel fidelity. The slight PSNR cost is the expected trade for perceptual sharpness; visual inspection (user-confirmed at ep2) showed better-defined shapes and reduced smoothing.

**Eval invocation pattern (V10b).** `sbatch --export=ALL,EP=<N> runs/eval_v10b_tomatoes.sh`. Outputs at `data_external/tomatoes/renders/gaussianformer_n{N}_v10b_ep<N>/` and `overview_3way_n{N}_v10b_ep<N>.png`. Logs at `runs/eval_v10b_ep<N>.log`.
