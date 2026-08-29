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

## V11 — Per-field NeRF input encoding (started 2026-05-19)

**Goal.** Replace V10b's monolithic `Linear(14, 768)` Gaussian encoder with a per-field decomposition: NeRFEncoding on position (12 freqs) and `log(scale)` (6 freqs), plain linear for rotation/color/opacity, summed additively into the `gaussian_token`. Motivation: address V4–V10b's chronic blurriness and PSNR plateau by giving the transformer explicit high-frequency spatial features instead of relying solely on RoPE on the attention side.

**Architecture diff (vs V10b).**
- New `pe_type='nerf_perfield'` in `GaussianFormerConfig`.
- 5 per-field `Linear → RMSNorm(768)` projections (`pos_proj`, `scale_proj`, `rotation_proj`, `color_proj`, `opacity_proj`) replacing the monolithic encoder.
- `pos_pe = NeRFEncoding(3, num_freqs=12, include_input=True)` → 75-D → 768.
- `scale_pe = NeRFEncoding(3, num_freqs=6, include_input=True)`, applied to `log(scale + 1e-6)` → 39-D → 768.
- View-transformer reuses the existing `nerf` PE path (NeRF on ray camera origin → 768; added to ray tokens).
- 19 new "Gaussian-specific" params total: 10 weights+biases for the 5 projections, 5 norms, 3 view-side NeRF params, `gaussian_token`.

**Init.** Fresh from `microsoft/renderformer-v1-base` via `training/weight_transfer.transfer_weights` (not V10b — the encoder is structurally different and cannot be partially loaded). The 12-layer transformer, view-transformer, DPT decoder, and `reg_tokens` copy from RF; the 19 Gaussian-specific params init random. `weight_transfer.py` now derives "Gaussian-specific" dynamically from the RF state-dict (set difference) instead of a hardcoded list, so it's robust as the encoder evolves.

**Loss.** Log-HDR L1 only (LPIPS=0) — clean baseline isolating the architectural change. LPIPS fine-tune (V11b) deferred until V11 converges, mirroring the V9→V10b sequencing.

**Sanity (job 30616892, 2026-05-19).** Forward `(1, 2, 3, 32, 32)` on CUDA with bf16 autocast; grads flow into every per-field projection; `gaussian_specific_param_names()` identifies the 19 new params; `freeze_backbone` trainable count matches. Side fixes that landed: `tmp/__init__.py` for module discovery, `SLURM_SUBMIT_DIR` for sbatch cwd, `uv run --frozen` to skip dependency resolution, numpy publish-time allowlist in `~/.config/uv/uv.toml` (needed because of the global `exclude-newer = "7 days"` supply-chain hardening).

**Pre-launch optimization.** V10b at bs=4 sat at 40.6/46 GB, but training H5 scenes are small (500–2,600 Gaussians; inference runs 5k–30k), so most batches' `max_N_in_batch` is well under the worst case → headroom likely. Probing `bs ∈ {4, 5, 6, 8}` via 4 parallel sbatch jobs (`runs/probe_v11_bs{4,5,6,8}.sh`, 4 GPUs each, `--max_samples 32`, 1 phase-2 epoch, `--skip_phase1`). Each has an `nvidia-smi -l 10` sidecar logging peak `memory.used`. Decision rule: largest bs whose peak stays under ~44 GB on all 4 ranks. Data unchanged from V10b (`data_v9/h5s` + `data_v9/renders`) so V11 vs V10b is an apples-to-apples comparison of the encoder. Denser-N training data deferred to a future V12 experiment.

**Probe results (jobs 30617186-89, 2026-05-19, max_samples=32, 1 phase-2 epoch each).**

| bs | result | probe peak VRAM (GPUs 0–3) | notes |
|---|---|---|---|
| 4 | COMPLETED 0:0 | 33.2 GB | Elapsed 5:18. Reference probe point. |
| 5 | COMPLETED 0:0 | 35.9 GB | Elapsed 5:19. Δ vs bs=4 = +2.7 GB. |
| 6 | COMPLETED 0:0 | smi missed peak (271 MiB readings only) | Elapsed 2:10, training itself 7.7s — cudnn/flash-attn cache warm from bs=4/5, smi 10s polling never sampled during the active step. |
| 8 | FAILED 1:0 | OOM on rank 2 | `Tried to allocate 96 MiB. GPU 2 has total 47.40 GiB of which 7.19 MiB is free.` |

**Backend confirmed:** `attention backend: flash_attn` printed at startup on every probe — flash-attn imports cleanly under the venv (`uv run --frozen`), no SDPA fallback.

**Decision: bs=5 for V11.** V10b at bs=4 sat at 40.6 GB in production but our probe at bs=4 (small-data) saw only 33.2 GB — a ~7.4 GB probe-to-production gap from outlier high-N batches. Linear-extrapolating the same gap: bs=5 production peak ~43 GB (≥3 GB headroom on 46 GB cards), bs=6 production peak ~46 GB (OOM risk), bs=8 confirmed OOM even on small data. bs=5 gives a 25% throughput bump over V10b (global bs 20 vs 16) with verified headroom.

**Launch.** `runs/train_v11_perfield.sh` set to `--batch_size 5`, phase1=5 + phase2=100 epochs, log-L1 only, 4×g4 / 168 h walltime.

**Launch attempts (2026-05-19).** Three sbatch tries before V11 stabilized:
- Job `30617444` — cwd fail (`$(dirname "$0")/..` resolved to SLURM staging dir). Cancelled.
- Job `30617456` — uv resolver hit numpy publish-time exclusion (script lacked `--frozen`). Cancelled.
- Job `30617463` — DDP `find_unused_parameters` crash in Phase 1 (FAILED 5:14): Phase 1 freezes most of the 195M backbone *after* DDP wraps, so DDP's reducer saw params it expected grads for that never produced any. This had been latent — V9/V10/V10b all used `--skip_phase1 --resume`, so a real Phase 1 step + DDP was never exercised on this codepath. Fix: pass `find_unused_parameters=True` at DDP construction (`training/train.py`); small per-step overhead, no behavioral regression. Root-causing the specific unused param deferred — `find_unused_parameters=True` is the documented escape hatch and Phase 1 is only 5 epochs.

**V11 RUNNING.** Job `30617577`, launched 2026-05-19 ~14:48. `attention backend: flash_attn` and Phase 1 trainables (19) confirmed printed; awaiting first epoch loss. Persistent monitor `b6h2tgc2b` armed for state/epoch/error events.

### Eval-suite refactor (in-flight; CPU work complete, GPU steps queued)

For V11 vs V10b we wanted comparisons across multiple diverse scenes, not just tomatoes. The eval pipeline (`data_external/run_tomatoes.py`) was scene-specific — refactored into a parameterized version, and 3 additional Objaverse_Splats val scenes added to the registry.

- `data_external/scene_configs.py` (new) — `SceneConfig` dataclass + `SCENES` registry with `tomatoes`, `house` (UID `4272ba78...`, chunk `000-031`), `dragon` (UID `7cddac29...`, chunk `000-045`), `cartoon` (UID `235e2e4c...`, chunk `000-091`). Picked from `data_v9/object_list_val.json` for diversity vs tomatoes' single-organic profile: structured multi-object (house), thin/specular geometry (dragon), stylized OOD-leaning (cartoon).
- `data_external/run_scene.py` (new) — generalized pipeline. Takes `--scene <name>` from the registry; everything else (norm, scoring, pruning, render, metrics, overview grids) is scene-agnostic. Scene-specific bits (norm_scale vs norm_target_aabb, X-flip, orbit camera) live in `SceneConfig`.
- `data_external/run_tomatoes.py` — 4-line shim preserving `python -m data_external.run_tomatoes` so existing `runs/eval_v10b_tomatoes.sh` still works.
- `data_external/prep_objaverse_scene.py` (new) — `--scene <name>` downloads PLY by UID from the HF chunk zip + renders the 14-view gsplat-full reference. Supports `--download-only` for CPU-side prep.
- `runs/prep_eval_v11_scenes.sh` (new) — one-shot 1-GPU sbatch prepping all 3 new scenes.
- `runs/eval_v11_scene.sh` (new) — per-checkpoint per-scene eval template: `sbatch --export=ALL,SCENE=<name>,EP=<n> runs/eval_v11_scene.sh`.

**Done so far:** 3 PLYs downloaded CPU-side to `data_external/{house,dragon,cartoon}/raw.ply` (3.4 MB each, 50k Gaussians).

**Pending GPU work:** `sbatch runs/prep_eval_v11_scenes.sh` (renders gsplat-full references; queues behind V11) → then `runs/eval_v11_scene.sh` per (scene, ckpt) once V11 has checkpoints.

### V11 Phase 1 results (job 30617577, 2026-05-19)

Phase 1 ran clean across all 5 epochs (~2235 s/epoch at bs=5 / 4 GPUs / flash-attn). Loss curve was smooth and monotone:

| Phase 1 epoch | train avg | val | lr (cosine end) |
|---|---|---|---|
| 1 | 0.013833 | – | 9.05e-04 |
| 2 | 0.010068 | – | 6.58e-04 |
| 3 | 0.008688 | – | 3.52e-04 |
| 4 | 0.008512 | – | 1.05e-04 |
| 5 | 0.008389 | 0.008683 | 1.00e-05 |

Step-level binning showed the actual descent: 0.015 → 0.013 in ep 1, 0.013 → 0.009 in ep 2, ~0.008 by mid-ep 3, fully flat through ep 4–5. The "good learning" happened in ep 1–2; ep 4–5 were pure over-specialization. `phase1_epoch_5.pt` (781 MB, encoder-only) saved as the fallback.

### V11 Phase 2 plateau (low-LR, 2026-05-19)

Phase 2 began with the V9/V10b fine-tune recipe (lr=5e-5, cosine over 100 epochs). All 5 trained epochs flat-to-rising:

| Phase 2 epoch | train | val | lr |
|---|---|---|---|
| 1 | 0.014307 | – | 5.00e-05 |
| 2 | 0.013833 | – | 5.00e-05 |
| 3 | 0.013835 | – | 4.99e-05 |
| 4 | 0.013834 | – | 4.98e-05 |
| 5 | 0.013888 | **0.014490** | 4.97e-05 |

Notably, Phase 2 ep 5 val (0.014490) is **66% worse than Phase 1 ep 5 val (0.008683)** — the unfreezing actively degraded the model. Cancelled at end of ep 5.

### V11 phase1_ep5 eval on tomatoes (job 30619102, 2026-05-19)

To anchor the diagnosis, ran `data_external.run_scene` on `phase1_epoch_5.pt` (the best V11 ckpt — encoder trained, backbone at RenderFormer init). Side bug surfaced: `run_scene.py` hardcoded `GaussianFormerConfig()` with default `pe_type='rope'`, causing `load_state_dict` to reject the per-field state. Fixed by threading `--pe_type` and `--scale_pe_num_freqs` through to both the script and the SLURM wrapper.

| N | V11 phase1_ep5 (vs full) | V10b ep26 (vs full) | Δ |
|---|---|---|---|
| 5000 | 23.07 dB | 25.80 dB | −2.7 |
| 10000 | 23.24 dB | 26.70 dB | −3.5 |
| 20000 | 22.87 dB | 27.41 dB | −4.5 |
| 30000 | 22.56 dB | 27.81 dB | −5.3 |

Confirms: the per-field encoder learned a *useful but partial* mapping (23 dB > random), but the unchanged RenderFormer backbone can't fully decode it. The V11-vs-V10b comparison is meaningless until Phase 2 actually works.

### V11 Phase 2 hi-LR retry (job 30619107, started 2026-05-19 ~23:10)

Hypothesis: lr=5e-5 was too low to pull the backbone out of its RenderFormer init given the new input distribution. Restarted Phase 2 from `phase1_epoch_5.pt` (skip Phase 1) with `--phase2_lr 2e-4` (4× higher), output to `checkpoints_v11_hi_lr/`. Live trajectory:

| Phase 2 epoch | low-LR (5e-5) | hi-LR (2e-4) |
|---|---|---|
| 1 | 0.014307 | 0.019580 |
| 2 | 0.013833 | 0.014827 |
| 3 | 0.013835 | 0.014827 |
| 4 | 0.013834 | (in flight) |

ep 1 spiked (cold start at hi-LR), ep 2–3 plateaued ~0.0010 *above* the low-LR plateau. Hi-LR disturbed the backbone more than low-LR but didn't reach a better basin. Verdict pending ep 4: <0.014 = continue, ≥0.014 = abort + pivot to short-Phase-1.

### Diagnosis: freeze-then-unfreeze trap

Two failed Phase 2 attempts at different LRs both plateau (low at 0.0138, hi at 0.0148). The encoder dropped Phase 1 loss to 0.0084 (it learned something), but unfreezing the backbone consistently fails to find a better minimum. This isn't an LR problem — it's a **freeze-then-unfreeze trap**: during the 5-epoch Phase 1 the encoder over-specialized to a frozen RenderFormer-init backbone, putting joint optimization into a saddle that gradient descent (at any LR we tried) can't escape.

### Short-Phase-1 recipe (queued, ready to launch on wake)

`runs/train_v11_perfield_short_phase1.sh`:
- **Phase 1**: `--phase1_lr 3e-4` (3× lower than V11's 1e-3), `--phase1_epochs 2` (cut before the plateau — ep 2 was the elbow in the descent).
- **Phase 2**: `--phase2_lr 5e-5`, `--phase2_epochs 100` (V9 recipe).
- Init from `renderformer-v1-base` (no `--resume`), per-field arch unchanged, bs=5 / 4×g4.
- Output to `checkpoints_v11_short_p1/`.

Side patch in `training/train.py`: the final epoch of each phase now always saves a ckpt, even off the `save_interval` cadence — otherwise `phase1_epochs=2` with `save_interval=5` would leave no Phase 1 ckpt.

**Decision tree (autonomous overnight):**
- Hi-LR ep 4 lands <0.014 → gamble alive, let it continue.
- Hi-LR ep 4 lands ≥0.014 or crashes → wait for wake; ready commands: `scancel 30619107 && sbatch runs/train_v11_perfield_short_phase1.sh`.

### Hi-LR aborted; short-Phase-1 also plateaued (2026-05-20)

Hi-LR run (30619107): ep 2/3/4 all 0.014827 — plateau confirmed, cancelled. Short-Phase-1 run (`30619669`, `runs/train_v11_perfield_short_phase1.sh`): Phase 1 (2 ep, lr=3e-4) trained clean (0.0138 → 0.0126, `phase1_epoch_2.pt` saved). Phase 2 then plateaued **identically to the original V11**:

| P2 epoch | reported avg | step-mean |
|---|---|---|
| 1 | 0.013833 | 0.013412 |
| 2 | 0.013833 | 0.013406 |
| 3 | 0.013832 | 0.013678 |
| 4 | 0.013851 | – |
| 5 | 0.013864 (val 0.014475) | – |
| 6 | 0.013830 | 0.014018 |
| 7 | 0.013830 | 0.013333 |
| 8 | 0.013839 | – |

Step-level means confirm a real plateau (not a logging artifact). Three Phase 2 configs now all stuck ~0.0138. The freeze-then-unfreeze-trap hypothesis is **disproven** — short Phase 1 was designed to avoid it and plateaued at the exact same place.

### ~~Hypothesis: DDP bucket bug in the in-process phase transition~~ (SUPERSEDED — wrong)

> This section is kept as an investigation record. It was **disproven** by the
> `train_v11_phase2_resume.sh` run below and is not the real cause.

Initial hypothesis: `train.py` wraps the model in `DDP(..., find_unused_parameters=True)` then calls `freeze_backbone`, so Phase 2's `unfreeze_all` re-enables backbone params that DDP never synced → replicas drift → plateau. The `--skip_phase1 --resume` recipe was proposed as the fix.

**Why it was wrong:** the `runs/train_v11_phase2_resume.sh` run (job `30619869`) used exactly `--skip_phase1 --resume` — DDP constructed once with all params trainable, no freeze/unfreeze — and **plateaued identically** (ep 1 0.013841, ep 2 0.013833). That codepath has no DDP bucket issue, so the bucket bug cannot be the cause.

### Root cause (confirmed): scale fed to NeRF encoding ~14× out of range

Checked V9's *actual* Phase 2 trajectory: ep 1 = **0.003839**, descending smoothly to 0.000877 by ep 50. V11 plateaus at 0.0138 — ~4× worse than where V9 *started*. So V11 was never "slow", it was broken; and the only thing differing from V9 is the encoder.

The bug, verified on `data_v9/h5s/scene_0000.h5`: `construct_sequence` computes `log_scale = log(scale)`. Real scales are `[1e-6, 0.11]` → `log_scale ∈ [-13.8, -2.2]`, range 11.6 wide. `NeRFEncoding` requires inputs in ~`[0,1]` (per its own docstring) so the lowest band is smooth. At this range **even band 0 (freq 1) spans 1.8 periods** — `sin(log_scale)` is non-monotonic, non-injective. All 36 sinusoidal dims of the 39-D scale encoding are aliased oscillation; only the 3 raw `include_input` dims carry usable scale. The model effectively cannot read Gaussian scale — and scale is the splat footprint, so the renderer defaults to an average blur and plateaus. This explains both the chronic blurriness *and* the flat loss.

Position is fine by contrast (range `[-0.45, 0.29]`, band 0 spans 0.1 periods — textbook NeRF).

Secondary: `token = gaussian_token + Σ RMSNorm(proj_i)` over 5 fields → token RMS ≈ √6, vs √2 for the working `rope` path; the pretrained backbone's residual stream is balanced for √2.

### Fix applied: per-field encoder rewrite (2026-05-20)

`gaussianformer/models/gaussianformer.py`, `nerf_perfield` branch:
- **Scale no longer NeRF-encoded** — `scale_proj` is now `Linear(3, 768)` on raw `log_scale`; `scale_pe` removed. (Position keeps NeRF.)
- **Single final norm** — the 5 per-field RMSNorms replaced by one `gaussian_norm` on the summed projections.

Verified on CPU with realistic data ranges: fixed `nerf_perfield` token RMS = 1.41, matching the `rope` path's 1.40 (was ~2.45). `tmp/sanity_perfield.py` updated for the new param names. `scale_pe_num_freqs` config field is now unused (left in place; harmless).

### v11b run: scale fix did NOT move the plateau (job 30620114, 2026-05-20)

Launched the scale-fixed encoder (`runs/train_v11b_perfield_fixed.sh`). GPU sanity passed. Phase 1 (2 ep, lr 3e-4) → 0.0138. Phase 2: ep 1 = 0.013838, ep 2 = 0.013848 — same plateau. Killed at ep 2.

Five Phase 2 runs now plateau at 0.0138 to three sig figs (4× LR range, 2 Phase-1 schedules, broken *and* fixed encoder). That degree of reproducibility means the floor is **not** an optimization plateau and **not** the input encoder — it is a fixed component of the model that is broken identically every time.

### Root cause (confirmed, structural): RoPE was disabled for `nerf_perfield`

The `renderformer-v1-base` backbone is RoPE-pretrained (`RenderFormerConfig.pe_type` defaults to `'rope'`; V9/V10b set `rope_dim=12` and worked, which only succeeds against a RoPE-trained backbone). `transfer_weights` copies that backbone's RoPE-trained attention weights into GaussianFormer.

But the `nerf_perfield` path left `rope_dim = None` in **both** transformers — `gaussianformer.py` (view-independent) and `view_transformer.py` (view-dependent). `attention.py`: `rope_dim=None` → no RoPE applied. So every V11 run ran RoPE-trained attention weights **without RoPE** — Q/K never rotated by position, attention patterns meaningless, the entire scene encoder + view decoder compromised. This is input-encoder-independent → explains the identical 0.0138 floor across all 5 runs.

The V11 design assumed the per-field NeRF position encoding *replaces* RoPE. It can't — you cannot drop RoPE from a RoPE-pretrained backbone. NeRF encoding must be additive to RoPE.

**Fix:** `nerf_perfield` now sets `rope_dim = pos_pe_num_freqs` in both `gaussianformer.py` and `view_transformer.py`. Verified on CPU: `nerf_perfield` now has both `rope_emb` buffers identical to the `rope` path; names shared with RenderFormer rose 275 → 277 (the 2 rope buffers align); the 15 GF-only params are exactly the per-field encoder. (The `pe_type='nerf'` path has the same latent bug — left as-is, never used.)

Net: three fixes in the `nerf_perfield` path — (1) RoPE re-enabled [primary], (2) scale plain-linear not aliased-NeRF, (3) single final token norm.

Next: `runs/train_v11c_perfield.sh` → `checkpoints_v11c/`. Verdict at Phase 2 ep 1 — descent toward V9's ~0.004 confirms the fix; another 0.0138 freeze means deeper instrumentation.

### v11c: RoPE fix helped Phase 1 but Phase 2 still plateaued (job 30620496)

v11c (RoPE fixed) Phase 1 reached **0.0063** — the first time any V11 run broke below 0.0138 at any stage, proving the RoPE fix did something real (the frozen backbone now functions). But Phase 2: ep 1 = 0.014143, ep 2 = 0.013833 — back at the plateau. Killed at ep 2.

### Rendering v11c phase1_ep2: a featureless grey blob

Rendered `checkpoints_v11c/phase1_epoch_2.pt` (the 0.0063 model) on tomatoes. PSNR vs full: 21.9 / 19.4 / 16.9 / 15.6 dB at N = 5k/10k/20k/30k — and **degrading hard with N**. The image: a featureless grey-white blob — no color, no structure, correct rough position and extent, and the blob grows with N. The "0.0063 Phase 1" was never good: L1 is computed in log-space on ~90%-black images, so a correctly-placed grey blob lands at ~0.006. **The loss number cannot distinguish a blob from real content — only the render can.**

### Instrumenting the encoder: scale drowned every other field (self-inflicted)

`tmp/instrument_encoder.py` measured per-field embedding RMS on a real batch through the v11c encoder:

| field | emb RMS | token Δ if removed |
|---|---|---|
| pos | 0.56 | 15% |
| **scale** | **3.77** | **117%** |
| rotation | 0.42 | 11% |
| color | 0.47 | 13% |
| opacity | 0.60 | 16% |

`scale_emb` was ~7× every other field; the summed token was almost purely scale. Cause: the v11b "fixes" — replacing aliased scale-NeRF with a plain `Linear` on *raw* `log_scale` (input magnitude ~10 vs ~1 for other fields) made `scale_emb` hot, and removing the per-field RMSNorms (the safeguard that kept fields balanced) let it dominate unchecked. **Fix:** restored per-field RMSNorm on all 5 projections + kept the final norm. Re-instrumented on a fresh model: all 5 fields now RMS 1.0, uniform 45.5% ablation — balanced.

### v11d: balanced encoder, OOM, resumed — Phase 2 STILL plateaued (jobs 30623103, 30624257)

v11d (RoPE + balanced fields) Phase 1 → 0.0138; Phase 2 OOM'd on all 4 ranks (landed on 44 GB g4 cards; bs=5 peaks ~43 GB — fits 48 GB cards, not 44 GB). Resumed Phase 2 from `phase1_epoch_2.pt` at bs=4 (`train_v11d_phase2_resume.sh`). Phase 2: ep 1 = 0.013837, ep 2 = 0.013835 — plateau, again.

### Verdict: the input encoder was never the cause

**Seven Phase 2 runs, every one at 0.0138** — RoPE on/off, scale aliased/plain/dominated/balanced, every LR, every Phase-1 schedule. Three real, verified encoder bugs fixed; Phase 2 never moved. The plateau is not in the encoder.

### Control experiment: V9's recipe fails on current code

Ran `runs/control_rope.sh` — V9's exact recipe (`pe_type=rope`, `--skip_phase1 --resume checkpoints_v4/phase1_epoch_20.pt`, lr 5e-5) on the *current* codebase (job 30624978). Phase 2 ep 1 = **0.013838** — plateaued. V9's own recipe, run today, fails the same way. So the regression is in **shared code changed since V9**, not the encoder. Every V11 run varied the encoder and held shared code fixed — which is why encoder tweaking could never have found it.

### Root cause (confirmed): `find_unused_parameters=True`

`git diff main -- training/train.py`: for a `rope` run, exactly one change is behaviorally active — `DDP(...)` gained `find_unused_parameters=True`. It was added during the V11 effort to silence the Phase-1 freeze DDP crash (job 30617463). Consistency check is exact: **every run with the flag (7 V11 runs + the control) plateaued at 0.0138; V9/V10b predate it and reached 0.0009.** With `find_unused_parameters=True`, DDP marks "unused" params gradient-ready from a per-iteration graph traversal; data-dependent masking (padded Gaussians / `valid_mask`) can make the used/unused set differ across ranks, so DDP all-reduces inconsistent gradient sets and silently corrupts training.

**Fix:** `find_unused_parameters = not args.skip_phase1` in `train.py` — in-process Phase 1 needs it (frozen backbone), `--skip_phase1` runs (all params trainable, V9's mode) get `False`.

### Bisect re-run (job 30625057): find_unused_parameters IS a bug — for rope

Re-ran `control_rope.sh` with `find_unused_parameters = not args.skip_phase1`. Phase 2 ep 1 = 0.0115 avg, and the within-epoch binning showed a **clean monotonic descent**: 0.0157 → 0.0148 → 0.0127 → 0.0119 → 0.0091 → 0.0082 across the epoch — the first genuine descent in the whole investigation. So `find_unused_parameters=True` was a real bug. Confirmed fix for the `rope` path.

### But the per-field encoder still plateaus — there are TWO bugs

Re-ran v11d Phase 2 (`train_v11d_phase2_resume.sh`, `--skip_phase1` → `find_unused_parameters=False`), job 30625146. Phase 2 ep 1-11 all flat at ~0.0138 (val 0.0145). The within-epoch ep-1 binning was **identical** to the pre-fix v11d run — the `find_unused` fix changed nothing for the per-field path.

| config | find_unused | result |
|---|---|---|
| rope | True | plateau |
| rope | **False** | **descends** ✓ |
| per-field | True | plateau |
| per-field | **False** | **plateau** ✗ |

So: `find_unused_parameters=True` is one bug (fixed, affects rope). The per-field encoder has a **separate, still-open bug** — it plateaus even with `find_unused` fixed. The earlier claim "the encoder was never the cause" was an overreach: the rope control only proved `find_unused` was *a* bug, not that the encoder was clean.

### compare_forward.py: the per-field token is NOT degenerate

`tmp/compare_forward.py` — rope vs per-field models, both weight-transferred from the *same* RenderFormer (identical 12-layer backbone), one real Gaussian batch through `construct_sequence` + `model.transformer`:

| | rope | per-field |
|---|---|---|
| per-layer activation RMS | 4.2 → 13.9 | 3.2 → 13.6 (ratio 0.8–1.1× every layer) |
| final scene-token RMS | 13.94 | 13.64 |

The per-field token propagates through the backbone essentially identically to a rope token — no blowup, no collapse. The per-field encoder is not producing a token the backbone chokes on. **The plateau is a training-dynamics problem, not a broken-forward one.**

### Open: the per-field training-dynamics bug

Still unsolved. v11d `phase2_epoch_5.pt` / `phase2_epoch_10.pt` are saved (plateaued models, available to render).

The bug is precisely isolated: **encoder-specific** (rope descends on current code, per-field plateaus) and **training-dynamics, not forward** (`compare_forward` shows the per-field token propagates through the backbone within ~10% of a rope token at every layer). Two confounded candidate causes:
- **(A) Checkpoint provenance** — every per-field run resumed a per-field Phase-1 checkpoint that itself plateaued in Phase 1.
- **(B) The per-field architecture** — 5 per-field RMSNorms force every field to exactly equal magnitude; the encoder can only rotate field directions, not weight fields by importance.

### v11e: discriminating experiment (job 30629297, 2026-05-21)

`runs/train_v11e_perfield_jointscratch.sh` — `nerf_perfield`, `--skip_phase1`, **no `--resume`**: fresh RenderFormer transfer with a random per-field encoder, Phase 2 trains everything jointly from step 1. No Phase-1 checkpoint in the picture, so it isolates (A) vs (B):
- plateaus at 0.0138 → cause (B), the architecture.
- descends toward ~0.004 → cause (A); fix is a proper Phase-1 warmup.

`--skip_phase1` → `find_unused_parameters=False` (the rope-confirmed-correct path); bs=4. Verdict from the within-epoch binning of Phase 2 ep 1 (~46 min): a clean monotonic descent (like the rope control) = (A); flat ~0.0138 = (B).

### Verdict (job 30629297): cause (B) — the per-field architecture

v11e Phase 2 ep 1 = **0.013842** — the chronic plateau to the fourth decimal (prior per-field runs: 0.013833–0.013841). within-epoch ep-1 binning: 0.01589 → 0.01422 → 0.01324 → 0.01350 → 0.01470 — a token drop then a stall. The rope control with `find_unused` fixed, same code, descended 0.0157 → 0.0082 across ep 1; the per-field encoder cannot.

v11e was the cleanest possible isolation: fresh RenderFormer transfer, *random* per-field encoder, joint Phase 2 from step 1, **no Phase-1 checkpoint anywhere**. It still plateaus. Checkpoint provenance (A) is ruled out — the per-field architecture itself caps the model.

**Why the architecture caps it — precisely.** Per-field projections summed *without* the per-field norms are mathematically a single linear map over the concatenated features: `Σ_f W_f·feat_f = [W_pos|W_scale|…]·[feat_pos;feat_scale;…]`. So the *entire* difference between the per-field encoder and a plain concat encoder is the 5 per-field RMSNorms. RMSNorm divides each field's projection output by its own per-Gaussian RMS (over the 768 channels) — a nonlinearity that erases relative salience: a Gaussian at an "important" position and one at a "boring" position get their `pos_emb` normalized to the same magnitude. The encoder cannot make one field, or one Gaussian, louder than another. `compare_forward.py` already showed the per-field *forward* is healthy (token within ~10% of a rope token at every layer) — the cap is in what the encoder *can represent*, not numerical dynamics.

The per-field RMSNorms were not gratuitous — without them `scale_emb` ran 7× hot and dominated the token (the v11c bug). So per-field is caught between two failures: norms off → scale domination; norms on → per-Gaussian salience erased. The decomposition itself is the dead end.

### v11f: the concat encoder (`nerf` mode, repurposed) — 2026-05-21

The fix follows directly from the framing above: per-field-sum **minus the 5 norms** is exactly a single linear map over the concatenated fields. The concat encoder is that map:

```
feat  = cat[ nerf(pos)   # 75   3 dims, 12 freqs, +input
             log_scale   # 3
             quat        # 4
             color       # 3
             opacity ]   # 1   -> 86-dim
token = gaussian_token + norm( Linear(86, 768)(feat) )
```

One shared `Linear(86,768)` over the whole concat — weights every field freely via its columns, exactly like rope's `Linear(14,768)`, but with position lifted into a NeRF basis (the original V11 blur-fix hypothesis). RoPE stays on. No per-field norms, no sum.

**No input standardization.** An earlier draft standardized `log_scale` by training-set mean/std — dropped: those are dataset-fitted magic numbers, and the architecture does not need them. A *shared* projection with a bias absorbs per-field offset (the bias) and per-field magnitude (its learned weight columns) on its own — which is precisely why the concat encoder fixes per-field, where *isolated* per-field projections + norms made magnitude matter. `log` on scale stays: parameter-free, and the canonical 3DGS representation of a positive multiplicative quantity. `clamp(min=1e-6)` is a standard log-of-zero guard, not a fitted constant.

Decided: repurpose the (unused, no-checkpoint) `nerf` pe_type — keeps the mode count at 3; the `nerf` view transformer is made pure-rope (identical to `rope`), so `nerf` vs `rope` isolates exactly the scene encoder.

Implemented in `config.py` / `gaussianformer.py` / `view_transformer.py`. CPU sanity (`tmp/compare_forward.py`, rope vs concat through the shared backbone): the concat token propagates healthily — per-layer activation RMS 0.5–1.2× rope, final scene-token RMS 10.2 vs rope 14.3 (cooler, no blowup/collapse), gradients same order at every layer.

`runs/train_v11f_concat.sh` — recipe identical to v11e (`--skip_phase1`, no `--resume`: fresh RenderFormer transfer + random concat encoder, joint Phase 2), so the only variable vs v11e is the encoder architecture. Verdict from the within-epoch binning of Phase 2 ep 1: descends like the rope control (~0.008 within ep 1) → the concat encoder is the fix; flat 0.0138 → the bug is deeper than the encoder.

### v11f result: concat plateaus too — theory falsified, and a confound found (job 30629625)

First v11f submit (30629550) died at step 240 to a single-rank NCCL collective timeout — rank 2 hung in non-collective code, no Python exception / NaN / OOM anywhere; a transient infra stall, not the code. Resubmitted with `--exclude=firefoot-13`.

v11f Phase 2 ep 1 = **0.013837**. within-epoch binning: 0.0139 → 0.0178 → 0.0138 → 0.0124 → 0.0136 → 0.0132 → 0.0161 → 0.0137 — flat and noisy, no descent. **The concat encoder plateaus too.**

**The per-field-norm theory is falsified.** concat *is* per-field-projections-summed minus the 5 norms (shown mathematically in the v11f design above). Per-field-with-norms plateaus (v11e 0.013842); per-field-minus-norms plateaus (v11f 0.013837). Removing the norms changed nothing — they were never the bottleneck.

**Confound found in the comparison that "proved" rope works.** The rope control that descended (0.0157→0.0082, the "Bisect re-run" above) ran `--resume checkpoints_v4/phase1_epoch_20.pt` — an **already-trained** rope encoder. v11e and v11f start from **random** encoders. So "rope descends / nerf plateaus" confounds two variables: encoder init (trained vs random) and architecture (rope vs nerf). And the evidence now favors *init*: v11e (per-field) and v11f (concat) are very different architectures yet plateau at identically 0.01383-84 — when the architecture varies wildly and the result doesn't move, the architecture is not the variable.

### Control: rope, random init, joint Phase 2 from scratch (job 30629854, 2026-05-21)

`runs/control_rope_jointscratch.sh` — v11e/v11f's exact recipe (`--skip_phase1`, **no `--resume`**: fresh RenderFormer transfer + random encoder, joint Phase 2) with `pe_type=rope`. The only variable vs v11f is the encoder architecture; this is the control that should have been run instead of the resumed `control_rope.sh`.
- descends toward ~0.008 within ep 1 → random-rope-joint works → the nerf architecture is genuinely the problem.
- plateaus at 0.0138 → random-joint plateaus regardless of encoder → the plateau is the **recipe** (a Phase-1 encoder warmup is needed), and every encoder comparison since `control_rope.sh` was apples-to-oranges.

Verdict from the within-epoch binning of Phase 2 ep 1 (~46 min).

### Ruled out: collate_fn zero-padding (2026-05-21)

Checked whether `collate_fn`'s zero-padding interacts badly with the nerf encodings — a zero Gaussian gives `log_scale = log(1e-6) = -13.8` and `NeRF(pos=0)` has 36 cos-ones, so padding rows become large/structured tokens (benign for rope: `Linear(zeros)` → bias). Two independent findings kill it:
- **No padding ever occurs.** All 2667 `data_v9/h5s` training scenes have exactly N=5000 Gaussians (min=max=5000). `collate_fn` pads to the batch max, which is always 5000 → `pad_size=0` every batch. The padding branch is dead code for V9-data training.
- **Even forced padding is contained.** `tmp/test_padding_leak.py` — a batch with a truncated scene (2000 padded rows) run through `construct_sequence + transformer` with normal vs random-garbage padding: real-token outputs are bit-identical (`max|Δ| = 0.000e+00`) for both rope and nerf. The padding tokens themselves change (Δ 697 nerf / 274 rope — confirming zero-padding *is* pathological for nerf), but the key-padding mask fully contains them.

collate_fn is not a plateau suspect and needs no modification. The live discriminator remains the rope-jointscratch control.

### Verdict: the plateau is the recipe, not the encoder (job 30629854)

rope-jointscratch Phase 2 ep 1 = **0.013840**, ep 2 = 0.013834. within-epoch ep-1 binning: 0.01588 → 0.01423 → 0.01323 → 0.01352 → 0.01457 → 0.01429 — **bin-for-bin near-identical to v11e** (per-field: 0.01589 / 0.01422 / 0.01324 / 0.01350 / 0.01470). Two completely different encoders, the same loss curve step-for-step → the loss in this regime does not depend on the encoder at all.

**The encoder is ruled out — fully.** rope, per-field, and concat all plateau at 0.01384 under `--skip_phase1` + random encoder + joint Phase 2. The whole V11 encoder investigation (v11 → v11f) was a confound: every run used `--skip_phase1` with an untrained encoder, and the one baseline that descended (`control_rope.sh`, 0.0157→0.0082) *resumed a trained encoder* (`checkpoints_v4/phase1_epoch_20.pt`). "rope works, nerf doesn't" was always "trained encoder works, random encoder doesn't."

**Cause:** joint Phase 2 from a random encoder does not learn the scene — it settles into a scene-independent blob (~0.013, the v11c grey-blob render). V9/V10b reached 0.0009 because they ran a **Phase-1 encoder warmup** (frozen backbone) first; V11 dropped Phase 1 because in-process Phase 1 crashed under DDP (job 30617463: the backbone was frozen *after* the single DDP wrap → reducer waits on gradients that never come).

## V12 — corrected two-phase recipe + the concat encoder (job 30630480, 2026-05-21)

`training/train.py` fixed: DDP is now constructed **per phase, after** `requires_grad` is set (`wrap_ddp`). Freezing the backbone before the Phase-1 wrap means DDP registers only the trainable encoder params → `find_unused_parameters=False` is correct, no crash, no corruption. At the Phase-1→2 transition the model is unfrozen and re-wrapped (the old wrapper's reducer hooks go inert once unused). `--skip_phase1` removed; `--resume` kept as a phase-aware crash-recovery escape hatch (carries phase/epoch/optimizer/scheduler). `--pe_type` default → `nerf`.

`runs/train_v12.sh` — single end-to-end run: RenderFormer transfer → Phase 1 (20 ep, lr 1e-3, encoder warmup) → Phase 2 (100 ep, lr 5e-5, joint), `pe_type=nerf` (concat encoder), bs=4, 4×g4, no `--skip_phase1`/`--resume`. CPU pre-check confirmed `freeze_backbone` for `pe_type=nerf` yields exactly the 4 encoder params (68k trainable in Phase 1; 194.9M in Phase 2).

Verdict signal: Phase 2 ep 1 within-epoch binning **descends** (V9's Phase 2 ep 1 was 0.003839) rather than sitting flat at 0.0138 → the recipe is fixed and the concat encoder is finally testable for the V11 blur hypothesis.

### V12 results: the recipe is fixed — the plateau is escaped (2026-05-22)

**Phase 1** (encoder warmup, frozen backbone, 68k trainable) ran clean — no crash (the freeze-before-DDP-wrap fix works). ep 1 avg 0.00798 → ep 20 avg 0.003720 (val 0.003957): a clean descent to the encoder-warmup floor. **Phase 1→2 transition clean** — `unfreeze_all` + fresh DDP wrap printed `phase2: 194,924,431 trainable`, no crash (the per-phase DDP re-wrap works).

**Phase 2** — the plateau is **escaped**, the first run in the whole V11/V12 effort to do so:

| Phase 2 epoch | avg loss |
|---|---|
| 1–8 | ~0.0138 (plateau) |
| 9 | 0.012045 |
| 10 | 0.005315 (val 0.003221) |
| 11 | 0.002611 |
| 12 | 0.002256 |

Within-epoch binning across ep 9–11 is a clean monotonic descent (0.0120 → 0.0024). The model is now descending in V9 territory (V9's Phase 2 *started* at 0.0038; V12 is below that by ep 11) with ~88 epochs left.

**The 8-epoch plateau before breakout** — a wrinkle, not a failure. Phase 1 left the model at 0.0037 with a *frozen* backbone; unfreezing in Phase 2 jumped the loss to 0.0144 and it sat there 8 epochs. The Phase-1 encoder was tuned to the frozen RenderFormer-init backbone — once the backbone started moving, the encoder–backbone match broke and the model fell into the 0.0138 attractor, then took 8 epochs of joint training to climb out. V9's Phase 2 started at 0.0038 with no such transient. **Future refinement (V12b or a re-run): a Phase-2 LR warmup, or a lower initial `--phase2_lr`, should soften the unfreeze shock and skip the ~5 h / 8-epoch detour.** Not worth disrupting the current run — it has recovered and is descending normally.

Conclusion (refined). What the evidence solidly supports: (1) the encoder architecture does not cause the plateau — rope, per-field, and concat plateau bin-for-bin identically; (2) the corrected two-phase recipe with a proper Phase-1 warmup escapes it (V12, ep 9). What is *not* proven: whether a no-warmup run (`--skip_phase1`, random encoder) would also escape given enough epochs — v11e/v11f/the rope-jointscratch control were all cancelled at ep 1–2, before V12's epoch-9 escape point. The 0.0138 plateau is now known to be an escapable metastable state, not a dead end. Counter-evidence that no-warmup runs do *not* escape quickly: v11d's Phase 2 (job 30625146) ran 11 full epochs flat at 0.0138 without escaping — past V12's escape point — though with a poor 2-epoch buggy warmup. So the precise claim is: the Phase-1 warmup *enables and accelerates* the escape; whether it is strictly necessary is an open question, deliberately left untested (cheap to settle by running the v11f recipe ~20 epochs, but academic — V12 works regardless). The concat (`nerf`) encoder is now training under a working recipe and can be evaluated for the blur hypothesis once Phase 2 converges.

### V12 final (cancelled at Phase 2 ep 75, 2026-05-24)

V12 ran ~71 h end-to-end and was cancelled at Phase 2 ep 75 after `phase2_epoch_75.pt` saved — val plateaued at ~0.00164 from ep 30 onward (the dataset ceiling V9 also hit; V9 val was 0.001585 at ep 60, V12 val 0.001651 at ep 70), so further epochs would only nudge train. Phase 2 train: ep 50 0.001037, ep 60 0.000947, ep 70 0.000877 (matched V9's ep-50 number), ep 75 **0.000849**; val 0.001652 at ep 75. Tracked V9 at a ~15–20 epoch lag (the 8-epoch unfreeze plateau + slightly slower descent). Checkpoints on disk: `phase1_epoch_{5,10,15,20}.pt` and `phase2_epoch_{5,10,...,75}.pt`. Next: render against V10b on the multi-scene eval suite to test the V11 NeRF-position blur hypothesis — the actual unanswered question, since loss can't distinguish blur from sharp detail.

### V12 vs V10b eval — the NeRF blur hypothesis is NOT vindicated (2026-05-24)

Rendered V12 `phase2_epoch_75.pt` and V10b `phase2_epoch_26.pt` on all four eval scenes (tomatoes, house, dragon, cartoon) at N ∈ {5k, 10k, 20k, 30k}, against the gsplat-full reference. `data_external/run_scene.py` was extended to bake column-group captions ("gsplat-full", "gsplat-pruned (N=X)", "<model label>") plus a per-image title into the 3-way overviews; `runs/eval_v11_scene.sh` gained a `LABEL` env var defaulting to "`<ckptdir suffix> ep<N>`".

PSNR vs full-gsplat, mean across the 14 orbit views, at N=10k:

| scene | V12 ep75 | V10b ep26 | Δ (V12−V10b) |
|---|---|---|---|
| tomatoes | 27.65 | 26.69 | **+0.96** |
| house | 30.61 | 31.28 | −0.67 |
| dragon | 31.32 | 31.74 | −0.42 |
| cartoon | 30.20 | 30.03 | +0.17 |

Across all N the pattern holds: V12 clearly wins on tomatoes (~+0.8–1.0 dB at every N), V10b slightly edges V12 on house and dragon (mostly within 0.2–1 dB), cartoon is essentially tied. Average across scenes: ~tied, slight edge to V10b.

**Qualitative read** (directly comparing the GF column of V12 and V10b strips on dragon n=10k): the two are *visually near-indistinguishable*. Same shape recovery, same soft-surface character, same level of detail loss vs the gsplat reference. The PSNR deltas of 0.2–1 dB are within the noise floor of perceptual judgment. The blur evident in V11/V12 renders (and present in V10b too) is structural to the RenderFormer-on-3DGS setup at N=5–30k, not specific to the input encoder.

**Bottom line on the V11/V12 investigation:**
- *Falsified*: the original V11 thesis that NeRF-encoding the position (and log-encoding the scale) would reduce blur vs the rope encoder. V12 produces visually equivalent output and is not consistently better in PSNR.
- *Durable*: the recipe fix in `training/train.py` — per-phase DDP construction with `find_unused_parameters=False`, freeze-before-wrap for Phase 1, re-wrap at the Phase-1→2 transition. End-to-end runs no longer crash and the joint-from-random-encoder plateau is escapable. This is reusable for any future encoder experiment.
- *Open*: whether a no-warmup run (`--skip_phase1`, random encoder) would *eventually* escape the 0.0138 plateau given 15–20 epochs — deliberately left untested, academic.

Eval renders (captioned 3-way strips, all N variants for both models on all 4 scenes) are under `data_external/{tomatoes,house,dragon,cartoon}/renders/overview_3way_n*_checkpoints_v{12,10b}_phase2_ep*.png`.

## V13 — testing the compression hypothesis (job 30652016, 2026-05-24, in flight)

### Hypothesis

If the V11/V12 plateau-shaped blur is structural to RenderFormer-on-3DGS at N=5k (per the V12 vs V10b qualitative read above), the next leverage point is the *training-time* token budget. V13's hypothesis: **5,000 Gaussians per scene is too compressed**, and training at N=20,000 (4× more) should let the model represent richer scene content and reduce the residual blur that V12 inherited from V10b. The encoder fix is now durable, so V13 swaps only the data-side variable.

### Pre-launch infrastructure (2026-05-24)

**VRAM/time frontier probes.** Three new sbatch scripts under `runs/` directly measured the (bs, N) frontier on a g4 card with the V12 (`pe_type=nerf`) encoder, AdamW + bf16 autocast, expandable_segments on:

- `tmp/probe_bs1_n_scaling.py` (sbatch `runs/probe_bs1_n.sh`) — bs=1, N ∈ {5k, 7.5k, …, 30k}. Linear: ~4 GB baseline + ~0.75 GB per 1,000 Gaussians per sample. Peak at N=30k bs=1 = **26.5 GB**.
- `tmp/feasibility_n30k.py` (sbatch `runs/feasibility_n30k.sh`) — 10-iter training feasibility at N=30k for bs ∈ {1, 2}. bs=1 N=30k: **28.96 GB**, 2.72 s/step (mean after 2 warmup). bs=2 N=30k: **OOM at 47 GB** on cyril-01 A6000.
- `tmp/probe_bs2_n_scaling.py` (sbatch `runs/probe_bs2_n.sh`) — direct (bs=2, N) sweep. Key cells: bs=2 N=20k = **39.36 GB, 3.28 s/step**; bs=2 N=22.5k = 43.11 GB; bs=2 N=25k = 46.85 GB (right at the 46 GB g4 ceiling).

Verdict: **bs=2 N=20k is the sweet spot** — fits 44 GB g4 with thin margin, comfortable on 46 GB, expected ~4 h 13 m per epoch under 4-GPU DDP. The V12-era bs=4 N=5k → V13 bs=2 N=20k swap **halves the effective batch (16 → 8)** for a 4× richer per-scene token budget.

**Data regeneration.** Re-pruning the Objaverse_Splats sources to N=20,000 via `data_v9.process_objaverse --target_n 20000` would have re-rendered all 14 GT views per scene — 4× wasted compute, since GT renders are full-scene rasterizations and target_n-independent. Added a `--skip_renders` flag (`data_v9/process_objaverse.py`) that gates the render block and relaxes the skip-existing check so re-pruning can keep the existing renders dir. `runs/regen_data_n20k.sh` ran train + val splits sequentially (1h 6m on epona-02, job 30651115) → `data_v9_n20k/h5s/` (2667 H5s) and `data_v9_n20k/h5s_val/` (183 H5s). Training points at `data_v9_n20k/h5s*` for input and `data_v9/renders*` for GT.

**Disk cleanup** (2026-05-24, ~45 GB reclaimed): deleted V11 probe/short-run checkpoints, intermediate V12 epoch saves (kept `phase1_epoch_20.pt` + `phase2_epoch_{25,50,75}.pt`), and 5 empty V11 stub dirs. V12 went 42 GB → 8.6 GB. V5/V6/V9/V10b untouched.

### V13 launch (job 30652016, 2026-05-24)

`runs/train_v13.sh` — V12 architecture (`--pe_type nerf`, the concat encoder under the working two-phase recipe) on the regenerated N=20k data, **time-boxed at 30 epochs** to fit the 168 h walltime: **10 Phase 1** (encoder warmup, lr 1e-3, the recipe-critical step) + **20 Phase 2** (joint fine-tune, lr 5e-5, where the actual quality signal lives), bs=2 per GPU × 4 GPUs DDP, `--save_interval 5`. LRs left at V12 defaults — effective batch dropped from 16 (V12) to 8 (V13) so gradients are ~√2 noisier; acceptable for a time-boxed exploratory run.

Phase 1 progress as of 2026-05-25 13:00:

| Phase 1 epoch | avg total loss | wall |
|---|---|---|
| 1 | 0.006334 | 4h 26m |
| 2 | 0.004366 (−31% vs ep 1) | 4h 25m |
| 3 | 0.003820 (−12.5%) | 4h 25m |

Clean diminishing-returns descent under cosine decay; no crashes, no plateau. The recipe fix is doing its job. Phase 1 expected to complete ~2026-05-27; Phase 2 → ~2026-06-01.

### V12 / V10b val-set baselines at infer-N ∈ {5k, 20k} — V13 comparator setup (2026-05-25)

Built the comparator V13 will eventually need: V12 ep75 and V10b ep26 evaluated on the **full 183-scene val set** at inference N ∈ {5k, 20k}, computing per-scene PSNR + LPIPS over 14 orbit views (2,562 views per checkpoint).

New tooling:
- `eval_val_full.py` — single-GPU bs=1 bf16 eval; takes `--checkpoint`, `--pe_type`, `--h5_dir`, `--gt_dir`, `--out_json`; renders each scene/view, tone-maps to LDR (AGX, matching GT pipeline), computes PSNR + LPIPS (AlexNet backbone), writes a per-scene + summary JSON. Defaults preserve `eval_val_set.py`'s rendering call shape but compute over the full val set with multi-metric output.
- `eval_compare.py` — ingests any number of result JSONs, prints a sorted comparison table.
- Four sbatch wrappers under `runs/eval_{v10b,v12}_n{5k,20k}.sh`, all `--killable --requeue` per the new non-training-job policy. Each ran ~12–15 min on a g4 card; all four killable jobs RUNNING within seconds despite tri_level holding the lab quota.

The 2×2:

| Model | infer-N=5k (training dist.) | infer-N=20k | Δ (20k − 5k) |
|---|---|---|---|
| **V10b** ep26 (rope, LPIPS-FT) | **25.597** dB / 0.0466 | 24.690 dB / 0.0499 | **−0.91 dB**, LPIPS +0.0033 (worse) |
| **V12** ep75 (nerf, log-HDR L1) | 25.482 dB / 0.0550 | 24.650 dB / 0.0568 | **−0.83 dB**, LPIPS +0.0018 (worse) |

Per-scene PSNR std ≈ 2.93 dB; LPIPS std ≈ 0.025–0.033.

**Key finding: more inference Gaussians *hurts* both models** by a consistent ~0.83–0.91 dB. The 4-scene non-monotonic hint from the V12-vs-V10b investigation (house 30.4→28.2 dB, dragon 30.6→29.6 dB) generalizes across all 183 val scenes. Both models were trained at N=5k and don't know how to use the extra 15k tokens — the surplus actively degrades them rather than helping. LPIPS also degrades, ruling out "PSNR-only artifact" framings.

**Implication for V13.** The compression hypothesis is more demanding than initially framed:

- **Floor V13 must clear:** 24.65 dB — V12 at untrained N=20k. Failing this means the V13 recipe didn't recover what V12 has for free.
- **Real bar (validates the hypothesis):** **25.60 dB** — V10b at its trained N=5k, the current production baseline. V13 must learn to *use* 20k tokens better than V10b uses 5k. This is the call.
- **Strong win:** ≥26.5 dB. Would justify the 4× training-time cost + data regen.

V10b roughly tied with V12 at both inference Ns also reinforces the V12 verdict: the encoder and recipe are not the bottleneck. If V13 fails the 25.60 bar, the next move is *not* more recipe tinkering — it's training resolution, dataset diversity, model capacity, or revisiting the structural blur conclusion.

Eval JSONs under `eval_results/v{10b,12}_ep{26,75}_n{5k,20k}_val.json`; reproducible end-to-end via `uv run --frozen python -m eval_val_full ...` or the four sbatch wrappers.

### V13 result: trains clean, beats production on PSNR, but the output still looks bad (2026-05-30)

V13 ran the full 30 epochs (10 Phase 1 + 20 Phase 2) end-to-end, no crashes, ~5d 17h on epona-02.

**Training behaved better than V12.** Phase 1 descended to train 0.003089 / val 0.003314 by ep 10 (half V12's epochs, below V12's ep-20 floor of 0.003720/0.003957). The Phase-1→2 transition re-wrapped cleanly (194.9M trainable). Crucially, **V13 did not sit in the 0.0138 attractor** the way V12 did for 8 epochs: Phase 2 ep 1 = 0.013841, ep 2 = 0.010921, ep 3 = 0.003611 — it broke out by ep 3 (V12 didn't escape until ep 9). The richer per-scene token budget got more out of the warmed encoder. Phase 2 val: ep 5 0.001913, ep 10 0.001495, ep 15 0.001304, ep 20 **0.001243** — below the ~0.00165 dataset-loss ceiling V9 (0.001585) and V12 (0.001652) both plateaued at. Terminal train 0.001020.

**Full 183-scene val PSNR/LPIPS — V13 wins PSNR at each model's best inference N:**

| Model | train N | infer N | PSNR mean | PSNR median | LPIPS mean |
|---|---|---|---|---|---|
| **V13 ep20** | 20k | 20k | **26.150** | **26.33** | 0.0500 |
| V10b ep26 (production, LPIPS-FT) | 5k | 5k | 25.597 | 25.49 | **0.0466** |
| V12 ep75 (L1) | 5k | 5k | 25.482 | 25.49 | 0.0550 |
| V10b ep26 | 5k | 20k | 24.690 | 24.77 | 0.0499 |
| V12 ep75 | 5k | 20k | 24.650 | 24.77 | 0.0568 |

**The compression hypothesis is confirmed on PSNR — with a sharp causal isolation.** The bottom two rows show that taking V10b/V12 (trained at N=5k) *to* inference N=20k *loses* ~0.9 dB — more Gaussians at render time hurts a model that wasn't trained for them. V13, *trained* at N=20k, reaches 26.15. So the +0.55 dB mean / +0.84 dB median over production is attributable specifically to **training at the higher density**, not to inference-time Gaussian count. And V13 achieves this undertrained (20 epochs vs V12's 75).

**But the verdict is heavily qualified — the renders still look bad (user's direct assessment, and correct).** Three caveats:
1. **PSNR is the wrong judge here.** A low-contrast, washed-out render sits near the per-pixel mean and is never boldly wrong, so MSE/PSNR rewards it. The +0.55 dB partly measures "V13 hedges less badly," not "V13 looks good."
2. **Colors are washed / desaturated.** Most visible on textured scenes: the barrel (scene_0030) loses its rich wood+metal banding to pale beige in all three models, V13 included; the truck (scene_0180) loses its blue tint to grey. V13 recovers *some* of this vs V10b/V12 but is still far from the GT's saturation.
3. **Fine detail still missing.** V13's gain is in *global shape + color fidelity*, not texture. Wheels, hardware, fruit-level detail remain unresolved — matching the PSNR-up / LPIPS-flat split (V13's 0.0500 LPIPS is worse than LPIPS-fine-tuned V10b's 0.0466, comparable-to-better than L1 V12's 0.0550).

So: density helped, measurably, but did **not** crack the core quality problem. The structural softness + desaturation persists. Next-step ideas under discussion (perceptual/color-aware losses, the deferred LPIPS fine-tune V13b, tone-map/exposure audit, higher training resolution, dataset color-distribution check). **Open question still unanswered: is the ceiling the RenderFormer-on-3DGS architecture, the L1-in-tone-mapped-space loss, or the data?**

> **CORRECTION (2026-05-30, supersedes the AGX numbers above).** All eval/render numbers in this V13-result section and in the "V12 / V10b val-set baselines" section above were produced with `eval_val_full.py` / `render_compare.py` defaulting to **AGX tone mapping** — a methodological bug. The `data_v9` GT renders are written with **no** tone map (`process_objaverse.py:12`: "no tonemap, since 3DGS source already trained on LDR images"), the training target is `log10(LDR+1)` of that no-tonemap LDR, and canonical `infer_gaussian.py` defaults to `tone_mapper='none'` (clip). Applying AGX — a desaturating filmic curve — only at eval mismatched the entire pipeline, *manufacturing* the "washed colors" and corrupting the metrics. See the corrected section below.

### V13 verdict CORRECTED — tone-map bug fixed, win is much larger (2026-05-30)

Diagnosed that the "washed colors" were largely an eval artifact: my `eval_val_full.py`/`render_compare.py` defaulted to AGX while GT + training + canonical inference all use **no tone map (clip)**. Fixed the default to `none` in both scripts (+ the 5 sbatch wrappers) and re-ran all five evals + the comparison grid on the full 183-scene val set.

**Corrected full-val numbers (`tone_mapper=none`, matching the trained pipeline):**

| Model | train N | infer N | PSNR mean | PSNR median | LPIPS mean |
|---|---|---|---|---|---|
| **V13 ep20** | 20k | 20k | **33.58** | 33.64 | 0.0349 |
| V12 ep75 (L1) | 5k | 5k | 30.95 | 30.72 | 0.0416 |
| V10b ep26 (production, LPIPS-FT) | 5k | 5k | 30.90 | 30.83 | **0.0283** |
| V10b ep26 | 5k | 20k | 30.09 | 29.92 | 0.0277 |
| V12 ep75 | 5k | 20k | 30.04 | 30.03 | 0.0404 |

What the fix changed:
- **Everything jumps ~+6 dB** (AGX was crushing the whole range): e.g. barrel scene_0030 V13 21.2→27.4 dB, truck scene_0180 V13 24.9→32.8 dB.
- **V13's lead over production grows from +0.55 → +2.67 dB mean** (33.58 vs 30.90). AGX's filmic compression had been squashing the inter-model gap; the real margin is large.
- **Color is faithful** in the corrected renders — the "awful washed colors" were predominantly the AGX artifact, *not* the model. (The barrel regains rich wood/blue banding; the truck regains its blue tint.)
- **The "more inference Gaussians hurts" finding is robust to the fix** — V10b 30.90@5k → 30.09@20k (−0.81), V12 30.95@5k → 30.04@20k (−0.91). The earlier ~0.9 dB conclusion stands.
- **LPIPS:** V13 (0.0349) beats its L1 twin V12 (0.0416) but trails LPIPS-fine-tuned V10b (0.0283). Smaller gap than the AGX eval implied; a V13b LPIPS fine-tune (mirroring V9→V10b) would likely surpass V10b.

**What the fix did NOT change — the detail/softness problem is real and model-side.** Tone mapping is a color/contrast curve; it adds no texture. V13 (and V10b/V12) still lose fine detail — truck wheels/panel lines, hardware, fruit-level texture. The PSNR-up / LPIPS-still-behind-V10b split is the quantitative signature of "shape+color good, high-frequency texture missing." This is the genuine open problem and the motivation for the contemplated fundamental overhaul.

Corrected artifacts overwrite the AGX ones in place: `eval_results/*.json` (all now `"tone_mapper": "none"`), `compare_renders/v13_ep20_vs_baselines/` (grid + strips). AGX-only diagnostic kept at `compare_renders/v13_ep20_TONEMAP_none/` was the isolation test.

Artifacts: `eval_results/v13_ep20_n20k_val.json`; comparison strips under `compare_renders/v13_ep{10,20}_vs_baselines/` (4-way GT|V10b|V12|V13, all at N=20k); checkpoints `checkpoints_v13/phase2_epoch_{5,10,15,20}.pt`.

### V13b: LPIPS fine-tune of V13 — recovers the perceptual axis (2026-05-31)

Mirrored the V9→V10b recipe: warm-start V13 ep20 weights (`--init_from`, fresh Phase 2, fresh cosine 5e-5), add LPIPS (`lpips_w 0.2`, `log_w 1.0`), 5 epochs, 8×g4 DDP, N=20k. The `--init_from` path was added to `training/train.py` (load weights only; no optimizer/scheduler/epoch carryover — distinct from `--resume`, which is crash recovery).

Per-epoch full-val (alex LPIPS, `tone_mapper=none`), all monotonic-improving until a gentle PSNR/LPIPS trade settles:

| ep | PSNR | LPIPS |
|---|---|---|
| 1 | 31.25 | 0.02624 |
| 2 | 31.91 | 0.02396 |
| 3 | 32.30 | 0.02290 |
| **4** | **32.83** | **0.02089** |

**V13b ep4 = 32.83 dB / 0.0209 LPIPS** — beats production V10b (30.90 / 0.0283) on **both** axes. The FT recovered the perceptual sharpness V13's L1-in-log loss had blurred (LPIPS 0.0349→0.0209, −40%) for only −0.75 dB PSNR vs V13. Kept as the best model for the next several days. (The run was preempted mid-ep5 on a killable node; ep5 would have been a near-zero-LR no-op, so ep4 is the keeper — `checkpoints_v13b/evaluated_run1/phase2_epoch_4.pt`, md5-verified.)

### V14: rope encoder + RoMa scene-rotation augmentation (plan, 2026-05-31)

Two moves from a close read of the RenderFormer paper:
1. **Back to RoPE-only position** (`pe_type=rope`). The paper (p.6) explicitly reports that NeRF-encoding triangle *positions* "is not stable, and it is prone to converge to a suboptimal local minimum" — exactly the V11 0.0138 plateau. Our entire rope lineage (V10b) was N=5k; **rope @ N=20k was the missing clean control.**
2. **RoMa rotation augmentation.** The model is rotation-variant (relative PE → translation invariance only); the paper augments with on-the-fly scene+camera rotations. For our data this is *exactly image-preserving* (gsplat `sh_degree=None` → constant per-Gaussian color, no world-fixed lighting), verified by re-rasterizing rotated scene+camera and matching the unrotated render to **108–119 dB** PSNR. Implemented in `training/dataset.py` (`augment_rotation` flag; rotate `means`, compose rotation onto quats via the matrix path, `c2w'=R4@c2w`; WXYZ↔XYZW reorder at the RoMa boundary). Also added `--keep_last_n` rolling checkpoint prune + atomic saves.

### V14 (rope @ N=20k, no-aug): nerf > rope — but the plateau was a clue (2026-06-02→03)

Trained rope @ N=20k, log-L1, seeded from an aug-warmed Phase-1 ep10. **Phase 2 sat at the 0.0138 plateau for ~10 epochs, then broke out at ep11** and descended cleanly to 0.001473 train / 0.001631 val. Renders confirmed: at the plateau the model outputs *sparse* (~1% lit pixels), not a dead all-zero collapse; at break-out the geometry/silhouette appears first (grey blobs in the right shape), then color/texture fills in.

Final full-val: **V14na = 31.49 dB / 0.0486 LPIPS.** vs V13 (nerf, no-aug) 33.58 / 0.0349 → **nerf beats rope by ~2.1 dB at N=20k.** So the paper's "NeRF-on-position unstable" caution did *not* translate to a problem for us; the richer NeRF position-lift actually helps organize 20k tokens. *Provisional verdict: keep nerf.* (This was later overturned — see below.) Process lesson banked: this recipe can plateau ~10 epochs before breaking out, so **don't judge a run at ep1–2.**

### V14 with augmentation: the breakthrough — aug is transformative (2026-06-04)

Re-ran rope @ N=20k **with augmentation**, extended to 30 Phase-2 epochs. **It broke out at ep1 (no plateau at all)** — the opposite of the no-aug run. The explanation reframes the whole plateau scare: both runs seeded from the *aug-warmed* Phase-1 ep10, so **aug Phase 2 = warmup-matched (instant break-out); no-aug Phase 2 = mismatched (10-epoch plateau while it un-learns rotation-invariance).** The plateau was a warmup/data-mismatch artifact, not a fundamental property.

Val descended faster than no-aug throughout (aug ep14 val 0.001356 already beat no-aug's *final* 0.001631). The run hit a cluster-contention burst and stalled at ep20 (3 preemptions, the last two 35 min apart, 0 free 8-GPU nodes), so ep20 was taken as the result.

**V14aug ep20 full-val = 33.70 dB / 0.0319 LPIPS — a new best-PSNR model:**
- vs V14na (rope, no-aug) 31.49: **+2.2 dB** — augmentation single-handedly erases the rope-vs-nerf gap and more.
- vs V13 (nerf, no-aug) 33.58: **V14aug WINS** (+0.12 dB, better LPIPS too).
- **So rope+aug > nerf-no-aug: augmentation is a *bigger* lever than the encoder choice**, overturning the provisional "nerf > rope" verdict (true only without aug). Render confirms V14aug sharpest, wins PSNR on every test scene; V14na visibly softest.

### V14aug-LPIPS: new best model on BOTH axes (2026-06-05)

LPIPS fine-tune of V14aug ep20 (`--init_from`, aug ON, `lpips_w 0.2`, 5-epoch fresh cosine 5e-5). Classic over-shoot-then-recover trajectory (full-val, alex):

| ep | PSNR | LPIPS |
|---|---|---|
| 1 | 31.37 | 0.02484 |
| 2 | 31.60 | 0.02338 |
| 3 | 32.56 | 0.02200 |
| 4 | 32.98 | 0.02044 |
| **5** | **33.57** | **0.01948** |

**V14aug-LPIPS ep5 = 33.57 dB / 0.01948 LPIPS — the new best model overall, beating V13b (32.83 / 0.0209) on PSNR by +0.74 dB *and* LPIPS by −0.0014.** It barely cost PSNR vs the base (33.70→33.57) while crushing LPIPS (0.0319→0.0195). Wins PSNR on every test scene in the comparison render. ep1's −2.3 dB PSNR dip is the normal LPIPS-FT over-shoot; it fully recovered by ep3–4 (lesson: don't judge an LPIPS FT at ep1).

Crucially, **ep5 was still climbing** (+0.60 dB PSNR, −0.001 LPIPS from ep4) — the 5-epoch cosine cut it off mid-ascent. A **10-epoch FT** (same recipe, `phase2_epochs 10`) is running now (`runs/train_v14auglp10.sh`) to extend the runway and likely push PSNR past the base's 33.70 with LPIPS lower still.

**Model leaderboard (full 183-scene val, `tone_mapper=none`):**

| Model | encoder | aug | LPIPS-FT | PSNR | LPIPS |
|---|---|---|---|---|---|
| **V14aug-LPIPS ep5** | rope | yes | yes | **33.57** | **0.01948** |
| V14aug ep20 | rope | yes | no | 33.70 | 0.0319 |
| V13 ep20 | nerf | no | no | 33.58 | 0.0349 |
| V13b ep4 | nerf | no | yes | 32.83 | 0.0209 |
| V14na ep20 | rope | no | no | 31.49 | 0.0486 |

**Headline takeaways:** (1) **augmentation is the dominant lever** — bigger than encoder choice, and it's free/image-preserving for our data; (2) **rope+aug+LPIPS-FT is the winning combo**; (3) the long Phase-2 plateau was a warmup-mismatch artifact, not a failure — patience + a matched warmup avoids it. Infra lessons banked along the way: `/dev/shm` dataset staging eliminates an NFS I/O bottleneck (3.0→1.2 s/step, ~3× speedup; the synthetic speed-probe had hidden it); killable jobs need `-c 32` not 64 (CPU was the scheduling blocker, not the 256 GB mem of which we use ~29); and `#!/bin/zsh` resume-aware launchers must build conditional args as zsh *arrays* (no scalar word-splitting).

Artifacts: best model `checkpoints_v14aug_lpips/phase2_epoch_5.pt`; evals `eval_results/v14{na,aug,auglp}_*_n20k_val.json`; renders `compare_renders/v14auglp_ep5_FINAL/` (GT|pruned-GT|V13b|V14aug-LP) and `compare_renders/v14aug_ep20/` (5-way). 10-epoch FT in progress → `checkpoints_v14auglp10/`.

### V14aug-LPIPS-10: the final best — more epochs paid off (2026-06-07)

The 5-epoch LPIPS FT was still climbing at ep5, so re-ran it with a **10-epoch** cosine (same recipe: `--init_from` V14aug ep20, rope+aug, `lpips_w 0.2`, fresh cosine 5e-5, `/dev/shm`). The longer schedule's gentler anneal gave a milder ep1 over-shoot (32.38 vs the 5-ep run's 31.37) and kept climbing through the back half:

| ep | PSNR | LPIPS |
|---|---|---|
| 5 | 33.11 | 0.02025 |
| 6 | 33.23 | 0.01998 |
| 7 | 33.40 | 0.01960 |
| 8 | 33.50 | 0.01922 |
| 9 | 33.75 | 0.01877 |
| **10** | **33.835** | **0.01853** |

**v14auglp10 ep10 = 33.835 dB / 0.01853 LPIPS — the definitive best model.** It beats the 5-epoch FT by +0.27 dB / −0.001, the old V13b by **+1.0 dB / −0.0024**, and even edges the V14aug *base* PSNR (33.70) while cutting LPIPS to a quarter of it (0.0319→0.0185) — a strict improvement on both axes over the base. So the "more epochs" call was right: the 5-epoch run was genuinely cut off mid-climb (+0.27 dB recovered), though the FT is now near its ceiling (~33.8 / ~0.0185, gains slowing to +0.08/epoch by ep10).

**Final model leaderboard (full 183-scene val, `tone_mapper=none`):**

| Model | encoder | aug | LPIPS-FT | PSNR | LPIPS |
|---|---|---|---|---|---|
| **v14auglp10 ep10 (BEST)** | rope | yes | 10-ep | **33.835** | **0.01853** |
| V14aug-LPIPS ep5 | rope | yes | 5-ep | 33.57 | 0.01948 |
| V14aug base | rope | yes | no | 33.70 | 0.0319 |
| V13 base | nerf | no | no | 33.58 | 0.0349 |
| V13b (prior best) | nerf | no | yes | 32.83 | 0.0209 |
| V14na | rope | no | no | 31.49 | 0.0486 |

**Campaign summary:** the winning recipe is **rope encoder + RoMa rotation augmentation + LPIPS fine-tune (~10 epochs)**. Augmentation was the dominant lever (bigger than the nerf-vs-rope encoder choice, and free/image-preserving for our data); the long Phase-2 plateau was a warmup/data-mismatch artifact, not a failure; and the LPIPS FT — given enough epochs — lifts both PSNR and LPIPS over the base. Best model: `checkpoints_v14auglp10/phase2_epoch_10.pt`.

Artifacts: `checkpoints_v14auglp10/phase2_epoch_10.pt`; evals `eval_results/v14auglp10_phase2_epoch_{5..10}_n20k_val.json`; render `compare_renders/v14auglp10_ep10_BEST/` (GT|pruned-GT|V13b|V14best).

## Side experiment: single-object overfit — capacity probe (2026-06-12, branch `exp/single-object-overfit`)

**Why.** Every version through V14 leaves residual blur — even our best can't match the
*pruned*-GT, and we keep debating whether the fix is **more data** or a **different
architecture**. Before iterating further (V15/16…), step off the main line and answer the
prior question directly: **is the model, at its current size/config and N=20k, even
*capable* of representing high-frequency texture/colour detail?** Overfit a single object
exclusively (augmentation OFF) long enough to memorise it, and read the ceiling off with
generalisation removed entirely.

**The decomposition.** Split the quality gap into two parts with opposite fixes:
```
real-GT --(pruning loss, fixed by N=20k)--> pruned-GT --(model loss)--> model output
```
- Overfit reaches **pruned-GT** → architecture *can* render the detail; bottleneck is
  data/N → **scale up** (validates the ~50× Objaverse_Splats headroom plan).
- Overfit plateaus **below pruned-GT** → architecture/decoder is the ceiling (most likely
  the DPT band-limited upsampling) → **change architecture**, not data.
- RF-base low but V14best higher → optimisation-limited, not capacity.
The decisive number is **model-vs-pruned-GT**: pruned-GT is what the model's *own* 20k input
can render, so failing it on one memorised object indicts the architecture.

**Matrix (4 killable single-GPU jobs)** — two objects × two inits:
- **objects**: `boxes` (Objaverse `scene_1441`, clean controlled, src-fit LPIPS 0.034) and
  `tomatoes` (the V8–V14 hero benchmark, real captured texture). Both texture/colour-rich and
  well-source-fit, so the detail genuinely lives in the 20k input.
- **inits**: `base` (RF transfer → Phase 1 encoder warmup → Phase 2 joint) and `v14best`
  (warm-start `checkpoints_v14auglp10/phase2_epoch_10.pt`; `--init_from` auto-skips Phase 1 —
  confirms a low base ceiling is capacity, not optimisation).
- **recipe**: `pe_type=rope`, aug **OFF**, in-train val **OFF**, bs=1, N=20k, Phase 1 50 ep /
  Phase 2 1500 ep (~21k steps over 14 samples), `save_interval 250`, `keep_last_n 6`.

**Infra.** Self-contained `experiments/overfit/` (`setup_data.sh` symlink-only,
`run_overfit.sh` parameterised by `OBJ`/`INIT`, `eval_overfit.py` for the three-way, README).
**Zero changes** to `train.py`/`render_compare.py`. Both objects' real-GT and pruned-GT are
already on disk (Objaverse full-scene renders; tomatoes `gsplat_full` + `gsplat_n20000`), so
prep = symlinks. tomatoes shares the exact orbit rig (radius 1.7, fov 45°, 14 views) as the
Objaverse data → `render_compare --pruned_gt` works unchanged.

**Status (2026-06-12, launched).** All 4 RUNNING on epona-01, healthy at ~24–27 s/epoch
(~11 h/run). `v14best` inits start at the dataset-loss floor (0.0008–0.0015); `base` inits are
warming the encoder. First evaluable checkpoint (ep250) ~2 h out. Eval pending →
model-vs-pruned-GT / model-vs-real-GT / pruned-vs-real for both objects under both inits.
Jobs: 30815617 `boxes_base`, 30815618 `boxes_v14best`, 30815619 `tomatoes_base`,
30815620 `tomatoes_v14best`.

### Results (2026-06-13, all converged at ep1500)

Three of four runs converged to near-zero train loss (`tomatoes_v14best` 0.000067,
`tomatoes_base` 0.000101, `boxes_v14best` 0.000136 — flat at LR-floor 5e-7 for the last
epochs, i.e. converged, *not* cut short). `boxes_base` fell into the joint-unfreeze
scene-blob plateau and sat flat at 0.0123 (LR-robust — a 4× rescue at 2e-4 also stuck);
retired. Final three-way (`eval_overfit.py`, alex-LPIPS, all 14 views):

| Run | model vs real-GT | model vs pruned-GT | pruned vs real-GT |
|---|---|---|---|
| `boxes_v14best` ep1500 | **51.81 dB** / 0.0009 | 37.53 / 0.0055 | 37.58 / 0.0057 |
| `tomatoes_base` ep1500 | **49.13 dB** / 0.0008 | 29.65 / 0.0221 | 29.61 / 0.0219 |
| `tomatoes_v14best` ep1500 | **53.04 dB** / 0.0003 | 29.62 / 0.0221 | 29.61 / 0.0219 |
| `boxes_base` ep250 (stuck) | 17.77 / 0.2027 | 18.32 / 0.1973 | 37.58 / 0.0057 |

**Verdict — the architecture is NOT the wall.** Overfit reaches **49–53 dB / LPIPS
0.0003–0.0009 vs real-GT** (visually pixel-perfect; model column sharper than pruned-GT).
The **DPT-band-limit hypothesis is falsified**: the decoder can output arbitrarily sharp
high-frequency detail when fit. So the residual blur in every general model (V9–V14) is a
**data/generalisation** problem, not an architectural ceiling → **scale the data** is the
right direction.

**Reframing — "model vs pruned-GT = ceiling" was the wrong lens.** Trained on real-GT, the
model memorises it to ~50 dB and *transcends* the pruned-GT (model-vs-pruned ≈ pruned-vs-real
because model ≈ real-GT). The decisive number is **model-vs-real-GT**, not model-vs-pruned.
(`eval_overfit.py`'s "CAPACITY CEILING" label on the pruned column is misleading and should
be relabelled.)

**Honest caveat.** Overfit = *memorisation* of 14 (object, view) targets, so it proves
**output capacity** (decoder can produce the detail), not that the model can *render* that
detail *from the 20k input* in a generalising way. It rules out "architecture is fundamentally
incapable"; it does not by itself guarantee data-scaling closes the generalisation gap.

**Second finding — N=20k is itself a bottleneck for high-detail real scans.** `pruned vs
real-GT` = the cap a perfect renderer of the 20k input could reach: **tomatoes 29.6 dB**
(real captured texture — 20k Gaussians can't hold it) vs **boxes 37.6 dB** (cleaner synthetic
object). So for detailed objects, raising N matters independently of the model. Boxes (simpler)
loses far less to pruning.

**Recipe finding.** The bs=1 two-phase recipe is **object-dependently fragile** at the joint
unfreeze: `tomatoes_base` escaped the plateau instantly, `boxes_base` never did, and LR was
not the lever (2e-4 sat at the same 0.0123 as 5e-5). `boxes_v14best` (warm-started) overfits
boxes fine, so boxes is overfittable — the plateau is an optimisation artifact, not capacity.

Artifacts: `experiments/overfit/eval/*_metrics.json` + `*_strip.png` (GT|pruned|model),
hi-res `tomatoes_v14best_hires.png`. Branch `exp/single-object-overfit`.

### Novel-view generalisation probe (2026-06-13) — resolves the memorisation caveat

Rendered the converged tomatoes overfits at **held-out poses** (azimuths halfway between the 14
training views, in-between elevation — `experiments/overfit/novel_view.py`), vs gsplat-full
(real-GT) and gsplat-pruned (20k) at the same pose. Training-view controls reproduce the ~50 dB
memorisation number, confirming the setup.

| tomatoes | TRAIN views (seen) | NOVEL views (held out) | pruned-GT ceiling |
|---|---|---|---|
| `v14best` | 52.5 dB | **30.5 dB / LPIPS 0.012** | 29.7 dB |
| `base` | 48.6 dB | 24.2 dB / LPIPS 0.039 | 29.7 dB |

**The overfit was NOT pure memorisation.** Novel views are coherent and correct (not collapsed) —
the model learned a *renderable 3D representation* from the 20k input. And **at novel views the
model saturates the N=20k ceiling** (`v14best` 30.5 dB ≈ pruned-GT 29.7 dB): when it can't
memorise, it renders the 20k Gaussians about as well as gsplat does. So **the test-time bottleneck
for a known object is the pruning (N), not the model** → raise N for detailed objects. The
`v14best` vs `base` gap (30.5 vs 24.2) shows full-dataset pretraining priors are what enable
view-generalisation. *Still untested:* cross-**object** generalisation (rendering an unseen object)
— that is what the data scale-up addresses, and the clean next experiment.

**Infra lesson (banked).** `sbatch --wrap` runs under `/bin/sh`, where `source`/`module` don't
exist, so `module load cuda` silently fails → gsplat's JIT CUDA backend can't load → `_C=None`
("'NoneType' has no attribute 'CameraModelType'"). GPU jobs needing the CUDA toolkit (gsplat,
nvcc) MUST use a real `#!/bin/zsh` sbatch script that sources `huji-lmod.sh`, never `--wrap`. The
node-exclusion chase was a red herring. Artifacts: `experiments/overfit/eval/*_novelview_*`,
script `experiments/overfit/{novel_view.py,run_novel.sh}`.

## 5x data scale-up + pruning recovery (2026-06-13/14, branch `data/v10-scaleup`)

### `data_v10/` — 5x additive dataset (full splats, rotation-fixed, un-pruned)
Built a 5x additive dataset: **14,307 objects** (13,405 train + 902 val, 0 failures) with full
**un-pruned** ~50k-gaussian splats (SH-stripped, ~2.8 MB each) + GT renders (14 views, 512).
- **Rotation bug fixed**: Objaverse_Splats is Z-up, our orbit Y-up → objects rendered on their
  side. Fix = **−90° about X (Rx-90)** on means+quats in `data_v10/process_full.py`; verified
  upright across chair/robot/tank/bike/soldier/etc.
- **Additive**: kept data_v9's 3,000+200 selection verbatim, added 12,800 disjoint new objects
  (same PSNR≥32/LPIPS≤0.06/num_GS=50000 filter; 86,727 unused pass) → 15k train + 1k val entries.
- **Un-pruned by design** so we can explore a better N=20k pruning. ~46 GB fulls + 28 GB renders.
- Ran as 6 killable chunk-disjoint shards (~1.3 h). Scripts: `process_full.py`, `build_additive.py`,
  `make_shards.py`, `run_process.sh`. All N uniform = 50,000.

### Pruning: we were skipping LightGaussian's recovery step (2026-06-14)
Studied LightGaussian (arXiv 2311.17245 + `prune_finetune.py`). Their method = **score → prune
~60% → RECOVER** (fine-tune survivors, L1+SSIM, densification off, ~5k iters). **Our pipeline
does score+prune and STOPS** — i.e. their "pruning only" baseline, which their own ablation shows
is −1.36 dB; the recovery restores it to +0.17 over baseline. Our score
(`Σ_views opacity·proj_radii² · max_scale^γ`) is a fine LightGaussian variant; the gap is the
**missing recovery**. **Doubly important for us**: the novel-view probe showed GaussianFormer
*saturates the pruned-GT ceiling*, so lifting pruned-GT via recovery lifts the model's achievable
ceiling **at the same N=20k, no architecture/data/inference cost.**

`data_v10/prune_recovery.py`: prune 50k→20k → recover (gsplat Adam on means/log-scale/quat/
opacity-logit/color-logit, L1+SSIM vs full-splat renders at 64 views, no densification, exp-LR,
~1500 iters), then eval naive-topk vs recovered vs full on **canonical (14) + strict held-out
(16, unseen elevation)** views.

**Initial 3 texture-rich objects (1500 iters):** gains far bigger than LightGaussian's (theirs
starts from already-good gaussians; our naive top-k leaves *holes* the recovery fills):
| scene | canon naive→rec | held-out naive→rec |
|---|---|---|
| boxes (1441) | 36.45 → 51.35 (**+14.9**) | 35.77 → 47.33 (+11.6) |
| cannon (1196) | 37.11 → 51.28 (**+14.2**) | 36.77 → 50.46 (+13.7) |
| helmet (2426) | 39.45 → 55.76 (**+16.3**) | 39.01 → 53.06 (+14.1) |
Held-out gains confirm it's **not** recovery-view overfitting. Visually verified (full ≈ recovered,
no artifacts). These 3 are geometrically simple → a **30-object diverse gate** (incl. detailed/
high-freq) is running (6 killable shards) to get the true distribution before committing the full
14k recovery. Scripts: `prune_recovery.py`, `run_recovery.sh`. **No pruning written to disk yet.**

### Recovery gate PASSED — 30 diverse objects (2026-06-14)
Across 30 diverse objects (incl. detailed/high-freq), recovery gain is large and consistent:
- **held-out views: naive 35.0 → recovered 49.5 dB (mean +14.6, median +14.5, min +9.2, max +18.5)**
- canonical: mean +15.8 dB. Worst (hardest) objects still +9.7..+14.8, reaching 43–45 dB.
Decisive go for the **full-14k recovery** as data-prep. (Bigger than LightGaussian's +1.5 because
our naive top-k leaves holes the 50k didn't have; the over-parameterised 50k re-fits to ~20k well.)

### V14best on data_v10 — baseline before retrain (`model_on_v10.py`)
Ran V14best (`checkpoints_v14auglp10/phase2_epoch_10.pt`) on naive-pruned 20k v10 objects incl.
**unseen** ones (scene_idx>3000 — never trained). Rotation-fix + RoMA robustness CONFIRMED (clean
on upright). Mean **33.3 dB** (unseen 32.6 ≈ seen 34.7 → modest cross-object generalisation). KEY:
on unseen objects the model sits **~3 dB BELOW even its own naive pruned-GT** and **caps ~33–36 dB
regardless of input quality** (where pruned-GT~39, model falls ~6 dB short) — the generalisation/
blur gap (unlike the overfit probe which *saturated* pruned-GT on a memorised object).

**Two quantified gaps → two validated levers:**
```
current model (unseen):  ~33 dB
  gap1 model blur:       ~3 dB below naive pruned-GT  -> MORE DATA (5x) + training
naive pruned-GT:         ~35 dB
  gap2 pruning waste:    +14.6 dB                     -> RECOVERY (validated)
recovered pruned-GT:     ~49.5 dB
```

### NEXT (pending user go-ahead on "step A"):
**(A) Full-14k prune-and-recovery** — run `prune_recovery.py --save_h5_dir` over all 14,307 objects
(parallel killable shards like the data gen; ~per-object gsplat fine-tune, tune iters down from
1500 if the knee allows). Produces `data_v10/h5s_20k_rec/` (recovered 20k H5s) for training.
**(B) Retrain** on 5x data + recovery-pruned 20k + **256→512 curriculum**, then re-measure on these
same unseen objects (current baseline 33.3 dB) to see if closing both gaps lifts out of the low-30s.
Branch `data/v10-scaleup`. Eval artifacts (gitignored): `data_v10/{recovery_eval,model_eval}/`.

---

## Session: 2026-06-15

### Repo migration tomhope → sagieb (done)
Migrated the live workspace from `/cs/labs/tomhope/shahaf_levy/gaussianformer` to
`/cs/labs/sagieb/shahaf_levy/gaussianformer` (account `-A sagieb`). Tom's copy kept as a cold
backup (nothing deleted). Data verified bit-identical (checkpoints, H5s, code, PROGRESS), `.venv`
rebuilt via `uv sync --frozen` (torch 2.9.1+cu128 / gsplat 1.5.3 import-clean), memory copied to
the sagieb project slug. All training scripts gained `#SBATCH --account=sagieb`.

### Step A — full-14k prune-and-recovery (DONE)
Ran `prune_recovery.py --save_*` over the whole dataset → recovered 20k H5s on disk:
**`data_v10/h5s_20k_rec/` = 13,405 train + `h5s_20k_rec_val/` = 902 val = 14,307 objects.**
Resume-safe shards (skip-existing) survived preemption/requeue; OOM-per-object fix (free GPU mem
between objects) landed mid-run. These recovered H5s are the training input for V15 (input ceiling
now ~49.5 dB held-out vs ~35 naive — gap2 closed in the data).

### Step B — V15 256→512→LPIPS curriculum LAUNCHED (chained)
Three SLURM jobs submitted as an `afterok` dependency chain (8×g4, `--killable --requeue`,
`-c 32`, `--mem 200GB`; /dev/shm staging of the 66 GB recovered H5s+renders):

| Stage | Script | Res | Phase budget | LPIPS w | Job | Trigger |
|---|---|---|---|---|---|---|
| 1 bulk | `train_v15_256.sh` | 256² | P1 5ep @1e-3, P2 10ep @5e-5 | 0.0 | 30837514 | — (running) |
| 2 refine | `train_v15_512.sh` | 512² | P2 3ep @5e-5 (`--init_from` 256) | 0.0 | 30837515 | afterok:…514 |
| 3 LPIPS FT | `train_v15_lpips.sh` *(new)* | 512² | P2 10ep @5e-5 (`--init_from` 512) | **0.2** | 30837516 | afterok:…515 |

Stage 3 mirrors the v14auglp10 recipe (rope + RoMa aug + 10-epoch LPIPS FT @0.2 — the stage that
produced every prior best; v14auglp10 ep10 = 33.835 dB / 0.0185, still climbing). `--init_from`
loads weights-only and skips Phase 1 → fresh Phase 2 cosine from 5e-5. `keep_last_n 10` so every
LPIPS epoch is an eval candidate. Chain is preemption-safe: requeue keeps dependents waiting; a
terminal crash leaves downstream blocked (`DependencyNeverSatisfied`) rather than seeding from a
bad checkpoint.

**bs/LR NOT adjusted per resolution (deliberate).** All three stages use `batch_size 1` (×8 GPU =
eff batch 8 == V14) and the same LRs (P1 1e-3 / P2 5e-5), to reuse V14's tuned LR pair and keep the
curriculum a pure resolution change. Keeping LR fixed across resolutions is correct at fixed
effective batch (LR tracks batch size, not resolution). The one unexploited lever: at 256² there's
~4× activation-memory headroom, so a larger bs there could speed the bulk stage — but that would
require LR re-tuning, so it was left for a later pass.

**Eval is run independently** (`model_on_v10.py`, baseline 33.3 dB on unseen v10 objects) against
`checkpoints_v15_lpips/phase2_epoch_*.pt` once the chain produces them — not part of the launch.

---

## Session: 2026-06-17 .. 2026-06-21 — V15 diagnostic, V16 retrain, and the blur diagnosis

### V15 = under-trained diagnostic
The 5×-data + recovered-20k + 256→512→LPIPS chain ran end-to-end (V15) but came out soft:
phase-2 loss still descending, LPIPS ~2× V14best. Cause: epoch budget too small for 4.7× data
(~9× less per-object exposure than V14) + a 3-epoch 512 stage. Kept as reference.

### Infra wins (reused by V16)
- **tar-staging**: the per-job `cp -r` of ~214k tiny render files sat at ~1.8 MB/s over NFS
  (~4.5 h/job). `data_v10/build_tars.sh` packs them into 27 shard-tars once (~12 min, parallel),
  each job extracts a few big tars → staging dropped to ~5 min.
- **view-subsampling** (`--views_per_epoch 4`): the dataset emits one sample per (scene,view), so
  a full-views epoch is 187,670 samples (~7.5 h @256). K=4 reslices the epoch ~14/K with finer
  checkpoints; the **batch probe proved training is compute-bound on the 20k attention** (256 bs1
  6.2 ≈ bs2 6.5 samples/s; bs>1 no gain, 256 maxes bs2 / 512 bs1 on 46 GB), so bigger batch is no
  lever. Multi-node (12 GPU) failed (NCCL inter-node); khan 96 GB nodes too contended to pin.

### V16 = bigger-budget retrain (256 P2 20, 512 P2 12, LPIPS 12), bs=1 + K=4
**V16's phase-1 silently failed to train** (val 0.0135 vs V15's 0.0038 — pinned with the
standalone `data_v10/diag_val.py`), starving phase-2 (frozen at 0.0126 for 16 epochs). Earlier
DDP/resume theories were wrong. **Fix**: seed phase-2 via `--init_from checkpoints_v15_256/
phase1_epoch_5.pt` (V15's known-good warmup). Then it converged cleanly: 256→0.00103, 512→0.000956
(both beat V15). Lesson: verify phase-1 val converges before trusting phase-2.

### THE BLUR DIAGNOSIS — whole-image PSNR is a lying metric
Shahaf flagged the renders look soft despite "+2.6 dB over V14best". Confirmed: objects are on
black bg and are only **2–5 % of pixels** (skull 1.7 %), so whole-image PSNR is ~95–98 % "match the
black" → inflated, blind to object sharpness. Added **`model_on_v10.py --crop_fg`** (crop to GT
object bbox, metric there, zoom the crop). Object-only, V16-512-L1 is ~27.7 dB / LPIPS 0.31 on the
skull vs the recovered-input ceiling ~40 dB → a real ~13 dB model-blur gap the metric hid. **Eval
everything `--crop_fg` from now on.**

### CAPACITY PROBE v2 — the blur is a GENERALISATION gap, not architecture or loss
Overfit ONE detailed object (skull 9869) from v16_512, L1 vs high-LPIPS
(`experiments/overfit/run_overfit_lpips.sh`). **Both arms reproduce the fine engravings**
(converged: L1 35.3 dB/0.062, hi-LPIPS 33.9 dB/**0.022**; general model 0.311). → the architecture
CAN render fine detail at N=20k; the general model smears it only because it can't *generalise* the
sharp mapping. **High-LPIPS is a real lever** (0.31→0.022 memorised).

### Acting on it — LPIPS-weight sweep on the full model (running)
The planned LPIPS@0.2 stage FAILED at ep3 (node fault, no requeue). Pivoted to a perceptual-weight
sweep, all `--init_from` v16_512, eval `--crop_fg`:
- **hi** `train_v16_lpips_hi.sh` — log_w 0.5 / **lpips_w 1.0** (job 30891156). ep2 on unseen skull:
  **LPIPS 0.31→0.159** (transfers to generalisation!) but grainy (ep1-2 over-shoot); statue (low
  detail) slightly worse — 1.0 may be too strong there.
- **mid** `train_v16_lpips_mid.sh` — log_w 0.5 / **lpips_w 0.5** (job 30894280), same log anchor so
  only lpips_w differs. Tests detail-gain-vs-graininess sweet spot.
Re-eval both ~ep5–6 (past over-shoot) `--crop_fg` to pick the winner. Deeper fix for the
generalisation gap remains more/better data.

---

## Session: 2026-07-12 — LPIPS sweep verdict, V14-vs-V16 apples-to-apples, and "we under-trained"

Picking up after a ~3-week gap (the Claude SLURM node died mid-sweep). Both sweep arms had
actually finished 12 epochs and been eval'd on 06-23; nobody had read the results.

### LPIPS-weight sweep — settled: w=0.5, and 1.0 buys nothing
Foreground-cropped (`--crop_fg`, `--input_mode recovered`), 3 unseen objects (10916 statue /
9869 skull / 10751 honeypot):

| model | statue | skull | honeypot | mean PSNR |
|---|---|---|---|---|
| V16-512 (L1 only) | 28.5 / 0.035 | 27.5 / **0.311** | 27.0 / 0.120 | 27.7 dB |
| LPIPS **hi** (w=1.0) | 27.0 / 0.031 | 26.2 / 0.129 | 26.8 / 0.070 | 26.7 dB |
| LPIPS **mid** (w=0.5) | 27.3 / **0.030** | 26.5 / **0.128** | 27.0 / **0.067** | 26.9 dB |

**mid ≥ hi on both axes on every scene** → cranking lpips_w past 0.5 is pure PSNR cost, no
perceptual gain. LPIPS FT halves LPIPS on the hard objects for ~1 dB PSNR — a real, cheap win.
**`checkpoints_v16_lpips_mid/phase2_epoch_12.pt` is the new best model.**
Caveat that keeps it honest: on the skull the FT renders *invented* carved texture, not the
actual engraving. Perceptual loss bought texture, not fidelity.

### V14best vs V16, apples-to-apples in object space (the missing number)
Every V14-era metric was whole-image, i.e. from the lying-metric era. Ran V14best
(`checkpoints_v14auglp10/phase2_epoch_10.pt`) `--crop_fg` on the same 3 scenes, both input modes:

| model (input) | statue | skull | honeypot | mean |
|---|---|---|---|---|
| V14best (naive — its own training distribution) | 22.6 / 0.061 | 26.5 / 0.183 | 25.5 / 0.091 | **24.88 dB** |
| V14best (recovered input) | 23.2 / 0.058 | 25.2 / 0.183 | 25.3 / 0.093 | 24.55 dB |
| **V16 + LPIPS mid** | 27.3 / 0.030 | 26.5 / 0.128 | 27.0 / 0.067 | **26.94 dB** |

**The v10 scale-up paid off: +2.1 dB and ~½ the LPIPS on every object.** Renders confirm —
V14best's skull is nearly featureless and it drops the honeypot's small props entirely.

Two structural findings:
- **The naive-pruned ceiling is meaningless.** V14best's naive input ceiling is only 25–29 dB
  (vs 40–43 recovered). V14best's skull *scores above its own ceiling* (26.5 > 25.0) — not
  because it's good, but because the naive input is so degraded that a smooth blob lands nearer
  the true GT than the input render does. Discount every naive-ceiling comparison.
- **Recovered input does NOT help a model that wasn't trained on it** (V14best: 24.55 rec vs
  24.88 naive — slightly *worse*). The gain is from the retrain, not from nicer eval-time
  Gaussians. You must train on recovered data to benefit from it.

### THE BIG ONE — V16 was never converged; we under-trained by a wide margin
Every stage's val loss was **still descending monotonically at its final epoch**, no plateau,
no overfitting signal (val tracked down throughout):
- 256 P2: ep19 0.001039 → ep20 0.001034 (cut at the budget, not at convergence)
- 512 P2: ep10 0.000979 → ep11 0.000962 → ep12 **0.000956**
- LPIPS mid: lpips term ep11 0.017998 → ep12 **0.017893**

**CORRECTION (same session, after launching V17): hardware, not budget, sets the wall-clock.**
I first read V16's `sacct` times (256: 7h35/20ep = 23 min/ep · 512: 4h43/12ep · LPIPS: 2h25/12ep)
and concluded "the whole V16 chain was ~15 h, we rationed a budget we didn't need to ration."
**That was wrong — the entire V16 chain ran on `khan-01` (RTX Pro 6000, 128 cores).** V17 landed on
`firefoot-11` (L40S, 64 cores) and measures **80 min/epoch, 3.5× slower**, on identical config
(1.39 steps/s/rank; 11.1 samples/s over 8 GPUs; epoch = 53,620 samples = 6,702 steps/rank). So the
original "multi-day, compute-bound" read was closer to right than my correction of it. **Never quote
a min/epoch without naming the node.** khan is normally 8/8 allocated by non-preemptible jobs, so a
killable job cannot bump it — getting khan is luck, not a plan.

Per-object exposure math: the whole V16 chain = ~12.5 passes over the 187,670-sample set. V14
got ~30 passes over its 3k objects. So V16 has seen each object-view **less than half** as often
as V14 did — on 4.5× more objects. **The "generalisation gap" and "under-trained" may be the
same problem: the model hasn't finished digesting the data it already has.** This reframes the
"need more data" conclusion — before buying more data, spend the training we already can afford.

### Next
V17 = V16 recipe, much longer, with LPIPS(w=0.5) folded into the 512 stage from the start rather
than bolted on as a 12-epoch tail FT. Gate on the LPIPS val term + periodic `--crop_fg` renders,
NOT on log-L1 (V4/V5/V6 precedent: log-L1 improvements do not reliably translate to perceptual
gain).

## 2026-07-23 — V17 finished; NEW BEST MODEL; skull-fidelity ceiling; scale-up prep + paper renders

> **[CORRECTION 2026-08-06]** The "unseen object" evaluations in this and earlier V16/V17 sections were measured on objects that were in the models' TRAINING set (`dataset.py` globs the whole h5 dir). True held-out mean for V17 is **30.29 dB**. The "generalisation gap" framing is also superseded: the blur is UNDER-fitting (capacity-limited memorisation). See the 2026-08-02..05 and 2026-08-05..07 sections.

### V17 completed — `checkpoints_v17_512lp/phase2_epoch_36.pt` is the new best model
Two-stage chain, all 8×g4 killable, seeded from V15's known-good phase1:
- **Stage A** (`train_v17_256.sh`) — 60 phase-2 epochs @256, pure log-L1. Converged flat at
  val log-L1 **0.000765** (~25% below V16's 0.001034 floor).
- **Stage B** (`train_v17_512lp.sh`) — 36 epochs @512 with LPIPS(0.5)+log(0.5) from epoch 1.
  LPIPS val term descended monotonically the whole way: 0.02061 (ep1) → 0.01783 (ep5) →
  0.01701 (ep10) → 0.01559 (ep17) → 0.01440 (ep27) → **0.01397 (ep36)**, cosine LR to 5e-7,
  converged. Final is **~22% below V16's 0.017893**.
- Wall-clock: 65.9 min/epoch on khan-01. Ran clean for 29 epochs, then got preempted twice
  (first preemptions of the run) and sat PENDING ~1.5 days on a fully-saturated g4 partition
  (all 22 g4 nodes 8/8; the one idle node was DRAINed). Requeue resumed from per-epoch ckpts,
  **zero work lost**. Confirms the killable trade-off: cheap when the cluster is free, unbounded
  wait when it's not.

### Object-space verdict (`--crop_fg --input_mode recovered`, scenes_profcompare, 3 unseen)
| model | statue | skull | honeypot | **mean PSNR** |
|---|---|---|---|---|
| V14best | 22.6 / 0.061 | 26.5 / 0.183 | 25.5 / 0.091 | 24.88 |
| V16+LPIPS-mid ep12 | 27.3 / 0.030 | 26.5 / 0.128 | 27.0 / 0.067 | 26.94 |
| V17 ep6 (over-shoot) | 25.0 / 0.038 | 25.7 / 0.129 | 25.5 / 0.068 | 25.40 |
| V17 ep17 | 28.3 / 0.025 | 25.6 / 0.126 | 27.6 / 0.052 | 27.17 |
| **V17 ep36** | **30.5 / 0.020** | **26.5 / 0.110** | **28.5 / 0.044** | **28.49** |

**V17 ep36 wins on every axis: +1.55 dB mean PSNR over V16, better LPIPS on all three objects.**
The ep6 reading (1.5 dB *behind*) was the documented fresh-cosine LPIPS over-shoot window — do
not judge an LPIPS run before ~ep15. "Train longer + LPIPS-throughout" was a real win.

### THE CEILING — longer training does NOT fix high-frequency fidelity
The skull LPIPS moved 0.128 → 0.110 (−14%) and the render is visibly sharper/higher-contrast,
**but it still renders *invented* swirly carving, not the true engraving** — same failure mode as
V16, just prettier. This is exactly what the June capacity probe predicted: overfitting the skull
alone reproduces the real engraving at 0.022 LPIPS, so the architecture CAN render it; the general
model can't *generalise* the sharp mapping. The two levers we picked (perceptual loss, more epochs)
are now both spent on this: LPIPS did 0.311→0.128, all of V17's extra epochs did 0.128→0.110.
**Faithful high-freq detail is the open problem, and it is NOT an epochs problem.** Next lever is
data curation (does the training set even contain enough high-freq surface detail?), not more training.
The input ceiling holds the engraving fine at ~40 dB, so the information is in the Gaussians — the
model is losing it, not the data.

### Scale-up storage check (Sagie volume `/cs/labs/sagieb`)
462 G free (2.0 T total, 1.6 T used, 78%; group quota 1587/2048 G — agrees). Current footprint:
`data_v10` = 117 G / ~13.4k scenes (incl. 25 G redundant `tars` staging) → ~8.7 MB/object all-in;
v17 ckpts ~39 G; **~113 G of stale v15/v16 experiment ckpts** (v16_lpips_hi/mid are 32 G each).
Scale-up headroom: **2× (~+117 G data +40 G ckpt) fits comfortably today**; 3× fits but tight;
4× needs cleanup first. Reclaiming the stale v15/v16 ckpts (~113 G) → ~575 G, makes 3–4× easy.
**Storage is not the bottleneck — data-generation compute is** (re-running prune+recovery, the
~1500-iter gsplat FT/object that bought +14.6 dB, as a long swarm on a saturated g4 cluster).

### Paper renders BEFORE pruning v15/v16 (`data_v10/showcase_versions.py`, `run_showcase.sh`)
To preserve the cross-generation visual comparison before reclaiming the v15/v16 ckpts, generating
raw per-render PNGs over **200 random unseen objects** (idx>3000, pool=10,737): each rendered by
V14best (naive input) / V16-LPIPS-mid (recovered) / V17-ep36 (recovered) at 4 views → **~2,400
renders** named `data_v10/showcase/s{scene:05d}_v{view:02d}_{model}.png`. Clean, unlabeled,
full-frame — **no grids/strips** (those are composed later). Each model fed its NATIVE input
distribution (feeding all the same input would mis-state the leap). Per-shard PSNR/LPIPS manifests
written for later selection. 8 killable 1-GPU shards. Once complete, **pruning v15/v16 is safe** —
their output is captured permanently.
- Gotcha logged: `seq -w 0 7` pads to width 1 (max is single-digit) → looked for `shard_0.json`
  not `shard_00.json`; first 8-job launch no-op'd on FileNotFound. Relaunched with explicit `00..07`.

## 2026-07-24..26 — showcase renders, v15/v16 cleanup, 2× DATA SCALE-UP, V18 launch (no tars)

### Paper showcase renders done, then reclaimed ~110 GB
Rendered 200 random UNSEEN objects × {V14best(naive), V16+LPIPS-mid(recovered), V17-ep36(recovered)}
at 4 views → **~2,400 clean per-render PNGs** in `data_v10/showcase/` (`showcase_versions.py`,
`run_showcase.sh`). Each model fed its NATIVE input; raw/unlabeled/full-frame so any grid can be
composed later. With the cross-generation comparison captured permanently, deleted the stale
v15/v16 checkpoints — **~110 GB reclaimed** (kept `checkpoints_v15_256/phase1_epoch_5.pt`, the
reusable phase-1 seed, + its HF export). Free space 462 → 556 GB.

### 2× DATA SCALE-UP — data_v10 doubled in place (train 13,405→26,820, val 902→1,806)
Three stages, all resume-safe:
1. **Select** (`build_expand_2x.py`, SEED=2): excluded all 16k current uids, drew +15k train /
   +1k val from a 73,927-object pool passing PSNR≥32/LPIPS≤0.06/num_GS=50k. New scene_idx
   continues after the max (train 15000–29999, val 1000–1999). 24 fresh chunks (~79 GB transient).
2. **Process** (`run_process.sh` → `process_full.py`): downloaded chunks, normalized (Rx-90),
   14-view 512 GT renders + full ~50k-splat h5. **13,415 new train + 904 val** survived (rest
   low-opacity/degenerate). Total: **26,820 train / 1,806 val**.
3. **Recover** (`run_recovery.sh` → `prune_recovery.py`): prune 50k→20k + gsplat FT ~1500 iters,
   **~9.6 s/object** (much faster than the 60 s feared → ~36 GPU-h, not 220). All 13,415 new train
   + 904 val recovered into `h5s_20k_rec` / `h5s_20k_rec_val`. **0 missing.**

### THE DATA-GEN GOTCHA (cost several hours) — gsplat co-tenancy "invalid device ordinal"
Two single-GPU gsplat sbatch jobs packed on ONE node → the 2nd dies `CUDA error: invalid device
ordinal` at first rasterization; a job ALONE on a node always works. Chased two false leads first:
- `--export=ALL` from the claude_node leaks parent SLURM GPU context → contributes; fix = submit
  clean `--export=<vars>` only (needs `export PATH=$HOME/.local/bin:$PATH` for uv). See memory.
- Pinning `CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS` (physical idx) → "No CUDA GPUs available", which
  PROVED the gg:g4 gres IS cgroup-isolated (allocated GPU = device 0). So CVD=0 was right all along.
- Real robust fix for the tail: run remaining work as ONE consolidated sweep (a lone process can't
  self-pack). Also `--exclude=firefoot-08` (separate gsplat shared-memory bug on that node).
**Lesson: prefer FEW big sweeps over wide swarms for gsplat data-gen.** `run_process.sh` /
`run_recovery.sh` now strip `--killable` (queue policy per-submit: `--account=sagieb` vs
`--killable`), pin PATH, echo the GPU binding, and warn against `--export=ALL`.

### V18 — V17 recipe on 2× data, STANDARD setup (tars retired)
User: the `/dev/shm` tar-staging doesn't scale as the dataset grows → **ditched tars**.
`train_v18_256.sh` / `train_v18_512lp.sh` (copies of V17) now read DIRECTLY from NFS
(`data_v10/h5s_20k_rec` + `renders`, 8 ranks × `--num_workers 8`). Training is compute-bound
(~6 samples/s) so on-demand small-file reads should hide behind compute — **watch first-epoch time;
bump workers if I/O-bound.** Deleted `data_v10/tars` (~25 GB). Stage A (job 31136104, 60ep@256)
+ Stage B (31136105, afterok, 36ep@512+LPIPS0.5) submitted 8×g4 killable; both PENDING on the
strained cluster. Gate on LPIPS val + `--crop_fg` vs **V17-ep36 (28.49 dB / skull 0.110)**.

### OPEN DISCUSSION — 4-GPU Sagie fallback vs 8-GPU killable (comparability)
Cluster strained; 4-GPU Sagie may get an allocation more reliably than 8-GPU killable. But
`training/train.py` has NO gradient accumulation, so effective batch = #GPU × bs = **8 (V17) vs 4
(V18 on 4 GPU)** — a real confound. At micro-batch sizes the effect is modest (total data/epochs/
per-step LR/aug unchanged; 4 GPU just takes 2× more, slightly noisier steps at the same epoch-based
cosine LR), but not nothing. Airtight fix if forced to 4 GPU: `batch_size=2` (→ effective 8), but
bs2@512 with N=20k may OOM (test first; bs2@256 likely fits). Wall-clock: 4 GPU ≈ 2× slower, and 2×
data already ≈ 2× V17/epoch → 4-GPU V18 ≈ 4× V17 per epoch (~2 weeks). **Decision: leave the 8-GPU
killable queued for now; revisit 4-GPU Sagie if it hasn't landed by ~a day.**

## 2026-08-02..05 — THE DIAGNOSIS ARC: ceiling at scale, three hypotheses killed, N-sweep → capacity floor

The week the project's framing changed. Sequence: Sagie meeting (Aug 2) → "treat rec-GT as an
asset; measure it at scale; hunt peculiar cases" → five measurements, each killing a live
hypothesis. Everything below is FG-cropped (`_fg_crop`), `--input_mode recovered`, per-OBJECT means
(views of one object are correlated — never count renders as samples).

### 0. Correction first: our "unseen object" numbers were TRAINING objects
`training/dataset.py` globs the whole h5 dir; V16/V17 trained on ALL of `h5s_20k_rec` (idx<15000 —
V17's log says "13405 scenes", exactly that count). The skull (9869), statue (10916), honeypot
(10751) and all 200 showcase scenes were in V16/V17's training set; only V14 (trained idx<3000) was
genuinely out-of-sample on them. The real held-out split is `h5s_20k_rec_val`+`renders_val` (1,806
objects, own 0–1999 index space) — never evaluated before this week. The old 3-object 28.49 dB
headline was actually PESSIMISTIC: true held-out mean is **30.29 dB** (the hand-picked objects were
harder than average). Everything below uses the proper splits.

### 1. rec-GT ceiling measured at scale (27,224 renders; `ceiling_eval.py` + `ceiling_report.py`)
Three slices, 4 matched views (0/4/7/11) each: TEST = all 1,806 held-out; TRAIN = 3,000 of V17's
own training objects; UNSEEN-2x = 2,000 of the idx≥15000 expansion (unseen by V17).

| slice | n | rec-GT | V17 | margin |
|---|---|---|---|---|
| TEST | 1806 | 44.94 | 30.29 | **14.65** |
| TRAIN | 3000 | 45.02 | 30.90 | **14.11** |
| UNSEEN-2x | 2000 | 44.91 | 30.21 | **14.70** |

- Ceiling is ~45 dB and remarkably uniform. The oft-quoted "~10 dB gap" was an underestimate.
- **Train→test gap = 0.55 dB → the blur is UNDER-FITTING, not a generalisation failure.** V17 is
  14 dB below the ceiling on data it saw ~27×/view. (Corroborated: train LPIPS 0.0113 vs val
  0.0140 at fully-annealed LR.) This overturns the "generalisation gap" framing we'd used since June.
- rec-GT is a hard practical ceiling: V17 ≥ rec-GT in **2 of 27,224** renders (one is a pruning
  hole in rec-GT that V17 smooths over).
- Whole-image PSNR inflates both rec-GT and V17 by ~6.3 dB — metric argument, quantified.
- Peculiar-case strips: `meeting_material/ceiling_cases/`. Smallest margins = LOW-ceiling objects
  (pruning-damaged inputs), not model strength. Largest margins (25–35 dB) = emissive objects.

### 2. Content stratification (`ceiling_content.py`, CPU-only): detail, not brightness
The emissive lead from the tails was a SELECTION ARTIFACT (sorted by tails, reasoned from tails):
only 2.57% of objects have any saturated pixel; excluding them moves the mean ~0.03 dB. The real
driver is **fine-detail content**: margin +2.9 dB per hf_energy tercile IN EVERY SIZE BAND
(hf_energy × fg_frac are −0.62 correlated; 2D table disentangles). Key asymmetry: **rec-GT is flat
44.7–45.6 dB across detail quintiles while V17 falls 33.98 → 28.51** — input carries the detail at
constant fidelity; only the model degrades with it.

### 3. Spectral probe (`spectrum_probe.py` + coherence): it isn't even BLUR
- MTF (model power / GT power per radial frequency, native 512 grid): V17 retains **65–95% of GT's
  power at every frequency down to 2 px**. No knee at the 8-px patch scale (0.125 cyc/px) — the
  ray-token tokenisation hypothesis is dead (also refuted by: 4× overcomplete token capacity, and
  the June overfit hitting 49–53 dB at 512 with the same patch size).
- Coherence (phase alignment with GT): on high-detail objects at ~4 px, **MTF 1.05–1.14 with
  coherence 0.27–0.33** → the model emits MORE fine structure than GT containing, essentially
  uncorrelated with truth. **The failure is misplaced/invented detail, not missing detail.**
  "Blur" was the wrong word all along; error energy in that band is ~1.5× GT energy.
- Lineage check: V14→V16→V17 improved BOTH MTF and coherence at every frequency — training
  progress was real placement learning, not cosmetics. A ~4 px MTF>1 anomaly appears in all three
  models (checkerboard signature at the DPT ConvTranspose stride — untested lead).

### 4. N-SWEEP (`train_nsweep.sh` / `run_nsweep_eval.sh`): the floor is architectural
Nested subsets 10⊂100⊂1000 (idx<15000), same init (`checkpoints_v18_256/phase2_epoch_30.pt` — see
V18 note below), same 30k optimizer steps, 4×bs1@512+LPIPS0.5. Evaluated vs ceiling on OWN training
objects (fit) + common 300 held-out (disjoint from the val100 used in-training):

| N | margin on OWN train | margin on heldout |
|---|---|---|
| 1 (June probe, different setup) | ~0 | — |
| 10 | **7.57** | 20.47 |
| 100 | **13.40** | 17.93 |
| 1000 | **15.79** | 16.70 |
| 13,405 (V17; ~8× steps, eff.batch 8 — reference not curve) | 14.11 | 14.65 |

- **The model cannot fit even 10 objects to the ceiling** (12,000 passes each, still 7.6 dB short).
- **Train and heldout curves CONVERGE to ~14–15 dB.** More data moves along the curve toward the
  floor; it cannot cross it. **The "more data" thesis is retired.**
- Shape = capacity signature (fixed weights spread over more objects), but routing (below) fits too.

### V18 status: stage A done and REPURPOSED; stage B not run (deliberately)
Stage A (256, 30ep on 2× data) finished Aug 4 (val log-L1 0.000793) but exited 7 (benign teardown
artifact — same class as the eval arrays) → `afterok` auto-cancelled stage B, which was anyway
spooled with the OOM bs2 config. Given §4, stage B's premise (more data) is dead; NOT resubmitted.
Stage A's checkpoint became the common init for the N-sweep. bs1@512 CONFIRMED fits on 45 GB nodes
(never previously validated off khan-01).

### Infra lessons that cost real time (all in memory + fixed in scripts)
- `uv run --frozen` still RECONCILES the venv every call (flash-attn version-string churn) → 45-job
  array raced the shared NFS venv: ImportErrors + silent FlashAttention→SDPA fallbacks. Fix:
  `--no-sync` everywhere parallel.
- torch JIT extension cache keys on py+CUDA but NOT GPU arch → heterogeneous g4 pool clobbers its
  own gsplat build; then two same-arch jobs raced too. Fix: per-arch `TORCH_EXTENSIONS_DIR`, and
  per-JOB dirs (seeded from arch cache) for anything parallel.
- Trailing `echo` in sbatch scripts masks python exit codes → jobs report COMPLETED 0:0 with zero
  output rows. Always `exit $rc`.
- Killable-pool preemption waves kill ALL killable jobs at once; jobs whose checkpoint interval ≈
  survival window make no net progress (N=10 thrashed). Guaranteed-quota chaining
  (`--dependency=afterany`) fixed it.

### WHERE THIS LEAVES US — the one live fork
Five hypotheses measured, five killed: not generalisation, not emissive/HDR, not tokenisation, not
blur, not data quantity. Remaining candidates, discriminated by the next experiment:
- **CAPACITY**: weights can't hold many objects' worth of detail-placement. Test: scale model
  (width/depth/scene tokens) at FIXED N=100, watch the train-fit margin.
- **ROUTING**: cross-attention can't resolve which of 20k Gaussians land in which 8×8 ray patch
  (rasterisation does this trivially by sort+splat — exactly why rec-GT is flat across detail).
  If wider models don't close the train-fit margin, this is it.

## 2026-08-05..07 — THE CONTROL CAMPAIGN: the floor is CAPACITY

Follow-up to the diagnosis arc: five controlled experiments that turned "architectural floor"
from a diagnosis into a mechanism. All N=* runs share the nested subsets
(1 ⊂ 10 ⊂ 100 ⊂ 1000, `data_v10/nsweep/`), the v18_256-ep30 init, 512+LPIPS(0.5), and equal
optimizer steps unless stated; readout = FG-cropped margin vs the rec-GT ceiling on each run's
OWN training objects (fit) + a common 300-object held-out set.

### Training-code cleanup first (semantics-preserving, PR'd)
`train.py`/`dataset.py`: EXR-resize corruption fixed, uint8 target re-quantization removed,
`--max_samples` prefix bias → seeded subset, per-step `.item()` syncs → on-device accumulators,
`--weight_decay`/`--fg_bg_weight`/`--latent_dim`/`--encoder_layers`/`--view_layers`/
`--from_scratch` flags added. Same-node OLD/NEW bench: **identical 117.4 s/epoch** (sync removal
is NOT a speedup at production shape — GPU-bound) with loss curves matching to 3 decimals.

### Recipe controls: aug + weight decay are EXONERATED at scale
| arm (N=100 unless noted) | train-fit margin | heldout margin |
|---|---|---|
| baseline (V17 recipe) | 13.40 | 17.93 |
| N=10 baseline → aug off + wd 0 | 7.57 → **5.72** | 20.47 → 20.55 |
| N=100 aug off + wd 0 | **13.13** | 17.90 |
| N=100 **fg-weighted loss** (`--fg_bg_weight 0.05`) | **11.77** | **17.66** |

- The aug/wd effect is −1.85 dB at N=10 but **−0.27 dB at N=100** — a small-N artifact
  (rotation-equivariance is a big relative burden on 10 objects). Not a lever at scale; keep both.
- **fg-weighted loss is the only intervention improving BOTH columns** (−1.63 fit, +0.27
  heldout; eval LPIPS unchanged → real pixel accuracy). The loss had the same whole-image
  dilution as the metric we already distrusted: objects are 2–7% of pixels, and the log-L1 val
  "floors" sit at the 8-bit target quantization noise. **Production-recipe candidate.**
- N=10 ctrl also exposed a perceptual/pixel split: train LPIPS reaches ~N=1 level (0.0011)
  while PSNR stays 5.7 dB short — fits perceptually, not pixel-precisely (coherence story).

### From-scratch is UNTRAINABLE — RenderFormer pretraining is load-bearing
Width pair (d768-scratch control vs d384-scratch, depth preserved): both arms — straight to 512
AND with a matched 256 log-L1 warmup — park at **LPIPS ≈ 0.092–0.093** and never move (d384 ran
its full 600-epoch schedule flat; from-scratch 256 converges only to log 0.013 vs warm 0.0008).
**The 0.092 plateau is a degenerate attractor** that swallowed four inits this week; only intact
pretrained weights escape it, and the 512+LPIPS objective supplies no useful gradient until the
output is roughly right. Width capacity comparison therefore unanswerable from scratch — but
"pretraining is structural, not convenience" is a finding.

### Depth-pruned probe: CAPACITY BINDS
d=768 kept (weights load), enc 12→6 + view 6→4 (**DPT taps the LAST 4 view layers — view depth
≥4 is an architectural minimum**; take-1 died on the unpack), warm layer-drop init
(`make_pruned_ckpt.py`, even layers, 114.6M vs 194.9M), 256-recovery stage R (escapes the
attractor; recovered to 0.0011 vs intact 0.0008 → init damage bounded small), then the exact
N=100 baseline schedule:
**train-fit margin 17.23 vs 13.40 (−40% params → −3.8 dB fit), heldout 22.75 vs 17.93.**

### N=1 controls (user-requested): anchors validated, and the model BEATS the ceiling
Under the CURRENT recipe (aug ON, wd ON, 30k steps): objav scene_0387 margin **−2.15 dB**
(48.47 vs rec-GT 46.31, LPIPS 0.001 — it corrects pruning artifacts toward true GT; **rec-GT is
an information bound, not a pixel bound**); tomatoes train LPIPS 0.00007 ≈ the June probe.
Heldout collapses to 19 dB (catastrophic forgetting, as expected).

### SYNTHESIS — the mechanism of the 14 dB floor
- fit vs **N** at fixed capacity: **−2.15 → 7.57 → 13.40 dB** (N=1→10→100)
- fit vs **capacity** at fixed N=100: **13.40 → 17.23 dB** (194.9M → 114.6M)

Both axes move together: GaussianFormer is a **capacity-limited memorizer** — per-object
fidelity tracks objects-per-parameter. That is why V17 under-fits its own training set by 14 dB
and why 2× data (V18) could never have helped. **V19 levers, evidence-backed:** (1) more
capacity via warm depth-EXPANSION (layer duplication; width is closed — scratch untrainable);
(2) fg-weighted loss; (3) keep aug/wd, keep the pretrained init.

### Infra (memory + scripts updated)
khan-01/02 insta-fail all jobs at prolog (0–1 s, no output file) while sinfo reports healthy —
excluded everywhere, report to admins. Killable preemptions register FAILED (not requeued) —
resubmit manually; checkpoint cadence must beat the preemption interval or a run thrashes.

## 2026-08-08..13 — capacity fully eliminated; the READING CEILING isolated; tomato-codec pivot

### The elimination table completed (all N=100, vs baseline train-fit margin 13.40)
Depth 18/9 (287M) 13.43 | enc-only 13.38 | view-only 13.42 | FFN x8 (322M) 13.35 | input-head MLP
13.43 | log-scale input 13.43 — **all exactly flat**. From-scratch untrainable at any width (LPIPS
~0.092 attractor; only intact pretrained inits escape; 256-ramp required for any damaged init).
What DID move fit: **fg-weighted loss 11.77** and **cosine-cycle restarts**: 30k→60k→90k steps =
13.40→12.26→11.48, IDENTICAL for 195M and 287M at every point (capacity dead at all budgets);
decelerating toward an extrapolated ~10 dB optimization asymptote.

### Tomatoes rebuilt with the modern pipeline (219k-Gaussian real scan, 11x prune)
Recovery lifts the ceiling 29.61 → **46.05 dB** — its biggest validated win; the June "20k can't
hold real scans" cap was pruning quality, not representation. 4-way (same rec input to both
models): GT | rec-GT 46.05 | **plain V17 32.44** | **overfit 53.02** (train views).

### THE KEYSTONE: the ~30 dB reading ceiling (novel-view overfit probe, user-suggested)
Overfit on train views 52.8 dB; on NOVEL views **29.9 dB while the input holds 44.9 dB at those
same angles**. June's 30.5-vs-29.7 was input-starved and ambiguous; now input +15 dB → output +0.
Convergence across regimes: overfit-novel 29.9 ≈ V17-on-tomatoes 32.4 ≈ general-model 30.9.
**Reading scenes from tokens caps at ~30 dB regardless of training regime, data, capacity,
optimization, conditioning, or input quality. Weights-recall (train views, N=1) bypasses it.**
The bottleneck is the cross-attention readout (ray tokens must approximate projection+occlusion
with learned dot-products; rasterization does it exactly — why rec-GT is flat at ~45-46).

### NEXT ARCHITECTURE DIRECTION (flagged for after the codec work): geometry-biased cross-attention
Add computed projection proximity as a zero-init-gated bias on view-transformer attention logits
(camera-space positions already available as gaussians_view_tf). Warm-safe by construction;
falsifiable on the tomatoes-novel-view harness (prediction: 30 → toward 45). Alternatives ranked:
projection-restricted top-k keys; splat-then-refine hybrid (rasterized canvas as conditioning);
transmittance/occlusion bias. See session notes 2026-08-13.

### IN FLIGHT: tomato "neural codec" overfit (branch exp/tomato-codec)
Rationale (user): a per-scene model that beats rasterizing its own compressed splat on EVERY view
is a useful artifact. Train = 200 RANDOMIZED views (az/el free, radius 1.35-2.15 zoom) targeted on
full-splat renders, input = recovered 20k. Eval = disjoint random views + radius EXTRAPOLATION
(1.15 / 2.45) so view-interpolation cannot masquerade as reading. Jobs 31272109 (datagen) →
31272110 (train, Sagie 4x; no idle 8-GPU killable at submit). Gate: novel-view PSNR vs rec-GT 46.

## 2026-08-13..16 — the CODEC arc: view coverage is a lever; near-parity with rasterization at N=1

### The reading-ceiling revision (codec v1, 200 randomized views incl. zoom 1.35-2.15)
Novel-view (in-range) 39.6 dB vs the 14-view overfit's 29.9 — **the "~30 dB reading ceiling" was
substantially VIEW SPARSITY**, not a hard readout limit. But zoom EXTRAPOLATION collapsed
(close 27.0 / far 27.4 vs rec-GT 38.7/45.8): the model learns the covered view MANIFOLD, not a
camera-independent object. 0/40 views beat rec-GT.

### v2 (500 views, radius 1.05-2.55 = eval interior): zoom collapse GONE
40.7 / 36.1 / 42.0 (rand/close/far) — close +9.1, far +14.6 purely from radius coverage in
training. First 3 individual novel-view wins over rec-GT.

### v3 (1500 views + 2nd cosine cycle): NEAR-PARITY
**42.38 / 38.60 / 43.83 vs rec-GT 44.00 / 38.66 / 45.79** — close-range statistical parity
(-0.06 dB, wins 5/8); 11/40 views beat rec-GT overall; far-set LPIPS BETTER than rec-GT.
Trajectory 39.6→40.7→42.4 not yet bent. Strips: data_external/tomatoes/renders/codec*_verdict*.
User rationale: a per-scene model beating rasterization of its own compressed splat = useful
artifact (neural codec for single Gaussian scenes). Remaining ~1.6-2.0 dB: more views/cycles
(brute) vs geometry-biased attention (mechanistic) — direction call for Sagie.

### IN FLIGHT: the codec method at N=10 (jobs 31286027→28→29→30, branch exp/tomato-codec)
150 random views/object x the 10 nested objects (full-splat targets from the retained full_h5s),
2 cycles = 60k steps; eval = the standard orbit views, which are NOVEL for this model. Question:
does the coverage lever survive weight sharing? Ladder refs: N=10 ctrl train-view fit 5.72;
reading regime ~30; codec-at-N=1 novel 42.4. If it transfers -> view-dense supervision (free via
rasterization) joins the V19 recipe; if not -> coverage was substituting for per-object capacity
and geometry-biased attention inherits the burden.

### Standing V19 recipe facts (unchanged): fg-weighted loss + multi-cycle schedule; architecture
changes all flat at N=100; geometry-biased cross-attention = flagged next architectural probe.

### LATE-READ VERDICT (2026-08-16): fg-loss STACKS with cycling
expand-r2+fg (60k steps + fg-weighted loss): train-fit margin **10.04 dB** — best of the campaign
(vs 11.48 for 90k plain cycles, 11.77 for fg alone at 30k). The two validated levers are additive.
Heldout 18.44 (vs 17.93): at fixed N=100 the extra fit is memorization-flavored — full-N behavior
is the V19 question. V19 recipe: fg + multi-cycle, confirmed compound.

## 2026-08-16 (later): geometry-biased cross-attention BUILT + N=100 probe launched
Branch exp/geom-bias-attn. Implementation surfaced the smoking gun: the view transformer's RoPE
assigns EVERY patch token the same position (the camera origin), so cross-attention logits carry
zero per-patch geometry — "which Gaussians lie on my ray" is inferred from direction features
alone (consistent with the coherence probe: energy present, spatially misplaced). Change: per
view layer, a zero-init scalar gate x cos(patch ray dir, camera->Gaussian dir) added to the
cross-attn logits (--geom_bias, train + ceiling_eval). Gate=0 verified BIT-EXACT vs v18_256-ep30;
biased layers run SDPA (~105 s/epoch, roughly baseline cost). Chain: probe 31287211 (passed,
no OOM on 45G) -> main 31287212 (EXACT N=100 baseline schedule, 30k steps, killable) -> eval
31287213 (TAG=nsweep_n100_geombias). Read: train-fit margin vs baseline 13.40; also read the
learned gate values — gates parked at zero mean the model declined the hint. 8-GPU attempt
abandoned: no whole node free (firefoot-13 IDLE+DRAIN bad GPU; khan excluded), and the 4-GPU
fallback restores exact step-count comparability anyway.

## 2026-08-17: BOTH probes land — two clean negatives
### Geom-bias verdict: the model DECLINED the geometry hint
nsweep_n100_geombias train-fit margin **13.40 dB — identical to baseline 13.40** (heldout 17.93,
also identical). The learned gates settled at -0.005..-0.029, i.e. essentially zero: given a free,
exact "which Gaussians are on your ray" signal in the cross-attn logits, 30k steps of training
chose not to use it. The routing-prior hypothesis is falsified at this scale — the bandwidth
bottleneck is NOT cross-attention routing. (Chain 31287211-13, branch exp/geom-bias-attn.)

### N=10 codec verdict: view coverage does NOT survive weight sharing (at this budget)
nsweep_n10codec on the standard orbit views (NOVEL for this model): model 30.21 dB, margin
14.24 — parked exactly at the old ~30 dB reading regime, nowhere near codec-N=1's 42.4 novel.
150 views/object at N=10 (vs 1500 at N=1) does not transfer; dense-view supervision was
substituting per-object capacity, not teaching generalizable rendering. Heldout300 margin 20.76
(worse than V17's 14.65 — expected, 10 training objects). (Chain 31286027-30.)

### Where this leaves V19
Both mechanistic escape routes just closed: not routing (geom-bias declined), not supervision
density (codec doesn't share). Confirmed levers remain fg-weighted loss + multi-cycle schedule
(stack to 10.04 at N=100). Remaining open hypotheses: pruning-score bias (input quality) and
raw optimization budget (cycles asymptote ~10 dB).

## 2026-08-18: CODEC4 — GOAL MET on the tomatoes: model beats rec-GT on average
Direction (Sagie + user, 2026-08-17): the per-object codec line IS the line — beat rec-GT on
average on the tomatoes, then scale to other objects. Codec4 scaled v3's two validated levers:
**3x views** (4500 randomized, r 1.05-2.55, seed 41, free supervision from the full 219k splat)
and **cycle continuation** (seeded from codec3_r2 ep80, two more ~30k-step cosine cycles;
27 ep x 1125 steps each). Branch exp/tomato-codec4; chain 31290091-95, flash-attn confirmed.

**Cycle 1** (checkpoints_tomato_codec4/phase2_epoch_27.pt): rand -0.50 / close **+1.74** /
far -0.81 — close-set flips positive for the first time; average gap -0.11 dB (v3 was -1.38).

**Cycle 2 FINAL** (checkpoints_tomato_codec4_r2/phase2_epoch_27.pt): rand **-0.06** (13/24
views won) / close **+2.29** (6/8) / far **-0.34** (3/8) = view-weighted average **+0.36 dB
over rec-GT, 22/40 views won**; model LPIPS beats rec-GT on all three sets. Verdicts:
data_external/tomatoes/renders/codec4{_c1,}_verdict.{png,json}.

Both levers still deliver: 3x views alone was worth ~+1.3 dB average, the extra cycle ~+0.45.
Residual loss is concentrated in ONE view — the grazing-angle plate view (close v1, -3.5);
targeted view sampling near grazing elevations is the obvious next lever if we need margin.
**Next per the direction: scale the recipe to other objects.**

## 2026-08-18: CODEC5 launched — squeeze pass targeting views-won
User directive: before scaling out, push the tomatoes further; the KPI is a higher views-won
count (codec4: 22/40). Two changes: (1) **9000 views** = codec4's 4500 (reused via symlinks) +
4500 new views drawn from GRAZING elevations (el -15..20 deg, same az/radius coverage) — the
surviving losses concentrate there (close v1 = edge-on plate view, -3.5); (2) **8xbs1** on a
whole idle node (firefoot-09/10/17 were free; runs on firefoot-17), so 9000/8 = the same 1125
steps/epoch and each ~30k-step cosine cycle sweeps 2x the data at codec4's wall time (~8.5h).
Seeded from codec4_r2 ep27 (cycle continuation). Effective batch 4->8 (user-approved).
Chain: 31296841 datagen -> 31296842 cycle1 -> 31296843 eval-c1 (TAG codec5_c1) -> 31296844
cycle2 -> 31296845 eval-c2 (TAG codec5). Eval sets FROZEN (codec_eval_*) for v1-v5
comparability; evals reuse tomatoes_codec4_eval.py via CKPT/TAG env. Faster-iteration knob
if needed: EPOCHS_OVR=14 = ~15k-step cycle in ~4.5h (untested annealing, kept at 30k for now).
Next in parallel: pick 10 scale-out objects with the user (color-rich + high-freq, >=1 simple
object as a convergence case study; NO HDR-streak/emissive objects). NOTE: data_v10/full_h5s
was deleted 2026-08-17 — chosen objects need their FULL splats rebuilt via the data_v10
pipeline before dense-view rasterization.

### Scale-out object list LOCKED (user + Claude, 2026-08-18)
10 objects from the 1806-object UNSEEN val pool (full splats in data_v10/full_h5s_val, recovered
20k in h5s_20k_rec_val — nothing to rebuild). Rich: scene_0262 painted plate, scene_0772 sandal,
scene_1078 anime figure, scene_1342 boxing ring, scene_0031 seahorse, scene_0223 circus tent,
scene_1423 molecule toy, scene_1223 brain-hair doll. Simple (cycles-to-beat-rec-GT case study):
scene_0959 apple, scene_0874 clay vase. All pass the HDR-streak screen (blown<2% + halo/core
ratio + visual curation; the first auto-pick surfaced the chest & two glow objects — dim
volumetric streaks evade a saturation-only filter). One model per object; recipe frozen after
the codec5 verdict lands.

## 2026-08-18 (late): CODEC5 cycle 1 — ALL sets beat rec-GT; recipe FROZEN; fleet launched
Codec5 c1 verdict (checkpoints_tomato_codec5/phase2_epoch_27.pt): rand **+0.56** (17/24) /
close **+2.89** (7/8) / far **+0.28** (6/8) = **+0.97 dB view-weighted avg, 30/40 views won**
(codec4: +0.36, 22/40). The grazing-view half did its job: the edge-on plate view went
-3.5 -> -1.8 and both rand & far flipped positive. Cycle 2 (31296844) running for the margin.

**RECIPE FROZEN for scale-out:** 9000 views (4500 uniform el -15..55 + 4500 grazing el
-15..20, r 1.05-2.55), 8xbs1, ~30k-step cosine cycles (27 ep x 1125 steps, lr 5e-5, aug ON,
LPIPS 0.5), 2 cycles, cold start from v18_256-ep30. Apple pilot (scene_0959, 31300959)
already running this exact recipe on firefoot-10, save_interval 9 for crossing-point analysis.
**9-object fleet launched** (killable + requeue; cluster saturated, they queue for freed
GPUs): 0262/0772/1078/1342/0031/0223/1423/1223/0874, each c1 -> evals(9/18/27) -> c2 ->
evals, jobs 31304895-31304963. Generic scripts: data_v10/train_codec_scaleout.sh,
data_v10/codec_scaleout_eval.sh, data_external/codec_scaleout_eval.py.

## 2026-08-19: CODEC5 FINAL — tomatoes decisively closed: +1.30 dB avg, 34/40 views
Cycle-2 verdict (checkpoints_tomato_codec5_r2/phase2_epoch_27.pt): rand **+0.88** (20/24) /
close **+3.23** (7/8) / far **+0.61** (7/8). Ladder: v3 -1.38 avg -> v4 +0.36 (22/40) -> v5c1
+0.97 (30/40) -> **v5 final +1.30 (34/40)**. Model LPIPS beats rec-GT on every set. The 6
remaining losses are all small; the grazing plate view is down to -1.2 (from -3.5 at v4).
The tomato squeeze directive (higher views-won) is satisfied; per-object codec DECISIVELY
beats rasterizing its own compressed splat. Attention shifts to the 10-object scale-out
fleet (running under the 2-slot GPU throttle per user request; first verdicts pending).

## 2026-08-19: branch/PR hygiene — the campaign stack is now reviewable
Local `main` was stale at the public-release commit (87fce5e) while origin/main had already
merged PR #5 (data/v10-scaleup). Measured against the *real* origin/main (8109ae6) the whole
experiment stack is **42 commits / ~3.2K insertions**, not the ~310K the stale baseline
implied — the giant diff was entirely data_v10/object_list_train.json (240K lines), already
upstream. Local main fast-forwarded.

The nine branches are strictly nested (each contains its predecessors), so they were opened
as a **stacked PR chain**, each based on the one below — merge in numeric order:
  #6  exp/capacity-controls    -> main            N=100 memorization control (aug off, wd 0)
  #7  exp/fg-weighted-loss     -> #6              --fg_bg_weight; 13.40 -> 11.77 (surviving lever)
  #8  exp/width-capacity       -> #7              --latent_dim/--from_scratch; width FLAT
  #9  exp/depth-capacity       -> #8              layer-pruned probes + N=1 controls; depth FLAT
                                                  (+ real fix: view transformer needs >=4 layers,
                                                   DPT taps the last 4)
  #10 docs/campaign-log        -> #9              campaign log; corrects superseded claims
  #11 exp/v19-depth-expansion  -> #10             growth arms FLAT; r2 gain was steps not capacity;
                                                  tomato rec-20k rebuild (ceiling 29.61 -> 46.05)
  #12 exp/tomato-codec         -> #11             codec v1-v3 (near-parity) + N=10 NEGATIVE
  #13 exp/geom-bias-attn       -> #12             --geom_bias FALSIFIED (gates ~0); bit-exact off
`exp/tomato-codec4` (codec4/codec5 + the 10-object scale-out, 11 commits) is pushed as a
backup but deliberately NOT PR'd yet — the fleet is still producing verdicts on it.

**Repo fix while doing this:** the 2026-08-18 storage cleanup had deleted *tracked source*
under data_v2/, data_v9/ and gaussian_h5s/ (the bulk data there was gitignored, the scripts
and demo h5s were not). Restored via `git checkout` — 142 MB, 25 files. Lesson: `rm -rf` on a
data directory needs a `git status` check afterwards.

## 2026-08-20: warm-restart dip — mid-cycle evals are NOT comparable
Evaluated the apple's cycle-2 epoch-9 checkpoint early (job 31332255) instead of waiting for
the cycle to finish. Result looked alarming: **41.35/39.03/39.83 vs rec-GT 50.17/46.03/53.07
= -9.34 avg, 0/40** — a ~7 dB REGRESSION from cycle 1's end (47.96/45.65/49.17, -2.18, 6/40).
The rec-GT baseline is byte-identical across all four apple evals, so this is not an eval bug.

**Cause: the cosine warm restart.** Each cycle re-instantiates CosineAnnealingLR at
phase2_lr=5e-5, so epoch 1 of a cycle yanks LR back to peak and knocks the model out of the
minimum it had annealed into; it re-anneals over the cycle. Epoch 9/27 is measured near peak
LR, i.e. at the worst point. This retro-explains why every END-of-cycle number in the tomato
ladder improved monotonically (-1.38 -> +0.36 -> +0.97 -> +1.30) while nothing mid-cycle was
ever measured.

**Consequences:**
- Only END-of-cycle checkpoints (epoch 27; epoch 20 for the 6-GPU vase c1) are comparable
  across objects/cycles. Intra-cycle points measure schedule phase, not capability.
- My earlier extrapolation ("apple crosses rec-GT early-to-mid cycle 2", from the +2 dB/9
  epochs trend inside cycle 1) was WRONG for this reason — within-cycle slope cannot be
  extended across a restart boundary. Expect the crossing at the END of cycle 2.
- The fleet's scheduled epoch-9/18 evals still run (they cost ~15 min on a killable GPU) but
  should be read as progress traces only, never as cross-object comparisons.
- Dashboard updated: end-of-cycle points draw solid, mid-cycle points hollow, with a legend
  note (data_v10/codec_scaleout_dashboard.py).

### Checkpoint cadence vs preemption window (2026-08-20)
The ring and vase killables spent ~12 h making ZERO durable progress: the cluster was handing
out slots shorter than their save intervals (9 ep ~3 h, 5 ep ~2.5 h), so every preemption
discarded the whole slot's work. Neither had written a checkpoint since Aug 19.
**Rule: save_interval must be shorter than the typical preemption window, and must DIVIDE the
epoch count** (else the final epoch gets no numbered checkpoint — 27 admits 1/3/9/27, 20
admits 1/2/4/5/10/20). Rebuilt both chains: ring saves every 3 ep (~1 h, resumes from ep 18),
vase every 2 ep (~1 h on 6 GPUs, resumes from ep 15). Mid-cycle evals dropped — per the
warm-restart finding only end-of-cycle checkpoints are comparable, so each cycle now has
exactly one eval. Jobs: ring 31333331/33, vase 31333335/37.

## 2026-08-20: apple cycle 2 — FLAT, and the "simple object" premise inverts
Apple (scene_0959) end-of-cycle results: **c1 −2.18 dB / 6-of-40 → c2 −2.14 dB / 7-of-40**.
A full second 30k-step cycle bought +0.04 dB and one view. Cycles have SATURATED for this
object, unlike the tomato ladder where every cycle paid (−1.38 → +0.36 → +0.97 → +1.30).

**Why — the bar, not the model.** Per set the apple's c2 is model 48.00/45.64/49.26 against
rec-GT **50.17/46.03/53.07**. Compare the finished tomato: model 44.88/41.89/46.40 against
rec-GT 44.00/38.66/45.79. **The apple's model is ~3 dB BETTER in absolute PSNR than the
tomato's and still loses**, because a smooth simple object is rasterized almost perfectly by
its recovered 20k splat, putting rec-GT ~6 dB higher.

So "beat rec-GT" difficulty is governed by **how well the compressed splat represents the
object**, not by how hard the object is to render:
- SIMPLE/smooth (apple): model renders it superbly (48 dB) but rec-GT is near-perfect (50) → hard.
- RICH/detailed (plate, sandal, figure at ~−8.5): rec-GT is beatable in principle, but the
  MODEL cannot yet render the detail → hard for the opposite reason.
- TOMATOES: the sweet spot — a real scan detailed enough to handicap the 20k splat (rec-GT
  44.0) yet learnable to 44.9 with 180k steps + 9000 views.

This reframes the scale-out: the tomato win may sit in a middle band rather than generalising
uniformly, and the simple-object case study answers its own question — a simple object is NOT
the quick win we assumed. Open question for the remaining objects: where does the crossover
band actually lie, measured as (rec-GT quality) vs (model attainable PSNR)?

**Bug fixed while reading this:** the eval's default TAG was scene+epoch, so cycle-2 verdicts
silently OVERWROTE cycle-1's (both end at phase2_epoch_27). Cycle-1 apple JSON/PNG were lost
and are being regenerated with cycle-qualified tags; the numbers themselves survived in this
log. Tag now includes the checkpoint dir (data_external/codec_scaleout_eval.py).

## 2026-08-22: cycle-2 seed bug — empty _r2 dir cold-started c2 from v18
The c2 seed check tested `[ ! -d ..._r2 ]`. A preempted first slot mkdir's the r2 dir
before saving anything, so the requeue found the dir, skipped SEED_OVERRIDE, and
cold-started from v18_256 — i.e. re-ran cycle 1 labelled as cycle 2. Bit **plate 0262**
(its r2 ep9, quarantined as TAINTED_*.pt.bad) and **sandal 0772** (caught at ~1.2k steps,
requeued). Fix (ca42da4): cycle 2 always sets SEED_OVERRIDE (the resume glob still wins
when checkpoints exist), and default save_interval 9→3 per the preemption-cadence rule.
0959/0874/1342 r2 runs predate the empty-dir path and their c2 numbers improved over c1 —
lineages clean.

**Fleet standing (end-of-cycle only, view-weighted avg / views won):**
apple −2.14 7/40 (c2, SATURATED) · vase −5.49 0/40 (c2) · boxing ring −7.18 3/40 (c2) ·
plate −8.61 0/40 (c1) · sandal −8.21 0/40 (c1) · figure −8.80 0/40 (c1).
No scale-out object beats rec-GT yet; cycles pay ~+1 dB on the rich objects (ring c1→c2
−8.13→−7.18, vase −6.66→−5.49). 0031/0223/1423/1223 still user-held (2-slot throttle).

## 2026-08-22: two real-scan objects added — octopus & crocs slipper (superspl.at, CC BY 4.0)
Octopus `f9063eda` (1.84M g, fine sucker detail + Rubik's cube color patch) and "new_gopro"
`6bc0df7c` (actually a fuzzy Crocs slipper, 1.17M g, high-freq fleece texture). Found the
scriptable download path: the viewer's SOG bundle is public on CloudFront, and the site's
official PLY download is generated FROM it (MD5-identical) — no lossless original exists.
Both need the tomato flip_x (superspl.at gravity-down); probes confirmed. New generic prep
(data_external/prep_external_codec.{py,sh}): raw PLY -> normalize -> 20k prune+recovery ->
codec_scaleout layout (seed bases 200000/200100). Prep jobs 31351309/10; training chains
submitted HELD per the 2-slot throttle: octopus c1 31351312 / c2 31351314, gopro c1
31351316 / c2 31351318, end-of-cycle evals chained (31351313/15/17/19).

## 2026-08-23: c2 sweep COMPLETE — rich objects converge to −7.0…−7.4, none crosses
All six active objects now have two full 30k-step cycles (end-of-cycle verdicts, weighted
avg / won-of-40): plate **−6.99** 0 · sandal **−7.13** 0 · ring −7.18 3 · figure **−7.37**
0 · vase −5.49 0 · apple −2.14 7 (saturated). Cycle-2 gains: plate +1.62, figure +1.43,
vase +1.17, sandal +1.08, ring +0.95, apple +0.04 — NO decay yet on rich objects, and the
four rich ones converged to a 0.4 dB band from an 0.7 dB-wide c1 spread. The quota burst
(user-approved, then rescinded: lab strained, killable-only until further notice) ran the
three c2s in ~1 day. Proposed next: c3 on vase+ring to measure cycle-gain decay before
concluding whether cycles can close the gap or the band is bandwidth-limited.
Octopus c1 (killable, epona-02) training; slipper c1 held.

## 2026-08-24: OCTOPUS c1 — best cold-start in the campaign: −2.44 avg after ONE cycle
First from-the-wild scan verdict (octopus f9063eda, 1.84M g → 20k rec = 92× compression):
rand −3.18 (0/24) / close **−0.75** (1/8) / far −1.91 (0/8) = **−2.44 avg, 1/40** at 30k
steps. Fleet objects sat at −8…−12 at the same budget. Cause per the rec-GT thesis: the
92× squeeze wrecks the recovered splat — rec-GT bars 38.1/30.6/39.9 are the lowest yet
(fleet: 43–53) — while the model's 34.9/29.9/38.0 absolute is unbothered by scan density.
Crossover band reframed: not "tomato-like objects" but "scans dense enough to break their
own 20k compression." c2 released (killable). Slipper (~60×) is the obvious next test.

## 2026-08-24 (eve): OCTOPUS c2 −1.68 — first regime crossing; 4-GPU accum recipe validated
Octopus c2 (31351314, 60k steps, seeded from c1 ep27): rand −2.43 (1/24) / close **+0.07
(4/8)** / far −1.16 (0/8) = **−1.68 avg, 5/40** (c1 −2.44, gain +0.76). Close-range is the
first eval regime on any scale-out object with a positive delta. Gain per cycle is smaller
than the rich objects' +0.95…+1.62 — expected since it started nearer the bar.

**4-GPU runs.** Full 8-GPU killable nodes are rare (only firefoot-01/khan-01 free, both
inside the debian13 upgrade reservation 5787), 4-free slots exist. bs=2 is out (OOM @512 on
45 GB, no speedup @256 — attention is compute-bound), so `training/train.py` gained
`--grad_accum` (DDP no_sync on inner micro-steps) and the scale-out script pins the
effective batch to 8 via `GRAD_ACCUM=8/NPROC` (c41f2a9, 77f4109 adds SAVE_OVR). Sanity
(31374203, octopus, 4×L40S×accum2, same schedule): ep1 0.015988 / ep2 0.011319 vs the 8×A40
c1's 0.016186 / 0.011448 — ~1% match. Wall time 2210 s/epoch vs 2257 s on 8×A40 (L40S ≈ 2×
A40), so 4×L40S costs nothing vs 8×epona. Slipper chain resubmitted as 4-GPU killable:
c1 31374518 (running epona-02) → eval 31374519 → c2 31374520 → eval 31374521.
Cluster upgrades to debian13 on 2026-10-05 (test via --reservation=5787); smoke test with an
isolated venv pending. Standing policy: killable only.

## 2026-08-25: debian13 reservation nodes put to work — 16 idle GPUs claimed (killable)
The cluster upgrades to debian13-5787 on 2026-10-05; the upgraded test nodes (firefoot-01
8×L40S, khan-01 8×RTX Pro 6000) sit idle behind `--reservation=5787`. Smoke test
(`data_v10/deb13_smoke.sh`, isolated venv `/cs/labs/sagieb/shahaf_levy/venvs/gf-deb13`):
first run failed "no NVIDIA driver" — the 595.80 user-space libs exist in
`/etc/lib64/nvidia` but ld.so.cache is stale and nvidia-smi is absent; `module load nvidia
cuda` (nvidia/595.80, cuda/13.3) + `LD_LIBRARY_PATH=/etc/lib64/nvidia` fixes it. Octopus c2
eval on debian13 reproduces the debian12 verdict to 0.002 dB. Scripts now auto-detect
debian13 (c7aee4b: venv/cache/lib path/arch-from-torch). Chains launched on the reservation:
scene_0031 c1 31375607 → 08 → c2 31375609 → 10 (firefoot-01), scene_0223 c1 31375611 → 12 →
c2 31375616 → 17 (khan-01, sm_120 — flash_attn works). Slipper c1 31374518 (4×A40) running.
Stale held 8-GPU chains for 0031/0223 (31304927–34, 31304935–42) still queued-held; cancel.

## 2026-08-25: scene_0223 c1 −2.02 (best c1 in the campaign) — khan-01 runs a cycle in 5 h
scene_0223 c1 (31375611, khan-01 RTX Pro 6000 ×8, 657 s/epoch ≈ 3.4× A40): rand −1.90 (1/24) /
close −0.21 (3/8) / far −4.19 (0/8) = **−2.02 avg, 4/40** after one cycle — better than octopus
c1 (−2.44) with a HIGH rec-GT bar (43.8/38.9/46.5), i.e. the model's absolute PSNR (41.9/
38.7/42.4) is the highest seen; far-range is the whole deficit. c2 31375616 seeded from c1
ep27, running on khan-01. scene_0031 (firefoot-01) at ep16/27, 1118 s/epoch.

## 2026-08-25: seahorse (scene_0031) c1 −10.4 — highest rec-GT bar yet; debian13 HC drains nodes
seahorse c1 (31375607, firefoot-01, 8.5 h): rand −10.84 (0/24) / close −8.06 (0/8) / far −11.41
(0/8) = **−10.4 avg, 0/40**. Model 40.2/36.6/40.4 is ordinary; the bar is 51.0/44.7/51.8 — the
highest in the campaign (simple object → near-perfect 20k rasterization; cf. apple/vase).
Confirms the rec-GT-bar thesis from the other side: the fleet spread (−1.7 … −10.4) is bar
spread, not model spread (model absolute sits at 35–42 for every object).
Ops: the debian13 health check flags the RUNNING job's own slurm_script as "fugitive" and
DRAINs the node (firefoot-01 01:31, khan-01 07:27) → chained jobs pinned there hung. Unpinned
the evals (scontrol update ReqNodeList= Reservation=), seahorse c2 resubmitted 4×L40S
killable off-reservation (31377176 → eval 31377177). Reported to system group.

## 2026-08-25 (eve): TENT c2 −0.78, 17/40 — closest scale-out object yet; c3 queued
tent c2 (31377511; resumed from ep21 after the disk-full truncation of ep24/27): rand −0.57
(**10/24**) / close **+1.00 (7/8)** / far −3.21 (0/8) = **−0.78 avg, 17/40**. Gain +1.24 over c1
(−2.02) — the largest c2 gain so far. Model 43.2/39.9/43.3 is the highest absolute PSNR of any
object; far (bar 46.5) is the whole deficit. Script generalised to CYCLE=N (_rN, seeds from
_r{N-1}); tent c3 submitted 4-GPU killable. Lab share hit 100 % at ~11:00 (2 TB quota, mine
351 GB): freed ~80 GB by dropping intermediate ckpts of finished cycles; keep_last_n 2.

## 2026-08-26: seahorse c2 −9.11 (gain +1.3); bar-limited as expected
seahorse c2 (31377491, 4×L40S): rand −9.66 / close −6.35 / far −10.19 = **−9.11 avg, 0/40**
(c1 −10.4). Same +1.3 dB cycle gain as the rich objects; with a 51 dB bar it is the clearest
case for the K-sweep (rate–distortion) framing: the model, not the bar, is the constant.

## 2026-08-26: SLIPPER c1 −0.55, 14/40 — best c1 ever; close-range sweep 8/8 (+1.64)
gopro/crocs slipper c1 (31374518, 4×A40 accum2, 33 h): rand −1.15 (5/24) / close **+1.64
(8/8)** / far −0.94 (1/8) = **−0.55 avg, 14/40** after ONE cycle. rec-GT bars 35.8/28.2/38.4
are the lowest in the campaign (1.17M g → 20k = 60× compression of fuzzy fabric). Model
34.6/29.8/37.5. The dense-wild-scan thesis holds a third time (octopus, tent, slipper): the
crossover band is set by how lossy the 20k compression is. c2 31374520 running (seeded from
c1 ep27); expected to cross on avg if the +1 dB cycle gain holds.

## 2026-08-26: K-sweep pilot (rate–distortion, no retraining) — NEGATIVE; codec is realization-specific
`data_v10/k_sweep.py` (1 GPU, jobs 31383269–72): re-prune the FULL splat to K ∈ {20k,10k,5k,
2.5k} with the fleet recipe, feed the trained c2 codec the K-splat unchanged, compare with
rasterizing the same K-splat (both vs full-splat renders; avg over the 40 frozen views).

| object (full N) | K=20k model/raster/Δ | 10k | 5k | 2.5k |
|---|---|---|---|---|
| seahorse (50k) | 40.1 / 49.9 / −9.9 | 32.5 / 46.5 / −14.0 | 28.9 / 42.5 / −13.6 | 26.7 / 38.4 / −11.7 |
| plate (50k) | 36.4 / 44.0 / −7.7 | 27.9 / 38.3 / −10.5 | 23.8 / 33.3 / −9.5 | 21.4 / 29.7 / −8.4 |
| tent (50k) | 39.2 / 43.3 / −4.1 | 26.3 / 33.4 / −7.0 | 23.3 / 30.0 / −6.8 | 20.6 / 28.7 / −8.1 |
| octopus (1.84M) | 33.4 / 36.9 / −3.5 | 28.4 / 34.4 / −6.0 | 24.3 / 31.0 / −6.7 | 21.7 / 28.5 / −6.8 |

1. **No free crossing from compressing harder**: the un-retrained model degrades FASTER than
   the rasterizer at every step (Δ widens by 2–4 dB from 20k→10k, then flattens). The RD
   framing only survives if retraining at each K recovers the model's 20k-level absolute
   PSNR; nothing here suggests it would, and it costs a full cycle per (object, K).
2. **The codec is specific to the splat REALIZATION, not the object**: a fresh 20k prune of the
   same object costs the model 2–4 dB (tent 43.2→39.2, octopus 35.6→33.4, seahorse 41.4→40.1,
   plate 37.0→36.4) while the rasterizer moves ≤1.2 dB. Per-object codec models overfit the
   exact token set they were trained on — a real caveat for any "train once, re-prune later"
   deployment story, and for eval protocols that regenerate inputs.
3. Studio scenes are only 50k splats, so 20k is a 2.5× "compression" — which is WHY their bars
   are 44–51 dB. Octopus at 20k is 92×. The crossover band is a property of the source scan
   density, consistent with everything since 2026-08-24.
Decision: K-sweep idea shelved (retrain-at-K unjustified). Stay on cycles for the three wild
scans (tent c3, slipper c2, octopus c3 running) + 1423/1223 c1s.

## 2026-08-27: TENT c3 −0.05, 25/40 — parity; rand crosses (+0.13, 16/24); c4 queued
tent c3 (31380320, 4×A40): rand **+0.13 (16/24)** / close **+1.69 (8/8)** / far −2.36 (1/8) =
**−0.05 avg, 25/40**. Cycle gains: +1.24 (c2) → +0.73 (c3) — decaying but not exhausted; far
(bar 46.5) is the only losing regime. First scale-out object to win a majority of views and the
random-view regime. c4 submitted (4-GPU killable) — expected to cross on avg.

## 2026-08-27: octopus c3 −1.21 (gain decaying); MOLECULE c1 −1.11 — a studio object in the band
octopus c3 (31383257): rand −1.95 (3/24) / close +0.49 (6/8) / far −0.70 (2/8) = **−1.21, 11/40**.
Gains +0.76 → +0.47: decaying; no c4 queued (would need ~3 more cycles at this rate).
molecule/scene_1423 c1 (31383249): rand −0.69 (8/24) / close **+0.75 (5/8)** / far −4.26 (0/8) =
**−1.11, 13/40** — second-best c1 in the campaign, and a STUDIO 50k object. Model 43.9/41.4/43.7
(highest absolute yet) vs bar 44.6/40.6/48.0. Revises the band: not "wild scans only" but
"moderate bar" — some studio objects (molecule, tent) sit there; seahorse/plate/sandal do not.
c2 31383251 running (seeded from c1 ep27). Far-range remains the universal deficit.

## 2026-08-27: doll (scene_1223) c1 −7.32 — bar-limited like plate/sandal
doll c1 (31383253): rand −7.64 / close −4.07 / far −9.60 = **−7.32 avg, 0/40**; bar 48.2/41.9/50.8.
Joins the high-bar studio group. c2 31383255 running.

## 2026-08-28: SLIPPER c2 **+0.48 dB, 23/40 — FIRST SCALE-OUT OBJECT TO BEAT rec-GT ON AVERAGE**
gopro/crocs slipper c2 (31374520, 4×A40 accum2, seeded from c1 ep27): rand −0.10 (11/24) /
close **+2.90 (8/8)** / far −0.20 (4/8) = **+0.48 avg, 23/40** at 60k steps. Gain +1.03 over c1
(−0.55). Model 35.7/31.1/38.2 vs bar 35.8/28.2/38.4. The tomato result (+1.30 after 5 cycles)
now reproduces on a second, independently sourced real scan after only two cycles — and with the
frozen scale-out recipe, no per-object tuning. Rand and far are at parity (−0.1/−0.2); close is
the model's regime everywhere (+2.9 here, +1.7 tent, +0.5 octopus). c3 submitted (4-GPU killable).
Standing: slipper +0.48 (c2) · tent −0.05 (c3, c4 running) · molecule −1.11 (c1, c2 running) ·
octopus −1.21 (c3) · apple −2.14 · vase −5.49 · doll −7.32 (c2 running) · plate/sandal/ring/figure
−7.0…−7.4 · seahorse −9.11.

### Slipper vs tomato at equal budget
| steps | tomato (codec3→5, tuned campaign) | slipper (frozen scale-out recipe) |
|---|---|---|
| 30k | — | −0.55 (14/40) |
| 60k | −1.38 | **+0.48 (23/40)** |
| 120k | +0.36 (first crossing, cycle 4) | c3 → 90k running |
| 150k | +0.97 | |
| 180k | +1.30 (34/40, final) | |
The slipper crosses ~2 cycles earlier than the tomato did, without per-object tuning. Why easier:
lower bar (35.8/28.2/38.4 — 60× compression of fuzzy fabric costs the rasterizer more than the
model) and a large close-range margin (+2.9). Open: whether it keeps climbing like the tomato
(+0.9 over its last three cycles) — c3 answers that.

## 2026-08-28 (eve): MOLECULE c2 **+0.08, 23/40 — second crossing, and a STUDIO object**
scene_1423 c2 (31383251, 4×L40S): rand **+0.49 (15/24)** / close **+1.97 (8/8)** / far −3.05
(0/8) = **+0.08 avg, 23/40** at 60k steps (c1 −1.11, gain +1.19). Model 45.1/42.6/45.0 — the
highest absolute PSNR of the campaign — vs bar 44.6/40.6/48.0. Second object over rec-GT within
a day, and the first from the synthetic-studio pool (50k splat, 2.5× compression): the band is
"moderate bar", not "wild scan". Far-range (bar 48) is again the only losing regime — now the
consistent pattern on every near-crossing object (tent, molecule, octopus, slipper): the model
wins close, ties rand, loses far. c3 submitted.
Scoreboard: slipper +0.48 (c2) · molecule +0.08 (c2) · tent −0.05 (c3) · octopus −1.21 (c3) ·
apple −2.14 · vase −5.49 · doll (c2 pending) · plate/sandal/ring/figure −7.0…−7.4 · seahorse −9.11.

## 2026-08-29: doll c2 −5.97 (gain +1.35, bar-limited); no c3
scene_1223 c2: rand −6.28 / close −2.63 / far −8.38 = **−5.97, 0/40** (c1 −7.32). Same ~+1.3 gain
as plate/figure; bar 48/42/51 keeps it out of reach. Not continued.

## 2026-08-29: TENT c4 **+0.30, 27/40 — third object over rec-GT**; c5 queued as the last tent cycle
scene_0223 c4 (31392272): rand **+0.45 (18/24)** / close **+2.05 (8/8)** / far −1.91 (1/8) =
**+0.30 avg, 27/40** at 120k steps. Cycle gains 1.24 → 0.73 → 0.35 (halving each cycle), so c5
is worth ~+0.15 more and is the last tent cycle. Three objects now over the bar (slipper +0.48,
tent +0.30, molecule +0.08), all with the same signature: close +2…+3, rand slightly positive,
far −2…−3. Far-range is the remaining frontier — its bar is the highest in every object.

## 2026-08-29 (pm): MOLECULE c3 **+0.75, 26/40 — best scale-out result so far**; c4 queued
scene_1423 c3 (31400001): rand **+1.20 (18/24)** / close **+2.55 (8/8)** / far −2.41 (0/8) =
**+0.75 avg, 26/40** at 90k steps (c2 +0.08, gain +0.67; c1→c2 was +1.19 — decaying at the same
~0.55× rate as tent). Model 46.5/43.2/45.6. Overtakes the slipper (+0.48 at 60k; c3 pending).
c4 submitted (expected ~+0.35 more).
Scoreboard: molecule +0.75 (c3) · slipper +0.48 (c2) · tent +0.30 (c4) · octopus −1.21 · apple −2.14
· vase −5.49 · doll −5.97 · plate/sandal/ring/figure −7.0…−7.4 · seahorse −9.11.

## 2026-08-30: SLIPPER c3 **+1.26, 30/40 — ALL THREE REGIMES POSITIVE, matches the tomato final at half the budget**
gopro c3 (31395334, 90k steps): rand **+0.70 (17/24)** / close **+3.76 (8/8)** / far **+0.46 (5/8)**
= **+1.26 avg, 30/40**. Gain +0.78 (c2 +1.03) — decaying slower than tent/molecule. First
object where far-range crosses (bar 38.4 is low enough). Equals the tomato's final +1.30 (180k
steps, 5 cycles) at 90k / 3 cycles with the frozen recipe. c4 submitted.
Scoreboard: slipper +1.26 (c3) · molecule +0.75 (c3, c4 running) · tent +0.30 (c4, c5 running) ·
octopus −1.21 · apple −2.14 · vase −5.49 · doll −5.97 · plate/sandal/ring/figure −7.0…−7.4 ·
seahorse −9.11.

## 2026-08-30: MOLECULE c4 **+1.18, 28/40**; c5 queued as the last molecule cycle
scene_1423 c4 (31406462, 120k steps): rand **+1.62 (19/24)** / close **+3.06 (8/8)** / far −2.01
(1/8) = **+1.18 avg, 28/40**. Gains 1.19 → 0.67 → 0.43; c5 (~+0.3) is the last. Far still loses
(bar 48.0) — the only near-bar object whose far bar is that high.
Scoreboard: slipper +1.26 (c3, c4 running) · molecule +1.18 (c4, c5 running) · tent +0.30 (c4, c5
running) · octopus −1.21 · apple −2.14 · vase −5.49 · doll −5.97 · plate/sandal/ring/figure
−7.0…−7.4 · seahorse −9.11.

## 2026-08-30 (noon): CAMPAIGN STANDING — 3/12 objects over rec-GT; all 12 through ≥2 cycles
End-of-cycle verdicts only (view-weighted avg dB vs rec-GT rasterization of the same 20k splat,
40 held-out views: 24 rand / 8 close / 8 far). Bars = rec-GT PSNR rand/close/far.

| object | source | bar (r/c/f) | c1 | c2 | c3 | c4 | best (won/40) | status |
|---|---|---|---|---|---|---|---|---|
| slipper (gopro) | superspl.at, 60× | 35.8/28.2/38.4 | −0.55 | +0.48 | **+1.26** | running | +1.26 (30) | c4 running |
| molecule (1423) | studio 50k | 44.6/40.6/48.0 | −1.11 | +0.08 | +0.75 | **+1.18** | +1.18 (28) | c5 running (last) |
| tent (0223) | studio 50k | 43.8/38.9/46.5 | −2.02 | −0.78 | −0.05 | **+0.30** | +0.30 (27) | c5 running (last) |
| octopus | superspl.at, 92× | 38.1/30.6/39.9 | −2.44 | −1.68 | −1.21 | — | −1.21 (11) | stopped (gain 0.47) |
| apple (0959) | studio | 50.2/46.0/53.1 | −2.18 | −2.14 | — | — | −2.14 (7) | saturated |
| vase (0874) | studio | 50.6/44.2/52.3 | −6.66 | −5.49 | — | — | −5.49 (0) | stopped |
| doll (1223) | studio | 48.2/41.9/50.8 | −7.32 | −5.97 | — | — | −5.97 (0) | stopped |
| plate (0262) | studio | 44.9/39.6/45.8 | −8.61 | −6.99 | — | — | −6.99 (0) | stopped |
| sandal (0772) | studio | 43.6/36.6/48.2 | −8.21 | −7.13 | — | — | −7.13 (0) | stopped |
| ring (1342) | studio | 46.0/38.0/49.0 | −8.13 | −7.18 | — | — | −7.18 (3) | stopped |
| figure (1078) | studio | 48.2/43.0/50.9 | −8.80 | −7.37 | — | — | −7.37 (0) | stopped |
| seahorse (0031) | studio 50k | 51.0/44.7/51.8 | −10.40 | −9.11 | — | — | −9.11 (0) | stopped |
| *tomato (ref.)* | superspl.at | — | — | −1.38 | — | +0.36 | +1.30 @c5 (34) | closed |

What the table says:
1. **Crossing = model absolute vs bar, and BOTH vary by object.** Bars range 43.6–51 (rand);
   model absolute ranges 36–48 (apple 48, molecule 46, tent 44, seahorse 41, plate 37, sandal 36).
   Apple (bar 50) sits at −2 because the model is excellent on it; plate (bar 45) sits at −7
   because the model is poor on it. So the earlier "model is flat, bar decides" reading was too
   strong: bar height explains seahorse/vase/doll/figure, model content-difficulty explains
   plate/sandal/ring (the fine-detail bandwidth limit from the V17 diagnosis). Objects cross
   when the two are within ~2.5 dB at c1.
2. **Cycle gains decay ~0.55×/cycle** (tent 1.24/0.73/0.35; molecule 1.19/0.67/0.43; slipper
   1.03/0.78) — each object has a ceiling ≈ c1 + 2.5 dB. That predicts who can cross from c1
   alone: c1 ≳ −2.5 → yes (slipper, molecule, tent, octopus-borderline); c1 ≲ −5 → never.
3. **Regime signature on every near-bar object: close +2…+4, rand ±1, far −2…−3.** Far loses
   because its bar is the highest (distant views hide splat artifacts); only the slipper, whose
   far bar is 38, has crossed there. Far-range is the open frontier.
4. The frozen recipe (v18-256 init, 9000 views, 30k steps/cycle, 4×bs1×accum2) reproduces the
   tomato's tuned result on three new objects without per-object work; the slipper reaches the
   tomato's final +1.3 at half the budget.
5. Negatives on record: K-sweep (no free crossing from harder compression; codec is realization-
   specific, −2…−4 dB on re-prune); more cycles cannot rescue bar-limited objects.
Ops: killable-only since 2026-08-23; lab share at 32 GB free (user chose to keep all data);
debian13 test nodes drained by the HC "fugitive" bug (reported); branch 26 commits ahead of origin.

## 2026-08-31: TENT c5 **+0.71, 28/40** — tent CLOSED at 5 cycles (150k steps)
scene_0223 c5 (31406005): rand **+0.87 (18/24)** / close **+2.41 (8/8)** / far −1.48 (2/8) =
**+0.71 avg, 28/40**. Gain +0.41 (c4 was +0.35 — the decay flattened rather than halving, so
the "ceiling ≈ c1 + 2.5" rule is conservative: tent is at c1 + 2.7). Trajectory −2.02 → −0.78 →
−0.05 → +0.30 → +0.71. Declared final per plan; a c6 would likely add ~+0.3 if ever wanted.
Final scale-out crossings: slipper +1.26 (c3, c4 running) · molecule +1.18 (c4, c5 running) ·
tent +0.71 (c5, closed).

## 2026-08-31: MOLECULE c5 **+1.47, 29/40 — best scale-out result; molecule CLOSED at 5 cycles**
scene_1423 c5 (31410702, 150k steps): rand **+1.91 (20/24)** / close **+3.26 (8/8)** / far −1.67
(1/8) = **+1.47 avg, 29/40**. Trajectory −1.11 → +0.08 → +0.75 → +1.18 → +1.47 (gains 1.19/0.67/
0.43/0.29 — clean ~0.6× decay). Model 46.5/43.9/46.3. Exceeds the tomato's final +1.30.
Ops: the lab share hit 0 GB during this eval (other members' writes) — eval + slipper c4 died with
no log (stdout unwritable); checkpoints verified intact; freed 12 GB by deleting the rebuildable
debian13 venv/uv-cache; both resubmitted (eval 31429211 → this verdict; slipper c4 31429212
resumes from ep21). Free space 8.8 GB — unsafe; user notified.
Final scale-out crossings: molecule +1.47 (c5, closed) · slipper +1.26 (c3; c4 running) · tent
+0.71 (c5, closed).
