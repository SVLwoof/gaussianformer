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
