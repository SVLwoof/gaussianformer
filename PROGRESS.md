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
