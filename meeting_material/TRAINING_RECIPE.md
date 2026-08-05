# GaussianFormer — training recipe (technical rundown)

All values read from the code, not from notes. **V18** is the run in progress; **V17** is the
current best model and used the identical recipe with a longer schedule and a different GPU layout.

---

## Version lineage at a glance

| | **V14best** (May) | **V16+LPIPS** (Jul 12) | **V17** (Jul 23) | **V18** (running) |
|---|---|---|---|---|
| Train objects | 2,667 | 13,405 | 13,405 | **26,820** |
| Val objects | 183 | 902 | 902 | **1,806** |
| Input Gaussians | 20k, **naive-pruned** | 20k, pruned **+ recovered** | same | same |
| Views per epoch | **all 14** (full pass) | 4 of 14 | 4 of 14 | 4 of 14 |
| GPUs × batch size | 8 × 1 | 8 × 1 | 8 × 1 | **4 × 2** |
| Effective batch | 8 | 8 | 8 | 8 |
| Curriculum | 512² ×20 ep (L1) → 512² ×10 ep (LPIPS 0.2 tail FT) | 256² ×20 → 512² ×12 (L1) → 512² ×12 (LPIPS 0.5 tail FT) | 256² ×**60** → 512² ×**36** (LPIPS 0.5 **throughout**) | 256² ×**30** → 512² ×**20** (LPIPS 0.5 **throughout**) |
| Object-space mean PSNR | 24.88 dB | 26.94 dB | **28.49 dB** | *pending* |

Three things changed between V14 and V17, and they compound: **~5× more objects**, **pruning
recovery** on the inputs (+14.6 dB of input quality at the same N=20k), and **LPIPS applied
throughout the 512 stage** instead of bolted on as a tail fine-tune. V18 changes exactly one
variable against V17 — **2× the data** — but see soft spot #2 on the schedule confound.

Note the views-per-epoch change: V14's epochs were full 14-view passes, so it got ~30 passes over
its 2,667 objects. V16 onward sample 4 of 14, which is why "more epochs" was not the same as "more
exposure per object-view."

---

## Model

Adapted from **RenderFormer** (`microsoft/renderformer-v1-base`); weights transferred, the
triangle-mesh input module replaced with a Gaussian one.

| | |
|---|---|
| Input token | 14-dim: `[pos(3), scale(3), quat_wxyz(4), color(3), opacity(1)]` |
| Gaussians per object | **N = 20,000** (fixed) |
| View-independent encoder | 12 layers, d=768, 6 heads, FFN 3072, SwiGLU, RMSNorm, pre-norm, QK-norm, no bias, 16 register tokens, dropout 0 |
| View-dependent transformer | 6 layers, d=768, 6 heads, FFN 3072; ray-token self-attn + cross-attn to scene tokens; patch size 8 |
| Decoder | DPT, 128 internal features |
| Positional encoding | **RoPE** on positions (`pe_type='rope'`), 12 frequencies |
| Attention kernel | FlashAttention (auto-falls back to SDPA) |

## Data

| | V18 (current) | V17 |
|---|---|---|
| Train objects | **26,820** | 13,405 |
| Val objects | **1,806** | 902 |

- Each object: original ~50k-Gaussian splat **pruned to 20k** (LightGaussian significance score)
  **plus a gsplat Adam recovery fine-tune** of the survivors (~1500 iters). Recovery is worth
  **+14.6 dB** of input quality at the same N.
- **14 ground-truth views** per object, rendered at 512² with gsplat.
- `--views_per_epoch 4`: each epoch samples **4 of the 14 views** per object.
  → epoch = 26,820 × 4 = **107,280 samples** (13,410 optimizer steps at effective batch 8).
- Validation uses **all 14 views** → 1,806 × 14 = 25,284 samples.
- Stage A downscales the 512 renders to 256 (bilinear); stage B uses them natively.

## Augmentation

**Rotation augmentation only.** One Haar-uniform random 3D rotation per sample, applied jointly to
the scene *and* the camera so the target image is unchanged: Gaussian means rotate, each Gaussian's
orientation world-rotates, and `c2w` rotates. Scale/colour/opacity are rotation-invariant.
No colour, exposure, or view jitter.

## Optimisation (identical in both stages)

| | |
|---|---|
| Optimiser | **AdamW**, `weight_decay=0.01`, default betas (0.9, 0.999) |
| LR schedule | **CosineAnnealingLR**, `T_max = phase2_epochs`, `eta_min = lr × 0.01` |
| Base LR | 5e-5 → anneals to **5e-7** |
| Scheduler granularity | **per epoch** (not per step) |
| Gradient clipping | global norm **1.0** |
| Precision | **bfloat16** autocast (no GradScaler — bf16 doesn't need one) |
| Warmup | **none** |

## Two-phase structure (and why phase 1 is skipped)

The code has a phase 1 (freeze the RenderFormer backbone, train only the Gaussian input module,
lr 1e-3) followed by phase 2 (unfreeze everything, lr 5e-5).

**V17 and V18 skip phase 1.** Both `--init_from checkpoints_v15_256/phase1_epoch_5.pt` — a
known-good warmup — and go straight to a clean phase 2. This was a deliberate fix: V16's own
phase 1 silently failed to train (val 0.0135 vs V15's 0.0038) and starved its phase 2. The
`--phase1_*` flags in the scripts are therefore inert.

## Curriculum

### Stage A — 256², bulk geometry/colour
| | V18 | V17 |
|---|---|---|
| Epochs | **30** | 60 |
| Resolution | 256 | 256 |
| Loss | **pure log-L1** (`log_w 1.0`, `lpips_w 0.0`) | same |
| LR | 5e-5 cosine → 5e-7 | same |

### Stage B — 512², perceptual refinement
| | V18 | V17 |
|---|---|---|
| Epochs | **20** | 36 |
| Resolution | 512 | 512 |
| Loss | **`log_w 0.5` + `lpips_w 0.5`, from epoch 1** | same |
| Init | from stage A's final checkpoint | same |
| LR | 5e-5 cosine → 5e-7 (**fresh restart**) | same |

**Loss definitions.** The model predicts in `log10(hdr + 1)` space.
- log term: `L1(pred, log10(target + 1))`
- LPIPS term: LPIPS-VGG on display space, `pred_ldr = clamp(10^pred − 1, 0, 1)`, mapped to [−1, 1]

`lpips_w = 0.5` was settled by a sweep: 0.5 ≥ 1.0 on **both** PSNR and LPIPS on **every** test
scene. LPIPS is applied *throughout* stage B rather than as a tail fine-tune — that change alone
was part of V17 beating V16.

## Distributed / infrastructure

| | V18 | V17 |
|---|---|---|
| Layout | **4 GPUs × batch 2** | 8 GPUs × batch 1 |
| Effective batch | **8** | 8 |
| Launcher | `torchrun --standalone`, DDP | same |
| Dataloader | 8 workers/rank | same |

**There is no gradient accumulation in the code**, so effective batch = #GPUs × batch_size. V18
uses batch 2 on 4 GPUs specifically to preserve V17's effective batch of 8. Checkpoints are written
every epoch and runs are resume-safe (V17 survived two preemptions with zero work lost).

## Evaluation

- Per-epoch validation on held-out objects, same loss terms as the active stage.
- Model selection / reporting: **foreground-cropped** PSNR + LPIPS on unseen objects with
  `--input_mode recovered`. Whole-image PSNR is not used — objects are 2–5% of pixels, so it
  mostly measures the black background.

---

# Known soft spots — the things I'd expect to be challenged

Listed deliberately so Sagie can go straight at them.

1. **Stage B restarts the cosine at full LR (5e-5)** on an already-converged stage-A model, rather
   than continuing the annealed LR or warming up. This is a warm restart, and it demonstrably
   causes a transient regression — V17 measured **1.5 dB *behind* the previous best at epoch 6** and
   finished **1.55 dB ahead** at epoch 36. Intentional? Or would a lower stage-B LR / short warmup
   get the same endpoint sooner and more safely?

2. **`T_max` = the epoch budget, and the scheduler steps per epoch.** Changing the budget changes
   the entire LR trajectory, so runs of different lengths are not directly comparable — V18's
   30-epoch stage A anneals much faster than V17's 60-epoch one. This confounds "did the data help?"
   with "did the schedule change?"

3. **No gradient accumulation** — effective batch is welded to GPU count. Any change in allocation
   forces either a batch-size change or a confounded comparison.

4. **The phase-1 warmup we initialise from is stale.** It comes from V15, trained at 256 on the
   *1× dataset*. V18 is 2× the data and two model generations later, and still starts from it.

5. **An "epoch" is a sampling unit, not a pass over the data.** With 4 of 14 views, per-object-view
   exposure is much lower than the epoch count suggests — V16 saw each object-view less than half as
   often as V14 did, on 4.5× more objects. Under-training and the "generalisation gap" may be the
   same phenomenon.

6. **The LPIPS term clamps.** `clamp(10^pred − 1, 0, 1)` produces exactly zero gradient wherever the
   prediction falls outside the display range — saturated highlights and any negative excursion get
   no perceptual signal at all. Unquantified; possibly harmless, possibly not.

7. **No perceptual signal during stage A.** Its validation is log-L1 only, so there's nothing to
   early-stop or model-select on for the metric we actually care about, across the majority of
   training.

8. **`weight_decay=0.01` is applied to every parameter**, including RMSNorm weights (biases are
   already disabled). Standard practice excludes norms from decay.

9. **Rotation is the only augmentation.**

10. **The pruning score is biased against fine detail** — `opacity × screen area × max_scale^0.1`
    preferentially keeps large translucent floaters and discards small high-opacity detail
    Gaussians. Covered in detail in `TALKING_POINTS.md`; currently the leading hypothesis for the
    unsolved high-frequency fidelity gap.
