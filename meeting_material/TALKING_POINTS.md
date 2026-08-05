# GaussianFormer — progress since late June

Meeting with Sagie, Sun 2026-08-02. Figures: `fig1_generations.png`, `fig2_detail.png`.

> **On the figure numbers:** every LPIPS/PSNR printed in the figures is computed **on the crop
> shown**, using the same convention as our eval path (`model_on_v10.py:_fg_crop` — square box
> from the GT object, resized NEAREST so interpolation can't flatter a blurry render). They are
> *not* whole-image numbers. Note the figure objects are a different, easier set than the three
> hard test objects in the table at the bottom, so the absolute values are higher — don't
> cross-quote the two.
>
> **A ready-made demonstration of point 1:** on the mountain diorama, the May model scores
> **20.8 dB whole-image but only 12.5 dB in object space**. Same render, same ground truth — an
> 8 dB illusion produced purely by counting the black background. That single pair of numbers
> makes the metric argument faster than any explanation.

---

## High-level: the four things that happened

**1. We found out our main metric was lying to us.**
Renders looked soft even while whole-image PSNR was climbing. The objects sit on a black
background and occupy only **2–5 % of pixels** (the skull: 1.7 %), so whole-image PSNR was
95–98 % "successfully matched the black" — inflated, and blind to whether the object itself is
sharp. We switched all evaluation to **foreground-cropped** metrics. Object-only, the model we'd
been calling "+2.6 dB better" turned out to have a **~13 dB gap** to its own input ceiling that the
metric had hidden. This reframed the whole campaign, and it's probably the most transferable
lesson of the last month.

**2. We proved the blur is a generalization failure, not an architecture limit.**
Overfitting a *single* detailed object reproduces its fine engravings faithfully
(LPIPS 0.022, vs 0.311 for the general model). So the network at N=20k Gaussians **can**
represent and render that detail — the general model simply fails to learn a sharp mapping that
transfers to unseen objects. That told us to stop redesigning the architecture and work on
training and data instead.

**3. "Train longer" was a real, cheap win — we had been under-training badly.**
Every stage of the previous model was still improving monotonically at its final epoch: we'd cut
at the compute budget, not at convergence. V17 = same recipe, much longer, with the perceptual
loss folded in from epoch 1 instead of bolted on as a short fine-tune at the end. Result: **new
best model on every axis.** Object-space mean PSNR **24.88 → 28.49 dB** since the May model, with
LPIPS improved on every test object.

**4. We doubled the dataset and started the next retrain.**
Because of where the remaining error lives (below), the next lever is data, not training. The
dataset went from 13,405 to **26,820 training objects** (val 902 → 1,806), all pruned to 20k
Gaussians and fine-tuned. V18 — the same V17 recipe on 2× data — is training now.

---

## The open problem (the honest part, and the best discussion topic)

**Faithful high-frequency detail is still unsolved, and it is no longer an epochs problem.**

On the hardest test object (a carved skull), V17 renders *sharper, more convincing* carving —
but it's **invented** carving, not the object's actual engraving. Both levers we had are now
spent on this one failure mode:

- perceptual loss: skull LPIPS 0.311 → 0.128
- all of V17's extra epochs: 0.128 → 0.110

Meanwhile the input Gaussians hold the true engraving at ~40 dB, so **the information is present
and the model is discarding it.** That's why the 2× data scale-up is the current bet, and it's
the thing I'd most like Sagie's read on — whether uniform scaling is right, or whether we should
be *curating for* high-frequency surface detail specifically.

---

## Technical points worth raising

- **Foreground-cropped evaluation** (`--crop_fg`) is now standard. Related trap we hit: the
  naive-pruned "ceiling" is meaningless — the old model actually **scored above its own input
  ceiling** on the skull, because naive-pruned input is so degraded that a smooth blob lands
  closer to ground truth than the input render does. We discount every naive-ceiling comparison.

- **Pruning recovery was the single biggest input-quality win.** We had been doing
  LightGaussian's score-and-prune but skipping its recovery step. Adding it (prune 50k→20k, then
  gsplat Adam fine-tune the survivors) took held-out pruned input from **35.0 → 49.5 dB
  (+14.6 mean)** at the *same* N=20k.

- **But recovered input only helps if you train on it.** Feeding the recovered Gaussians to the
  old model made it slightly *worse* (24.88 → 24.55). The gain came from retraining on the new
  distribution, not from nicer inputs at eval time. Good cautionary result.

- **Perceptual-loss weight is settled at 0.5.** A sweep at 0.5 vs 1.0 (same log-L1 anchor, only
  the weight differing) found 0.5 ≥ 1.0 on *both* PSNR and LPIPS on *every* scene — pushing
  harder is pure PSNR cost with no perceptual return.

- **Don't judge a perceptual-loss run early.** V17 read **1.5 dB behind** the previous best at
  epoch 6, and finished **+1.55 dB ahead** at epoch 36. There's a documented over-shoot window
  when the cosine schedule restarts; nothing before ~epoch 15 is meaningful. We nearly killed a
  winning run on an early reading.

- **Breadth evidence, not just three objects.** Across 200 unseen objects × 4 views (800 renders),
  **797/800 improve strictly monotonically** V14 → V16 → V17. Mean LPIPS 0.0199 → 0.0112 →
  0.0076. *(Whole-image numbers — inflated in absolute terms per point 1, so quote them as
  consistency/breadth evidence only; the foreground-cropped table below is the defensible one.)*

- **Infrastructure lesson worth one sentence:** never quote min/epoch without naming the node. The
  GPU pool is heterogeneous behind a single resource label (45 GB L40S/A40 through 96 GB
  RTX Pro 6000), and identical configs measured **3.5× apart** across node types. This also just
  bit us concretely: V17's final stage ran on a 96 GB node, so its batch configuration was never
  validated on the 45 GB nodes — and it does not fit there.

---

## Where V18 actually is (be precise, it isn't a result yet)

- Stage A (256 px, geometry/color, log-L1 only): **epoch 7 of 30**, validation descending
  cleanly, no preemptions. ETA **~Tue Aug 4**.
- Stage B (512 px with perceptual loss, 20 epochs) follows. **This is the stage that produces the
  first comparable quality number** — stage A optimizes log-L1 only, which is exactly the metric
  we've learned not to trust.
- So on Sunday the honest status is "**on track and converging as expected**", not a result. First
  real V18 vs V17 comparison is realistically ~1.5–2 weeks out.

---

## Asks / open questions for Sagie

1. **GPUs.** Stage B needs either 8 GPUs (matching V17's exact configuration, cleanest possible
   comparison) or a compromise on 4. The training code has no gradient accumulation, so GPU count
   directly sets the effective batch size — on 4 GPUs we'd either change the effective batch
   (a confound against V17) or add accumulation. Worth 5 minutes: is more allocation available?
2. **Data strategy for high-frequency fidelity.** Is uniform 2× scaling the right bet, or should
   we filter/curate the training set for objects with genuine fine surface detail?
3. **The N=20k Gaussian budget.** A side finding: N=20k itself caps detailed scans — a detailed
   object's pruned ceiling is ~29.6 dB vs ~37.6 for a simple one. Should N scale with object
   complexity rather than being fixed?
4. **NEW — our pruning criterion may be selecting against fine detail.** Found while explaining a
   grey smudge in one figure (see below). This is the most concrete lead on the high-frequency
   problem, and I'd like a view on whether to test it before or after V18 lands.
5. **Write-up timing** relative to the V18 result.

---

## New finding: the pruning score is biased against fine detail

Chasing a soft grey streak visible under the mountain diorama turned up something structural.

**What the streak is:** 83 large, faint, whitish Gaussians floating *below and in front of* the
object — horizontal scale 0.07–0.10 against a median of 0.0043 (≈20× oversized), opacity 0.11–0.23,
at depth 1.56–1.61 versus the object's 1.7–1.8. It's a classic 3DGS **floater**: the original
reconstruction parked translucent blobs in empty space, typically where a ground shadow was.
Inherited from Objaverse_Splats, present in the ground truth, faithfully reproduced by V17 (V14
smears it). Also found in the same splat: **30 pure-black flat sheets with ~918 px on-screen radius
in a 512 px frame** — a backdrop plate owning ~86% of the splat's footprint-weighted mass.

**Why it matters.** Our importance score (`prune_recovery.py:58`, from LightGaussian) is

    score = sum_views [ opacity x radius_x x radius_y ] x max_scale^0.1

i.e. **opacity x projected screen area**. A floater with a 200 px radius at opacity 0.15 outscores a
crisp 2 px detail Gaussian at opacity 0.9 by ~3 orders of magnitude. So the 50k->20k selection
structurally *prefers* translucent floaters and backdrop plates and *discards* the small,
high-opacity Gaussians that carry fine surface detail — which is exactly the failure mode that
neither perceptual loss nor longer training could fix.

Consistent (but not conclusive) evidence: the retained set's median horizontal scale is 2.4x the
original, 0.0043 -> 0.0105. Confounded, because the recovery fine-tune also legitimately grows
scales when fewer Gaussians must cover the same surface.

In fairness, LightGaussian's criterion was designed for whole-scene compression, where large
background Gaussians genuinely do carry the image. For single-object, detail-critical rendering the
area weighting works against us.

**Proposed test:** cap or reweight the area/scale terms, and/or filter floaters (large scale + low
opacity + isolated in empty space) *before* pruning; then re-measure the skull's engraving against
the recovered-input ceiling (~40 dB, so the information is provably there). Cheap to try — it
touches data preparation only, no model or training change.

---

## Numbers table (object-space, foreground-cropped, 3 unseen objects)

| model | statue | skull | honeypot | mean PSNR |
|---|---|---|---|---|
| V14best (May) | 22.6 / 0.061 | 26.5 / 0.183 | 25.5 / 0.091 | **24.88** |
| V16 + LPIPS-mid (Jul 12) | 27.3 / 0.030 | 26.5 / 0.128 | 27.0 / 0.067 | **26.94** |
| **V17 ep36 (Jul 23)** | **30.5 / 0.020** | **26.5 / 0.110** | **28.5 / 0.044** | **28.49** |

*(cells are PSNR dB / LPIPS)*

Supporting convergence numbers: V17 stage A validation log-L1 **0.000765** vs V16's 0.001034
floor (−25 %); V17 stage B perceptual validation term 0.02061 → **0.01397** (−22 % vs V16),
descending monotonically to a cosine LR of 5e-7.
