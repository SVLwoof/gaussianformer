# Single-object overfit — capacity probe

**Question.** Is the model (current size/config, N=20k) even *capable* of representing
high-frequency texture/colour detail? Or is the residual blur in every version a hard
architectural ceiling (most likely the DPT decoder's band-limited upsampling)?

**Method.** Train the model *exclusively* on ONE object's 14 views (augmentation OFF), long
enough to overfit, and measure the converged reconstruction. Overfitting removes
generalisation from the equation, so whatever the model still can't reproduce *is* the
capacity ceiling. We decompose the quality gap:

```
real-GT  --(pruning loss: lost by N=20k)-->  pruned-GT  --(model loss)-->  model output
```

| Converged overfit reaches | Verdict | Implied next move |
|---|---|---|
| **pruned-GT** (or real-GT) | architecture *can* render the detail | bottleneck is data/N → **scale up** |
| plateaus **below pruned-GT** | architecture/decoder is the ceiling | **change architecture** (DPT patch_size, decoder) |
| RF-base low **but V14best higher** | optimisation-limited, not capacity | better init/schedule, not a rewrite |

The **model-vs-pruned-GT** number is decisive: pruned-GT is what the model's *own* 20k-Gaussian
input can render, so failing to match it on a single memorised object indicts the architecture.

## Matrix (4 killable single-GPU jobs)

```
boxes  (Objaverse scene_1441)  × { base , v14best }
tomatoes (external scan)       × { base , v14best }
```

- **base** = RenderFormer transfer → Phase 1 (encoder warmup) → Phase 2 (joint).
- **v14best** = warm-start weights from `checkpoints_v14auglp10/phase2_epoch_10.pt`
  (`--init_from` auto-skips Phase 1). Confirms a low base ceiling is capacity, not optimisation.

Both objects are texture/colour-rich and well-source-fit (detail genuinely lives in the 20k).
tomatoes is the hero benchmark we've judged across V8–V14 (real captured texture).

## Run

```bash
experiments/overfit/setup_data.sh           # symlink-only; no copies, no rendering
for OBJ in boxes tomatoes; do for INIT in base v14best; do
  sbatch --job-name=ovf_${OBJ}_${INIT} \
         --export=ALL,OBJ=$OBJ,INIT=$INIT experiments/overfit/run_overfit.sh
done; done
```

Eval a checkpoint (three-way numbers + real-GT|pruned-GT|model strip):

```bash
uv run --frozen python -m experiments.overfit.eval_overfit \
  --ckpt experiments/overfit/ckpt/boxes_base/phase2_epoch_1500.pt \
  --h5 experiments/overfit/data/boxes/h5s/scene_1441.h5 \
  --realgt_dir data_v9/renders --label boxes_base \
  --out_dir experiments/overfit/eval
# tomatoes: --h5 .../tomatoes/h5s/tomatoes_n20000.h5  --realgt_dir .../tomatoes/renders
```

## Design notes
- **Augmentation OFF, in-train val skipped** — both would only slow the memorisation we're measuring.
- **bs=1, single GPU** — 14 samples; multi-GPU/large batch is meaningless, and bs=1 = most update steps.
- **Recipe = the working two-phase** (warmup → joint) so RF-base avoids the 0.0138 plateau.
- `data/` and `ckpt/` are git-ignored (symlinks to large data + checkpoints).
- Natural follow-up if log-L1 overfit plateaus *blurry*: an LPIPS-loss variant, to separate
  "loss can't push high-freq" from "architecture can't represent high-freq".
