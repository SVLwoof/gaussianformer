# CLAUDE.md

Guidance for Claude Code in this repository.

## Overview

GaussianFormer adapts [RenderFormer](https://github.com/microsoft/renderformer) (SIGGRAPH 2025), a transformer
renderer for triangle meshes, to 3D Gaussian Splatting input. Each Gaussian `[pos(3), scale(3), quat wxyz(4),
rgb(3), opacity(1)]` is one scene token; the two-stage architecture is warm-started from RenderFormer's weights.

- Released model: `shahafvl/gaussianformer-v17b` (default `--model_id` of `infer_gaussian.py`); earlier:
  `shahafvl/gaussianformer-v10b`.
- Current work: full-data run of the settled recipe (below); experiment log in `PROGRESS.md` (append-only, newest at
  the bottom).

## Environment

`uv` (`pyproject.toml` + `uv.lock`) is the source of truth.

```bash
uv sync
uv run python -c "import imageio; imageio.plugins.freeimage.download()"  # HDR I/O
```

Flash Attention is optional (SDPA fallback, force with `ATTN_IMPL=sdpa`), but windowed cross-attention needs it on
GPU. In SLURM jobs use `uv run --no-sync` (parallel `uv run` calls re-sync the shared venv).

## Layout

| Path | Contents |
|---|---|
| `gaussianformer/` | model: `models/` (config, GaussianFormer, ViewTransformer), `layers/` (attention, window, DPT, LoRA), `pipelines/`, `utils/` |
| `renderformer/`, `scene_processor/`, `infer.py` | upstream mesh pipeline, kept for reference |
| `training/` | `train.py` (two-phase trainer of the released models), `train_full.py` (warmup-stable-decay trainer, current), `train_lora.py` (per-object adapters), `dataset.py` |
| `data_v10/` | data generation, evaluation, probes, render scripts, and the data itself |
| `tests/` | CPU tests: `ATTN_IMPL=sdpa PYTHONPATH=. uv run --no-sync python tests/<file>.py` |
| `docs/report/` | render sheets and report figures; `medias/` holds README figures |

## Model

`GaussianFormerConfig` (`gaussianformer/models/config.py`) holds every architecture option; checkpoints store it,
and `--model_cfg key=value ...` overrides it in training and evaluation.

- Scene encoder: 12 layers over one token per Gaussian (Linear over the 14 parameters), 3-D RoPE on positions.
- View decoder (`ViewTransformer`): one token per image patch built from ray directions; 6 layers of patch
  self-attention plus cross-attention to scene tokens, DPT head to pixels. Output is log-HDR.
- Options used by the current recipe: `proj_rope_2d` (2-D RoPE on each Gaussian's projected image position),
  `xattn_window` (tile-windowed cross-attention), `patch_size=4` + `ray_embed_patch=8`, `view_bf16`,
  `view_grad_checkpoint`.
- New parameters must be zero-initialised so a seed checkpoint loads unchanged (`load_seed` asserts this); changes to
  pretrained behaviour need a 256 px recovery stage first.

## Data (`data_v10/`)

- `process_full.py`: Objaverse_Splats objects -> normalized full splats + 14 ground-truth views (r 1.7, 45° FOV, 512 px).
- `prune_recovery.py`: 20k-Gaussian inputs -> `h5s_20k_rec/` (train, 26,820 objects), `h5s_20k_rec_val/` (1,806).
- `multi_radius_full.py`: + 7 views each at r 1.15 and 2.45 -> `h5s_20k_rec_r3/` + `renders_r3/` (28 views/object;
  H5s are external links to `h5s_20k_rec/`, base views are hardlinks to `renders/`).
- `nsweep/`: fixed small sets (n10 train objects, val100, heldout300) for probes.

HDF5 fields: `means [N,3]`, `scales [N,3]`, `rotations [N,4]` (w,x,y,z), `colors [N,3]`, `opacities [N,1]`,
`c2w [V,4,4]`, `fov [V]`. Cameras use the Blender convention (-Z forward, +Y up, +X right).

## Training and evaluation

- Full-data run: `sbatch -A sagieb data_v10/train_full.sh` (stage R at 256 px, then 512 px; resumes automatically;
  `touch <save_dir>/FREEZE` checkpoints and exits within 20 steps).
- Architecture probes: `data_v10/probe_n10.sh` (10 objects, 30k steps, 4 GPUs) -> `data_v10/run_nsweep_eval.sh` ->
  `data_v10/probe_report.py`. Compare arms only at the same GPU count and schedule. Patch-2 models need
  `--view_chunk 1` in `ceiling_eval.py`.
- Evaluation metric: PSNR on the object's bounding box against the full-splat render (`ceiling_eval.py`); "margin" =
  rasterized input minus model, lower is better.
- Renders: `data_v10/report_renders.sh` (comparison sheets), `render_video.sh` (orbit/dolly/object-motion clips),
  `render_pair.sh` (two objects in one scene).

Settled recipe: `proj_rope_2d` + `xattn_window=8` + `patch_size=4`/`ray_embed_patch=8` + `view_bf16` +
`view_grad_checkpoint`, multi-distance views, `--fg_bg_weight 0.05`, log-L1 and LPIPS at 0.5 each.

## Cluster (HUJI SLURM)

- Every `sbatch`/`srun` needs an account: `-A sagieb` for training; evals, renders and data jobs run
  `--killable --account=killable-cs`.
- Scripts are `#!/bin/zsh`, source `/etc/profile.d/huji-lmod.sh`, then `module load nvidia` and `module load cuda`. Never
  `sbatch --wrap` (runs `sh`). Pass `--export=PATH,HOME,USER,...` explicitly, not `ALL`.
- gsplat compiles CUDA on first import; set `TORCH_EXTENSIONS_DIR` per GPU architecture.
- `gg:g4` nodes: 17 L40S, 2 A40 (epona-01/02, ~1.7x slower), 2 RTX Pro 6000 (khan-01/02, untested with this stack),
  1 A6000 (cyril-01). All have 8 GPUs.
- The lab share `/cs/labs/sagieb` is shared and often near full; check `df` before writing large outputs and never
  delete data or checkpoints without the user's approval.
