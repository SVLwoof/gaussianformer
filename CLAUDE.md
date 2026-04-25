# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository adapts [RenderFormer](https://github.com/microsoft/renderformer) (SIGGRAPH 2025) for **3D Gaussian Splatting** inputs. RenderFormer is a transformer-based neural renderer: it takes triangle-mesh scenes and outputs rendered images without per-scene training. **GaussianFormer** replaces the triangle-mesh input with Gaussians (position, scale, rotation, color, opacity) while keeping the same two-stage transformer architecture.

The project is at an early stage: the model architecture (`gaussianformer/`) has been adapted but is **not yet trained**. The current focus is generating high-quality scene descriptor JSONs that can be:
1. Rendered through the original RenderFormer mesh pipeline to produce ground-truth images
2. Used later as training data for GaussianFormer

## Environment Setup

```bash
pip install -r requirements.txt
python3 -c "import imageio; imageio.plugins.freeimage.download()"  # Needed for HDR image IO
```

Uses `uv` for dependency management (`pyproject.toml` + `uv.lock`). Flash Attention is optional; code falls back to SDPA automatically. Force SDPA with `ATTN_IMPL=sdpa`.

## Key Commands

### RenderFormer (mesh pipeline) -- use this to verify scene quality

**Single scene: JSON -> H5 -> rendered image:**
```bash
python3 scene_processor/convert_scene.py examples/cbox.json --output_h5_path tmp/cbox/cbox.h5
python3 infer.py --h5_file tmp/cbox/cbox.h5 --output_dir output/cbox/
```
See `render-images.sh` and `render-videos.sh` for full examples with tone mappers.

**Batch (video frames):**
```bash
python3 batch_infer.py --h5_folder <folder> --output_dir <output> --save_video
```

### Scene generation for training data

**Step 1 -- Generate randomized scene descriptor JSONs:**
```bash
python3 generate_training_data.py
```
Outputs to `training_examples/`. Edit `NUM_SCENES_TO_GENERATE`, camera/light ranges, and material definitions directly in the script.

**Step 2 -- Convert mesh examples to Gaussian format (OBJ -> PLY + JSON rewrite):**
```bash
python3 batch_convert_to_gaussian_examples.py
```
Reads `training_examples/` -> writes `gaussian_training_examples/`.

**Step 3 -- Compile Gaussian scenes to HDF5:**
```bash
python3 gaussian_scene_processor/batch_generate_h5.py
```
Reads `gaussian_training_examples/` -> writes `gaussian_training_h5s/`.

### GaussianFormer inference (WIP, model not yet trained)

```bash
python3 create_local_model.py                # Create random-weight model for testing
python3 infer_gaussian.py --h5_file path/to/scene.h5 --model_id ./my-gaussianformer-model
```

## Architecture

### Pipeline flow

```
Scene JSON descriptor
  |-- scene_processor/convert_scene.py --> H5 (triangles) --> infer.py --> rendered image
  |-- batch_convert_to_gaussian_examples.py --> Gaussian PLY + JSON
        |-- gaussian_scene_processor/batch_generate_h5.py --> H5 (gaussians) --> infer_gaussian.py --> rendered image
```

### Dual-package structure

| Package | Input | Scene processor | Inference |
|---|---|---|---|
| `renderformer/` | Triangle meshes (textured) | `scene_processor/` | `infer.py`, `batch_infer.py` |
| `gaussianformer/` | 3D Gaussians (14-dim) | `gaussian_scene_processor/` | `infer_gaussian.py` |

Both share identical internal structure: `encodings/`, `layers/`, `models/`, `pipelines/`, `utils/`.

### GaussianFormer model

Two-stage transformer:

1. **View-independent** (`GaussianFormer.transformer`): `TransformerEncoder` processes all Gaussians as a sequence. Each Gaussian is a 14-dim vector: `[pos(3), scale(3), rotation_quat(4), color(3), opacity(1)]`. Default positional encoding is RoPE.

2. **View-dependent** (`ViewTransformer`): Cross-attention from ray tokens to scene tokens, producing pixel patches. Uses DPT decoder for upsampling.

Key types:
- `GaussianFormerConfig` -- frozen dataclass, all hyperparameters
- `GaussianFormer` -- nn.Module + PyTorchModelHubMixin
- `GaussianFormerRenderingPipeline` -- wraps model with ray generation + coordinate transforms
- `transform_gaussians_to_cam_coord()` -- rotates Gaussian positions + quaternions into camera space

### Scene descriptor JSON format (mesh pipeline, v1.0)

Scenes are JSON files referencing `.obj` meshes from `examples/objects/` and `examples/templates/`. Structure:
- `objects`: dict of named objects, each with `mesh_path`, `material` (diffuse, specular, roughness, emissive, smooth_shading, rand_tri_diffuse_seed/max/type), `transform` (translation, rotation as Euler degrees, scale, normalize)
- `cameras`: list of cameras with `position`, `look_at`, `up` (Z-up), `fov`
- Backgrounds use `templates/backgrounds/{plane,wall0,wall1,wall2}.obj` at scale [0.5,0.5,0.5]
- Lights use `templates/lighting/tri.obj` with high emissive values

### RenderFormer training-data constraints (from README)

When generating scene descriptors, stay within these ranges for best results:
- Camera distance to scene center: [1.5, 2.0], fov: [30, 60] degrees
- Scene bounding box: [-0.5, 0.5] in x, y, z
- Light sources: up to 8 triangles (using `tri.obj`), scale [2.0, 2.5], distance to center [2.1, 2.7], emission sum [2500, 5000]
- Total triangles: up to 4096 (8192 usually still works at inference)
- Objects should be water-tight, uniform triangle sizes preferred

### Gaussian scene format (v1.1-gaussian)

References `.ply` files instead of `.obj`. Transform uses quaternion rotation (w,x,y,z) instead of Euler angles. Simplified material: `color_tint` + `opacity_multiplier`.

HDF5 fields: `means [N,3]`, `scales [N,3]`, `rotations [N,4]` (w,x,y,z), `colors [N,3]`, `opacities [N,1]`, `c2w [V,4,4]`, `fov [V]`.

Camera convention: Blender coordinate system (-Z = view direction, +Y = up, +X = right).

## Available assets

- **Templates**: `plane.obj`, `wall0.obj`, `wall1.obj`, `wall2.obj` (backgrounds), `tri.obj` (light)
- **Objects**: bunny, teapot (classical); cbox tall/short box; tree; fox + rock + tree-leaves/trunk; horse + hearts; lucy (3k/6k/11k); crystals (5 colors); shader-ball; compose objs; constant-width; room furniture; veach-mis; renderformer-logo
