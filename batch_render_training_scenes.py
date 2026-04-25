"""
Batch pipeline: convert all training scene JSONs to H5, then render them
with RenderFormer.

Usage:
    python3 batch_render_training_scenes.py
    python3 batch_render_training_scenes.py --skip_convert    # if H5s already exist
    python3 batch_render_training_scenes.py --tone_mapper agx
    python3 batch_render_training_scenes.py --max_scenes 5    # quick test
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import h5py
import imageio

from simple_ocio import ToneMapper


# ─── Directories ───────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).parent
TRAINING_EXAMPLES_DIR = ROOT_DIR / "training_examples"
H5_OUTPUT_DIR = ROOT_DIR / "training_h5s"
RENDER_OUTPUT_DIR = ROOT_DIR / "training_renders"
CONVERT_SCRIPT = ROOT_DIR / "scene_processor" / "convert_scene.py"


# ─── Step 1: Convert JSONs to H5 ──────────────────────────────────────────────

def convert_all_scenes(json_dir: Path, h5_dir: Path, max_scenes: int | None = None) -> list[Path]:
    """Convert scene JSONs to H5 via subprocess calls to convert_scene.py."""
    h5_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))
    if max_scenes is not None:
        json_files = json_files[:max_scenes]

    h5_paths: list[Path] = []
    total = len(json_files)

    for i, json_path in enumerate(json_files, 1):
        h5_path = h5_dir / f"{json_path.stem}.h5"
        h5_paths.append(h5_path)

        if h5_path.exists():
            print(f"[{i}/{total}] Skipping {json_path.name} (H5 exists)")
            continue

        print(f"[{i}/{total}] Converting {json_path.name}...")
        result = subprocess.run(
            [
                sys.executable,
                str(CONVERT_SCRIPT),
                str(json_path),
                "--output_h5_path", str(h5_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  FAILED: {result.stderr.strip()}")
            h5_paths.pop()
        else:
            print(f"  OK -> {h5_path.name}")

    return [p for p in h5_paths if p.exists()]


# ─── Step 2: Render H5 files ──────────────────────────────────────────────────

def load_h5(path: Path) -> dict[str, torch.Tensor]:
    with h5py.File(path, "r") as f:
        return {
            "triangles": torch.from_numpy(np.array(f["triangles"]).astype(np.float32)),
            "texture": torch.from_numpy(np.array(f["texture"]).astype(np.float32)),
            "vn": torch.from_numpy(np.array(f["vn"]).astype(np.float32)),
            "c2w": torch.from_numpy(np.array(f["c2w"]).astype(np.float32)),
            "fov": torch.from_numpy(np.array(f["fov"]).astype(np.float32)),
        }


def render_all_scenes(
    h5_paths: list[Path],
    output_dir: Path,
    model_id: str,
    precision: str,
    resolution: int,
    tone_mapper_name: str,
):
    """Load model once, render each scene sequentially."""
    from renderformer import RenderFormerRenderingPipeline

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"\nLoading model: {model_id}")
    pipeline = RenderFormerRenderingPipeline.from_pretrained(model_id)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        precision = "fp32"
        print("MPS detected, forcing fp32.")

    pipeline.to(device)

    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]

    tone_mapper = None
    if tone_mapper_name != "none":
        tm_name = "Khronos PBR Neutral" if tone_mapper_name == "pbr_neutral" else tone_mapper_name
        tone_mapper = ToneMapper(tm_name)
        print(f"Using {tm_name} tone mapper")

    total = len(h5_paths)
    for i, h5_path in enumerate(h5_paths, 1):
        base_name = h5_path.stem
        print(f"\n[{i}/{total}] Rendering {base_name}...")

        data = load_h5(h5_path)
        num_tris = data["triangles"].shape[0]
        mask = torch.ones(num_tris, dtype=torch.bool)

        # Add batch dimension and move to device
        triangles = data["triangles"].unsqueeze(0).to(device)
        texture = data["texture"].unsqueeze(0).to(device)
        vn = data["vn"].unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        c2w = data["c2w"].unsqueeze(0).to(device)
        fov = data["fov"].unsqueeze(0).unsqueeze(-1).to(device)

        rendered_imgs = pipeline(
            triangles=triangles,
            texture=texture,
            mask=mask,
            vn=vn,
            c2w=c2w,
            fov=fov,
            resolution=resolution,
            torch_dtype=torch_dtype,
        )

        nv = c2w.shape[1]
        for v in range(nv):
            hdr_img = rendered_imgs[0, v].cpu().to(torch.float32).numpy()

            if tone_mapper is not None:
                ldr_img = tone_mapper.hdr_to_ldr(hdr_img)
            else:
                ldr_img = np.clip(hdr_img, 0, 1)
            ldr_img = (ldr_img * 255).astype(np.uint8)

            png_path = output_dir / f"{base_name}_view_{v}.png"
            imageio.v3.imwrite(png_path, ldr_img)
            print(f"  Saved {png_path.name}")

        # Free GPU memory between scenes
        del triangles, texture, vn, mask, c2w, fov, rendered_imgs
        torch.cuda.empty_cache()

    print(f"\nAll renders saved to {output_dir}")


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch convert + render training scenes")
    parser.add_argument("--skip_convert", action="store_true", help="Skip H5 conversion, render existing H5s")
    parser.add_argument("--max_scenes", type=int, default=None, help="Limit number of scenes to process")
    parser.add_argument("--model_id", type=str, default="microsoft/renderformer-v1.1-swin-large")
    parser.add_argument("--precision", type=str, choices=["bf16", "fp16", "fp32"], default="fp16")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--tone_mapper", type=str, choices=["none", "agx", "filmic", "pbr_neutral"], default="none")
    args = parser.parse_args()

    if args.skip_convert:
        h5_paths = sorted(H5_OUTPUT_DIR.glob("*.h5"))
        if args.max_scenes is not None:
            h5_paths = h5_paths[:args.max_scenes]
        if not h5_paths:
            print(f"No H5 files found in {H5_OUTPUT_DIR}. Run without --skip_convert first.")
            return
        print(f"Found {len(h5_paths)} existing H5 files")
    else:
        print("=== Step 1: Converting scene JSONs to H5 ===\n")
        h5_paths = convert_all_scenes(TRAINING_EXAMPLES_DIR, H5_OUTPUT_DIR, args.max_scenes)
        if not h5_paths:
            print("No scenes converted successfully.")
            return

    print(f"\n=== Step 2: Rendering {len(h5_paths)} scenes ===")
    render_all_scenes(
        h5_paths=h5_paths,
        output_dir=RENDER_OUTPUT_DIR,
        model_id=args.model_id,
        precision=args.precision,
        resolution=args.resolution,
        tone_mapper_name=args.tone_mapper,
    )


if __name__ == "__main__":
    main()
