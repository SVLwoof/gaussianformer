import os
import torch
import h5py
import argparse
import numpy as np
import imageio
from pathlib import Path
from simple_ocio import ToneMapper

from gaussianformer.pipelines.rendering_pipeline import GaussianFormerRenderingPipeline


def load_single_gaussian_h5_data(file_path: Path):
    """
    Loads a single H5 file containing Gaussian scene data.
    It reads individual Gaussian components and assembles them into a single tensor.
    """
    with h5py.File(file_path, 'r') as f:
        positions = torch.from_numpy(np.array(f['means']).astype(np.float32))
        scales = torch.from_numpy(np.array(f['scales']).astype(np.float32))
        rotations = torch.from_numpy(np.array(f['rotations']).astype(np.float32))
        colors = torch.from_numpy(np.array(f['colors']).astype(np.float32))
        opacities = torch.from_numpy(np.array(f['opacities']).astype(np.float32))

        # Ensure `opacities` has a feature dimension for concatenation
        if opacities.dim() == 1:
            opacities = opacities.unsqueeze(-1)

        # Assemble the final 'gaussians' tensor in the correct order
        # Order: position (3), scale (3), rotation (4), color (3), opacity (1)
        gaussians = torch.cat([
            positions,
            scales,
            rotations,
            colors,
            opacities
        ], dim=-1)

        num_gaussians = gaussians.shape[0]

        # Create a mask for all gaussians
        mask = torch.ones(num_gaussians, dtype=torch.bool)

        # Load camera parameters
        c2w = torch.from_numpy(np.array(f['c2w']).astype(np.float32))
        fov = torch.from_numpy(np.array(f['fov']).astype(np.float32))

        data = {
            'gaussians': gaussians,
            'mask': mask,
            'c2w': c2w,
            'fov': fov,
        }
    return data


def main():
    parser = argparse.ArgumentParser(description="Infer using GaussianFormer model")
    # Use Path for type validation
    parser.add_argument("--h5_file", type=Path, required=True, help="Path to the input H5 file with Gaussian data")
    parser.add_argument("--model_id", type=str, help="Model ID on Hugging Face or local path",
                        default="shahafvl/gaussianformer-v10b")
    parser.add_argument("--precision", type=str, choices=['bf16', 'fp16', 'fp32'], default='fp16',
                        help="Precision for inference")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution for inference")
    parser.add_argument("--output_dir", type=Path, help="Output directory (Default: same as input H5 file)",
                        required=False)
    parser.add_argument("--tone_mapper", type=str, choices=['none', 'agx', 'filmic', 'pbr_neutral'], default='none',
                        help="Tone mapper for inference")
    args = parser.parse_args()

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # Use the GaussianFormer pipeline
    pipeline = GaussianFormerRenderingPipeline.from_pretrained(args.model_id)

    # Optional: Apply optimized kernels if available
    if device == torch.device('cuda') and os.name == 'posix':
        try:
            from renderformer_liger_kernel import apply_kernels
            apply_kernels(pipeline.model)
            print("Applied optimized kernels.")
        except ImportError:
            print("Optimized kernels not found, running with standard PyTorch.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif device == torch.device('mps'):
        args.precision = 'fp32'
        print("bf16 and fp16 can cause issues on MPS, forcing fp32.")

    pipeline.to(device)

    # Tone mapper setup
    if args.tone_mapper != 'none':
        if args.tone_mapper == 'pbr_neutral':
            args.tone_mapper = 'Khronos PBR Neutral'
        tone_mapper = ToneMapper(args.tone_mapper)
        print(f"Using {args.tone_mapper} tone mapper")

    # Load Gaussian data and move to device
    data = load_single_gaussian_h5_data(args.h5_file)

    # Add a batch dimension to all tensors
    gaussians = data['gaussians'].unsqueeze(0).to(device)
    mask = data['mask'].unsqueeze(0).to(device)
    c2w = data['c2w'].unsqueeze(0).to(device)
    fov = data['fov'].unsqueeze(0).unsqueeze(-1).to(device)

    # Call the rendering pipeline with the new arguments
    rendered_imgs = pipeline(
        gaussians=gaussians,
        mask=mask,
        c2w=c2w,
        fov=fov,
        resolution=args.resolution,
        torch_dtype=torch.float16 if args.precision == 'fp16' else torch.bfloat16 if args.precision == 'bf16' else torch.float32,
    )
    print("Inference completed. Rendered images shape:", rendered_imgs.shape)

    # --- Save output images using pathlib ---
    output_dir = args.output_dir if args.output_dir else args.h5_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = args.h5_file.stem

    nv = c2w.shape[1]
    for i in range(nv):
        hdr_img = rendered_imgs[0, i].cpu().to(torch.float32).numpy().astype(np.float32)
        if args.tone_mapper != 'none':
            ldr_img = tone_mapper.hdr_to_ldr(hdr_img)
        else:
            ldr_img = np.clip(hdr_img, 0, 1)
        ldr_img = (ldr_img * 255).astype(np.uint8)

        # Use Path object to construct output paths
        hdr_path = output_dir / f"{base_name}_view_{i}.exr"
        ldr_path = output_dir / f"{base_name}_view_{i}.png"

        # Temporary plugin download
        imageio.plugins.freeimage.download()

        imageio.v3.imwrite(hdr_path, hdr_img)
        imageio.v3.imwrite(ldr_path, ldr_img)

        print(f"Saved {hdr_path} and {ldr_path}")


if __name__ == '__main__':
    main()
