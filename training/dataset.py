"""Dataset for pairing Gaussian H5 scene data with ground-truth rendered images."""

from pathlib import Path

import h5py
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset


class GaussianRenderDataset(Dataset):
    """
    Each sample is a single (scene, view) pair:
      - Input: Gaussian parameters (14-dim) + camera (c2w, fov)
      - Target: Ground-truth rendered image from the mesh pipeline

    The dataset discovers all scene H5 files and their corresponding render
    images, expanding multi-view scenes into individual samples.
    """

    def __init__(
        self,
        gaussian_h5_dir: Path,
        renders_dir: Path,
        resolution: int = 256,
        max_samples: int | None = None,
    ):
        self.gaussian_h5_dir = Path(gaussian_h5_dir)
        self.renders_dir = Path(renders_dir)
        self.resolution = resolution

        # Build index of (h5_path, view_index, render_path) triples
        self.samples: list[tuple[Path, int, Path]] = []
        for h5_path in sorted(self.gaussian_h5_dir.glob("*.h5")):
            scene_name = h5_path.stem
            # Count views in H5
            with h5py.File(h5_path, "r") as f:
                num_views = f["c2w"].shape[0]
            for view_idx in range(num_views):
                # Try EXR first (HDR), fall back to PNG (LDR)
                exr_path = self.renders_dir / f"{scene_name}_view_{view_idx}.exr"
                png_path = self.renders_dir / f"{scene_name}_view_{view_idx}.png"
                if exr_path.exists():
                    self.samples.append((h5_path, view_idx, exr_path))
                elif png_path.exists():
                    self.samples.append((h5_path, view_idx, png_path))
                # Skip views with no matching render

        if max_samples is not None and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

        print(f"GaussianRenderDataset: {len(self.samples)} samples from "
              f"{len(set(s[0] for s in self.samples))} scenes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        h5_path, view_idx, render_path = self.samples[idx]

        # --- Load Gaussian scene data ---
        with h5py.File(h5_path, "r") as f:
            means = np.array(f["means"], dtype=np.float32)
            scales = np.array(f["scales"], dtype=np.float32)
            rotations = np.array(f["rotations"], dtype=np.float32)
            colors = np.array(f["colors"], dtype=np.float32)
            opacities = np.array(f["opacities"], dtype=np.float32)
            if opacities.ndim == 1:
                opacities = opacities[:, np.newaxis]
            c2w = np.array(f["c2w"], dtype=np.float32)  # [num_views, 4, 4]
            fov = np.array(f["fov"], dtype=np.float32)   # [num_views]

        # Assemble 14-dim Gaussian tensor
        gaussians = np.concatenate([means, scales, rotations, colors, opacities], axis=-1)
        mask = np.ones(gaussians.shape[0], dtype=np.bool_)

        # Select the single view
        c2w_view = c2w[view_idx]   # [4, 4]
        fov_view = fov[view_idx]   # scalar

        # --- Load target image ---
        img = imageio.v3.imread(render_path).astype(np.float32)
        if render_path.suffix == ".png":
            img = img / 255.0
        # Resize if needed (simple nearest for speed; bilinear would be better for quality)
        if img.shape[0] != self.resolution or img.shape[1] != self.resolution:
            from PIL import Image
            pil_img = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8) if img.max() <= 1.0
                                      else (img.clip(0, 65535)).astype(np.uint16))
            pil_img = pil_img.resize((self.resolution, self.resolution), Image.BILINEAR)
            img = np.array(pil_img, dtype=np.float32) / 255.0
        # Ensure 3 channels
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]

        return {
            "gaussians": torch.from_numpy(gaussians),          # [N, 14]
            "mask": torch.from_numpy(mask),                     # [N]
            "c2w": torch.from_numpy(c2w_view),                  # [4, 4]
            "fov": torch.tensor(fov_view),                      # scalar
            "target": torch.from_numpy(img),                    # [H, W, 3]
        }


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad gaussians and masks to the max count in the batch, stack the rest."""
    max_n = max(sample["gaussians"].shape[0] for sample in batch)
    dim = batch[0]["gaussians"].shape[1]

    padded_gaussians = []
    padded_masks = []
    for sample in batch:
        n = sample["gaussians"].shape[0]
        pad_size = max_n - n
        if pad_size > 0:
            padded_gaussians.append(
                torch.cat([sample["gaussians"], torch.zeros(pad_size, dim)], dim=0)
            )
            padded_masks.append(
                torch.cat([sample["mask"], torch.zeros(pad_size, dtype=torch.bool)], dim=0)
            )
        else:
            padded_gaussians.append(sample["gaussians"])
            padded_masks.append(sample["mask"])

    return {
        "gaussians": torch.stack(padded_gaussians),
        "mask": torch.stack(padded_masks),
        "c2w": torch.stack([s["c2w"] for s in batch]),
        "fov": torch.stack([s["fov"] for s in batch]),
        "target": torch.stack([s["target"] for s in batch]),
    }
