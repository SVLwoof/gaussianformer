"""Dataset for pairing Gaussian H5 scene data with ground-truth rendered images."""

import random
from pathlib import Path

import h5py
import imageio
import numpy as np
import roma
import torch
import torch.nn.functional as F
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
        augment_rotation: bool = False,
        views_per_epoch: int | None = None,
        log_scale_input: bool = False,
    ):
        # log10(scale)+3: median scale ~0.004 spans ~3 decades in a sliver of the raw input
        # range; log spreads it to O(1) (0.004 -> 0.6, 0.1 -> 2). Must match eval-side loading.
        self.log_scale_input = log_scale_input
        self.gaussian_h5_dir = Path(gaussian_h5_dir)
        self.renders_dir = Path(renders_dir)
        self.resolution = resolution
        # Per-epoch view subsampling: with K set, each epoch draws K random views per scene
        # (instead of all 14), shrinking the epoch ~14/K and giving finer checkpoint
        # granularity, while the model still covers all views across epochs. Call set_epoch()
        # each epoch to redraw (seeded by epoch so every DDP rank agrees on the selection).
        self.views_per_epoch = views_per_epoch
        # On-the-fly Haar-uniform scene+camera rotation (RenderFormer's RoMa aug). The
        # render is invariant under a joint scene+camera rotation (sh_degree=None ->
        # constant per-Gaussian color, no world-fixed lighting), so the GT image is
        # left untouched. Only the train dataset should set this; val stays fixed.
        self.augment_rotation = augment_rotation

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
            # Seeded random subset, NOT a prefix: the sample list is scene-sorted, so
            # truncation would silently keep only the lowest-index scenes.
            self.samples = sorted(random.Random(0).sample(self.samples, max_samples))

        # Group sample indices by scene for per-epoch view subsampling.
        self._by_scene: dict[Path, list[int]] = {}
        for i, (h5_path, _, _) in enumerate(self.samples):
            self._by_scene.setdefault(h5_path, []).append(i)
        self._active: list[int] = list(range(len(self.samples)))
        self.set_epoch(0)

        msg = (f"GaussianRenderDataset: {len(self.samples)} samples from "
               f"{len(self._by_scene)} scenes")
        if self.views_per_epoch:
            msg += (f"  (subsampling {self.views_per_epoch} views/scene/epoch "
                    f"-> {len(self._active)} samples/epoch)")
        print(msg)

    def set_epoch(self, epoch: int) -> None:
        """Pick this epoch's active (scene, view) samples. With views_per_epoch=K, draw K
        random views per scene seeded by `epoch` (identical across DDP ranks so the shared
        DistributedSampler splits a consistent pool); otherwise use every sample."""
        if not self.views_per_epoch:
            self._active = list(range(len(self.samples)))
            return
        rng = random.Random(epoch)
        active: list[int] = []
        for idxs in self._by_scene.values():
            active.extend(rng.sample(idxs, min(self.views_per_epoch, len(idxs))))
        self._active = active

    def __len__(self) -> int:
        return len(self._active)

    @staticmethod
    def _rotate_scene(gaussians: torch.Tensor, c2w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one Haar-uniform rotation R to the scene + camera, image-preserving.

        gaussians: [N, 14] = [pos(3), scale(3), rot_quat_wxyz(4), color(3), opacity(1)].
        c2w: [4, 4] camera-to-world. Positions rotate (means @ R.T), each Gaussian's
        orientation world-rotates (R @ M via the matrix path -- avoids quat-product
        operand ambiguity), and the camera pose rotates in world space (R4 @ c2w).
        Scale/color/opacity are rotation-invariant. RoMa quaternions are XYZW; ours are
        WXYZ -> reorder at the boundary.
        """
        R = roma.random_rotmat().to(gaussians.dtype)
        if R.ndim == 3:
            R = R[0]

        means = gaussians[:, 0:3]
        quats_wxyz = gaussians[:, 6:10]

        means_rot = means @ R.T

        q_xyzw = quats_wxyz[:, [1, 2, 3, 0]]
        M = roma.unitquat_to_rotmat(q_xyzw)            # [N, 3, 3]
        M_rot = torch.matmul(R, M)                     # broadcast R over the N batch
        q_rot_xyzw = roma.rotmat_to_unitquat(M_rot)
        q_rot_wxyz = q_rot_xyzw[:, [3, 0, 1, 2]]

        gaussians = gaussians.clone()
        gaussians[:, 0:3] = means_rot
        gaussians[:, 6:10] = q_rot_wxyz

        R4 = torch.eye(4, dtype=c2w.dtype)
        R4[:3, :3] = R
        c2w_rot = R4 @ c2w

        return gaussians, c2w_rot

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        h5_path, view_idx, render_path = self.samples[self._active[idx]]

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
        if self.log_scale_input:
            scales = np.log10(np.clip(scales, 1e-8, None)) + 3.0
        gaussians = np.concatenate([means, scales, rotations, colors, opacities], axis=-1)
        mask = np.ones(gaussians.shape[0], dtype=np.bool_)

        # Select the single view
        c2w_view = c2w[view_idx]   # [4, 4]
        fov_view = fov[view_idx]   # scalar

        # --- Load target image ---
        img = imageio.v3.imread(render_path).astype(np.float32)
        if render_path.suffix == ".png":
            img = img / 255.0
        # Ensure 3 channels (before resize, so the resize sees a fixed layout)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]
        # Resize to the target resolution (e.g. 512 GT downscaled to 256 for the curriculum's
        # bulk stage). Float bilinear with antialias, matching PIL's filtering. The old PIL
        # path round-tripped through uint8 (re-quantizing the interpolated target) and, for
        # HDR inputs with max > 1, cast to uint16 WITHOUT scaling then divided by 255 --
        # a silent corruption for any future EXR target.
        if img.shape[0] != self.resolution or img.shape[1] != self.resolution:
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            t = F.interpolate(t, size=(self.resolution, self.resolution),
                              mode="bilinear", align_corners=False, antialias=True)
            img = t.squeeze(0).permute(1, 2, 0).numpy()

        gaussians_t = torch.from_numpy(gaussians)               # [N, 14]
        c2w_t = torch.from_numpy(c2w_view)                      # [4, 4]
        if self.augment_rotation:
            gaussians_t, c2w_t = self._rotate_scene(gaussians_t, c2w_t)

        return {
            "gaussians": gaussians_t,                           # [N, 14]
            "mask": torch.from_numpy(mask),                     # [N]
            "c2w": c2w_t,                                       # [4, 4]
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
