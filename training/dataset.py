import os
import random
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
import roma
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def load_gaussians(h5_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HDF5 scene -> Gaussians [N, 14] (pos, scale, quat wxyz, rgb, opacity), c2w [V, 4, 4], fov [V] (degrees)."""
    with h5py.File(h5_path, "r") as f:
        fields = [np.asarray(f[k], dtype=np.float32) for k in ("means", "scales", "rotations", "colors", "opacities")]
        fields[-1] = fields[-1].reshape(-1, 1)
        return np.concatenate(fields, axis=-1), np.asarray(f["c2w"], dtype=np.float32), np.asarray(f["fov"], dtype=np.float32)


class GaussianRenderDataset(Dataset):
    """(object, view) samples: the object's Gaussians, one camera and its ground-truth image.

    Renders are `<scene>_view_<i>.png` next to `<scene>.h5`'s view i. With `views_per_epoch`, each epoch
    uses that many random views per object (call `set_epoch`). With `augment_rotation`, every sample
    rotates the scene and camera by one random rotation, which leaves the image unchanged.
    """

    def __init__(self, h5_dir: Path, renders_dir: Path, resolution: int, augment_rotation: bool = False,
                 views_per_epoch: int | None = None, objects: list[int] | None = None):
        self.resolution = resolution
        self.augment_rotation = augment_rotation
        self.views_per_epoch = views_per_epoch
        rendered = set(os.listdir(renders_dir))
        self.samples: list[tuple[Path, int, Path]] = []
        self.by_scene: list[list[int]] = []
        h5_paths = sorted(Path(h5_dir).glob("*.h5"))
        if objects is not None:
            keep = {f"scene_{i:04d}" for i in objects}
            h5_paths = [p for p in h5_paths if p.stem in keep]
        for h5_path in h5_paths:
            with h5py.File(h5_path, "r") as f:
                n_views = f["c2w"].shape[0]
            idx = []
            for v in range(n_views):
                name = f"{h5_path.stem}_view_{v}.png"
                if name in rendered:
                    idx.append(len(self.samples))
                    self.samples.append((h5_path, v, Path(renders_dir) / name))
            self.by_scene.append(idx)
        self.set_epoch(0)
        print(f"{len(self.samples)} views of {len(self.by_scene)} objects from {h5_dir}")

    def set_epoch(self, epoch: int) -> None:
        if not self.views_per_epoch:
            self.active = list(range(len(self.samples)))
            return
        rng = random.Random(epoch)  # same draw on every rank
        self.active = [i for idx in self.by_scene for i in rng.sample(idx, min(self.views_per_epoch, len(idx)))]

    def __len__(self) -> int:
        return len(self.active)

    @staticmethod
    def rotate_scene(gaussians: torch.Tensor, c2w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate the Gaussians and the camera by one uniformly random rotation."""
        R = roma.random_rotmat().to(gaussians.dtype).reshape(3, 3)
        gaussians = gaussians.clone()
        gaussians[:, 0:3] = gaussians[:, 0:3] @ R.T
        rot = roma.unitquat_to_rotmat(gaussians[:, [7, 8, 9, 6]])  # wxyz -> roma's xyzw
        gaussians[:, 6:10] = roma.rotmat_to_unitquat(R @ rot)[:, [3, 0, 1, 2]]
        R4 = torch.eye(4, dtype=c2w.dtype)
        R4[:3, :3] = R
        return gaussians, R4 @ c2w

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        h5_path, view, render_path = self.samples[self.active[i]]
        gaussians, c2w, fov = load_gaussians(h5_path)
        img = torch.from_numpy(iio.imread(render_path)[..., :3].astype(np.float32) / 255.0)
        if img.shape[0] != self.resolution:
            img = F.interpolate(img.permute(2, 0, 1)[None], size=(self.resolution, self.resolution),
                                mode="bilinear", align_corners=False, antialias=True)[0].permute(1, 2, 0)
        gaussians, c2w = torch.from_numpy(gaussians), torch.from_numpy(c2w[view])
        if self.augment_rotation:
            gaussians, c2w = self.rotate_scene(gaussians, c2w)
        return {"gaussians": gaussians, "mask": torch.ones(len(gaussians), dtype=torch.bool), "c2w": c2w,
                "fov": torch.tensor(fov[view]), "target": img}


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad the Gaussians (and masks) to the largest object in the batch; stack everything else."""
    n = max(s["gaussians"].shape[0] for s in batch)
    out = {k: torch.stack([s[k] for s in batch]) for k in ("c2w", "fov", "target")}
    out["gaussians"] = torch.stack([F.pad(s["gaussians"], (0, 0, 0, n - len(s["gaussians"]))) for s in batch])
    out["mask"] = torch.stack([F.pad(s["mask"], (0, n - len(s["mask"]))) for s in batch])
    return out
