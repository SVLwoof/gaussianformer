"""Convert a 3D Gaussian Splatting PLY into a GaussianFormer scene: activated, normalised Gaussians, pruned to
about 20k, plus orbit cameras.

  python -m tools.ply_to_h5 --ply object.ply --out object.h5 --up z
  python infer.py --h5 object.h5 --out renders/
"""
import argparse
from pathlib import Path

import h5py
import numpy as np

from gaussianformer.splat import KEEP, TO_Y_UP, load_ply
from gaussianformer.utils.cameras import FOV, orbit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--up", choices=TO_Y_UP, default="y", help="the file's up axis")
    ap.add_argument("--keep", type=int, default=KEEP, help="Gaussians to keep")
    ap.add_argument("--no_recovery", action="store_true", help="skip the fine-tune of the kept Gaussians")
    ap.add_argument("--views", type=int, default=14, help="cameras on an orbit around the object")
    ap.add_argument("--distance", type=float, default=1.7, help="orbit radius (the model is trained on 1.15-2.45)")
    a = ap.parse_args()

    g = load_ply(a.ply, up=a.up, keep=a.keep, recovery=not a.no_recovery).cpu().numpy()
    c2w = orbit(a.views, a.distance)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.out, "w") as f:
        for name, (lo, hi) in {"means": (0, 3), "scales": (3, 6), "rotations": (6, 10), "colors": (10, 13),
                               "opacities": (13, 14)}.items():
            f.create_dataset(name, data=g[:, lo:hi])
        f.create_dataset("c2w", data=c2w)
        f.create_dataset("fov", data=np.full(len(c2w), FOV, np.float32))
    print(f"{a.out}: {len(g)} Gaussians, {len(c2w)} cameras at distance {a.distance}")


if __name__ == "__main__":
    main()
