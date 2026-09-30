"""CPU tests of the PLY conversion: `uv run python -m tests.test_splat`."""
import tempfile
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from gaussianformer.splat import SH_C0, normalize, read_ply, to_tensor


def write_ply(path: Path, n: int = 500) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    raw = {"x": rng.normal(5, 2, n), "y": rng.normal(-3, 1, n), "z": rng.normal(0, 4, n),
           **{f"scale_{i}": rng.normal(-4, 1, n) for i in range(3)},
           **{f"rot_{i}": rng.normal(0, 1, n) for i in range(4)},
           **{f"f_dc_{i}": rng.normal(0, 1, n) for i in range(3)},
           **{f"f_rest_{i}": rng.normal(0, 1, n) for i in range(9)},
           "opacity": rng.normal(0, 2, n)}
    vertex = np.empty(n, dtype=[(k, "f4") for k in raw])
    for k, v in raw.items():
        vertex[k] = v
    PlyData([PlyElement.describe(vertex, "vertex")]).write(str(path))
    return raw


def test_read_and_normalize():
    with tempfile.TemporaryDirectory() as d:
        raw = write_ply(Path(d) / "splat.ply")
        g = read_ply(Path(d) / "splat.ply")
    assert np.allclose(g["scales"], np.exp(np.stack([raw[f"scale_{i}"] for i in range(3)], -1)), rtol=1e-5)
    assert np.allclose(g["opacities"], 1 / (1 + np.exp(-raw["opacity"])), rtol=1e-5)
    assert np.allclose(g["colors"], np.clip(0.5 + SH_C0 * np.stack([raw[f"f_dc_{i}"] for i in range(3)], -1), 0, 1), atol=1e-6)
    assert np.allclose(np.linalg.norm(g["rotations"], axis=-1), 1, atol=1e-5)

    for up in ("y", "z", "-y"):
        n = normalize(g, up)
        assert np.isclose(np.abs(n["means"]).max(), 0.45, atol=1e-5)
        assert np.allclose(np.median(n["means"], axis=0), 0, atol=1e-6)
        assert np.allclose(np.linalg.norm(n["rotations"], axis=-1), 1, atol=1e-5)
        # scales shrink with the positions: the ratio of any two scales is unchanged
        assert np.allclose(n["scales"] / g["scales"], (n["scales"] / g["scales"])[0, 0], rtol=1e-4)
    assert to_tensor(normalize(g)).shape == (500, 14)


if __name__ == "__main__":
    test_read_and_normalize()
    print("test_read_and_normalize: ok")
