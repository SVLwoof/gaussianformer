"""Orbit cameras around the origin in the Blender convention (-Z forward, +Y up, +X right).

Views alternate between a high (+0.4 r) and a low (-0.1 r) elevation to cover the top and the bottom of the object.
"""
import numpy as np

FOV = 45.0
RESOLUTION = 512
DISTANCES = ((1.7, 14), (1.15, 7), (2.45, 7))  # (orbit radius, views): 28 views per object


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-9
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-9
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, np.cross(right, forward), -forward, eye
    return c2w


def orbit(n_views: int, radius: float) -> np.ndarray:
    """Camera-to-world matrices [n_views, 4, 4]."""
    c2w = []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        elevation = 0.4 * radius if i % 2 == 0 else -0.1 * radius
        eye = np.array([radius * np.cos(theta), elevation, radius * np.sin(theta)], np.float32)
        c2w.append(look_at(eye, np.zeros(3, np.float32), np.array([0, 1, 0], np.float32)))
    return np.stack(c2w)


def all_views() -> np.ndarray:
    """The 28 training cameras: 14 at distance 1.7, then 7 each at 1.15 and 2.45."""
    return np.concatenate([orbit(n, r) for r, n in DISTANCES])


def to_gsplat(c2w: np.ndarray, fov: float = FOV, resolution: int = RESOLUTION) -> tuple[np.ndarray, np.ndarray]:
    """Camera-to-world matrices [V, 4, 4] -> gsplat view matrices (OpenCV convention) and intrinsics [V, 3, 3]."""
    flip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    viewmats = (flip @ np.linalg.inv(c2w)).astype(np.float32)
    f = 0.5 * resolution / np.tan(0.5 * np.radians(fov))
    K = np.array([[f, 0, resolution / 2], [0, f, resolution / 2], [0, 0, 1]], np.float32)
    return viewmats, np.tile(K, (len(c2w), 1, 1))
