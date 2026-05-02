"""Orbit camera helpers used by tomatoes / Objaverse pipelines.

All conventions are Blender (-Z view direction, +Y cam-up, +X cam-right).
The orbit pattern alternates between high (+0.4R) and low (-0.1R) elevation
to cover both top and bottom of the subject.
"""

import numpy as np


def look_at_blender(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build a c2w in Blender convention (-Z view direction, +Y cam-up, +X cam-right)."""
    eye, target, up = (np.asarray(x, dtype=np.float32) for x in (eye, target, up))
    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-9
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-9
    cam_up = np.cross(right, forward)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = cam_up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = eye
    return c2w


def c2w_to_viewmat(c2w: np.ndarray) -> np.ndarray:
    w2c = np.linalg.inv(c2w)
    flip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    return (flip @ w2c).astype(np.float32)


def make_orbit_views(
    n_views: int, radius: float, fov_deg: float, resolution: int, up_axis: str = "y"
) -> tuple[np.ndarray, np.ndarray]:
    """Return (viewmats[V,4,4], Ks[V,3,3]) for an orbit around the origin.

    up_axis selects which world-coord axis is "vertical" (gravity-aligned).
    Real-world COLMAP captures conventionally use Y-up; our synthetic training
    data uses Z-up. The orbit always circles in the horizontal plane.

    Half the views look slightly down (eye on the +up side), half slightly up
    (eye on the -up side), to cover both top and bottom of the object.
    """
    assert up_axis in ("y", "z")
    fov_rad = fov_deg * np.pi / 180.0
    focal = 0.5 * resolution / np.tan(0.5 * fov_rad)
    K = np.array(
        [[focal, 0, resolution / 2.0], [0, focal, resolution / 2.0], [0, 0, 1]],
        dtype=np.float32,
    )
    if up_axis == "z":
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        eye_fn = lambda theta, elev: np.array(
            [radius * np.cos(theta), radius * np.sin(theta), elev], dtype=np.float32
        )
    else:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        eye_fn = lambda theta, elev: np.array(
            [radius * np.cos(theta), elev, radius * np.sin(theta)], dtype=np.float32
        )
    viewmats = []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        elev = 0.4 * radius if i % 2 == 0 else -0.1 * radius
        c2w = look_at_blender(eye_fn(theta, elev), np.zeros(3), up=up)
        viewmats.append(c2w_to_viewmat(c2w))
    return np.stack(viewmats), np.tile(K[None, :, :], (n_views, 1, 1))
