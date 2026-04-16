"""Compose training scenes from processed objects + procedural Gaussian backgrounds.

Each scene has:
  - 1-2 objects placed on a floor with optional walls
  - Procedural Gaussian plane backgrounds (floor, back wall, optional side walls)
  - 1-3 camera views from a spherical shell

Output: H5 files in data_v2/h5s/ (same format as training/dataset.py expects).

Usage:
    uv run python data_v2/compose_scenes.py
    uv run python data_v2/compose_scenes.py --num_scenes 500 --views_per_scene 3
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


def look_at_to_c2w(
    camera_position: np.ndarray,
    target_position: np.ndarray = np.zeros(3),
    up_dir: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    """Blender-convention camera-to-world matrix (Z-up, -Z = view direction)."""
    camera_direction = np.array(camera_position) - np.array(target_position)
    camera_direction = camera_direction / np.linalg.norm(camera_direction)
    camera_right = np.cross(np.array(up_dir), camera_direction)
    camera_right = camera_right / np.linalg.norm(camera_right)
    camera_up = np.cross(camera_direction, camera_right)
    camera_up = camera_up / np.linalg.norm(camera_up)
    rotation_transform = np.zeros((4, 4))
    rotation_transform[0, :3] = camera_right
    rotation_transform[1, :3] = camera_up
    rotation_transform[2, :3] = camera_direction
    rotation_transform[-1, -1] = 1.0
    translation_transform = np.eye(4)
    translation_transform[:3, -1] = -np.array(camera_position)
    look_at_transform = rotation_transform @ translation_transform
    return np.linalg.inv(look_at_transform).astype(np.float32)


def make_gaussian_plane(
    center: np.ndarray,
    normal: np.ndarray,
    size: float,
    grid_n: int,
    color: np.ndarray,
    color_jitter: float = 0.02,
    opacity: float = 1.0,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Create a grid of flat disc Gaussians forming a plane surface.

    Uses overlapping discs (radius = spacing * 0.75) to ensure full coverage
    with no gaps. Opacity defaults to 1.0 for solid walls.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Build local coordinate frame from normal
    normal = normal / np.linalg.norm(normal)
    if abs(normal[2]) < 0.9:
        tangent = np.cross(normal, np.array([0, 0, 1]))
    else:
        tangent = np.cross(normal, np.array([1, 0, 0]))
    tangent = tangent / np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)

    spacing = size / grid_n
    half = size / 2.0
    n_total = grid_n * grid_n

    # Grid positions in local frame
    u = np.linspace(-half + spacing / 2, half - spacing / 2, grid_n)
    uu, vv = np.meshgrid(u, u)
    local_pos = np.stack([uu.ravel(), vv.ravel()], axis=-1)

    # World positions
    means = (
        center[None, :]
        + local_pos[:, 0:1] * tangent[None, :]
        + local_pos[:, 1:2] * bitangent[None, :]
    ).astype(np.float32)

    # Overlapping discs for solid coverage
    disc_radius = spacing * 0.75
    disc_thickness = 0.001
    scales = np.full((n_total, 3), [disc_radius, disc_radius, disc_thickness], dtype=np.float32)

    # Rotation: align local Z with the plane normal
    rot_mat = np.stack([tangent, bitangent, normal], axis=-1)
    r = Rotation.from_matrix(rot_mat)
    quats = r.as_quat()  # (x, y, z, w) scipy convention
    # Convert to (w, x, y, z)
    rotations = np.zeros((n_total, 4), dtype=np.float32)
    rotations[:, 0] = quats[3]
    rotations[:, 1] = quats[0]
    rotations[:, 2] = quats[1]
    rotations[:, 3] = quats[2]

    # Colors with slight jitter
    colors = np.tile(color, (n_total, 1)).astype(np.float32)
    colors += rng.uniform(-color_jitter, color_jitter, colors.shape).astype(np.float32)
    colors = np.clip(colors, 0, 1)

    opacities = np.full(n_total, opacity, dtype=np.float32)

    return {
        "means": means,
        "scales": scales,
        "rotations": rotations,
        "colors": colors,
        "opacities": opacities,
    }


# Saturated color palette for walls and surfaces
SURFACE_COLORS = [
    np.array([0.7, 0.1, 0.1]),   # red
    np.array([0.1, 0.6, 0.1]),   # green
    np.array([0.1, 0.2, 0.7]),   # blue
    np.array([0.7, 0.5, 0.1]),   # warm orange
    np.array([0.6, 0.1, 0.5]),   # purple
    np.array([0.1, 0.5, 0.6]),   # teal
    np.array([0.7, 0.7, 0.1]),   # yellow
    np.array([0.6, 0.3, 0.2]),   # terracotta
    np.array([0.3, 0.3, 0.7]),   # periwinkle
    np.array([0.7, 0.3, 0.4]),   # salmon
]


def _pick_surface_color(rng: np.random.Generator, saturated: bool = True) -> np.ndarray:
    """Pick a random surface color, optionally saturated."""
    if saturated:
        base = SURFACE_COLORS[rng.integers(len(SURFACE_COLORS))]
        return np.clip(base + rng.uniform(-0.1, 0.1, 3), 0, 1)
    else:
        gray = rng.uniform(0.25, 0.7)
        return np.clip(np.array([gray, gray, gray]) + rng.uniform(-0.05, 0.05, 3), 0, 1)


def _make_cornell_box(rng: np.random.Generator, grid_n: int) -> list[dict[str, np.ndarray]]:
    """Cornell box: floor + back wall + 2 colored side walls."""
    planes = []
    planes.append(make_gaussian_plane(
        center=np.array([0, 0, -0.5]), normal=np.array([0, 0, 1]),
        size=1.0, grid_n=grid_n, color=_pick_surface_color(rng, saturated=False), rng=rng,
    ))
    planes.append(make_gaussian_plane(
        center=np.array([0, 0.5, 0]), normal=np.array([0, -1, 0]),
        size=1.0, grid_n=grid_n, color=_pick_surface_color(rng, saturated=False), rng=rng,
    ))
    planes.append(make_gaussian_plane(
        center=np.array([-0.5, 0, 0]), normal=np.array([1, 0, 0]),
        size=1.0, grid_n=grid_n, color=_pick_surface_color(rng, saturated=True), rng=rng,
    ))
    planes.append(make_gaussian_plane(
        center=np.array([0.5, 0, 0]), normal=np.array([-1, 0, 0]),
        size=1.0, grid_n=grid_n, color=_pick_surface_color(rng, saturated=True), rng=rng,
    ))
    return planes


def _make_open_floor(rng: np.random.Generator, grid_n: int) -> list[dict[str, np.ndarray]]:
    """Large open floor only — clean pedestal look."""
    floor_color = _pick_surface_color(rng, saturated=rng.random() < 0.3)
    return [make_gaussian_plane(
        center=np.array([0, 0, -0.5]), normal=np.array([0, 0, 1]),
        size=1.4, grid_n=grid_n, color=floor_color, rng=rng,
    )]


def _make_backdrop(rng: np.random.Generator, grid_n: int) -> list[dict[str, np.ndarray]]:
    """Floor + single back wall — studio/photo backdrop style."""
    planes = []
    planes.append(make_gaussian_plane(
        center=np.array([0, 0, -0.5]), normal=np.array([0, 0, 1]),
        size=1.2, grid_n=grid_n, color=_pick_surface_color(rng, saturated=False), rng=rng,
    ))
    planes.append(make_gaussian_plane(
        center=np.array([0, 0.5, 0]), normal=np.array([0, -1, 0]),
        size=1.2, grid_n=grid_n, color=_pick_surface_color(rng, saturated=rng.random() < 0.5), rng=rng,
    ))
    return planes


def _make_corner(rng: np.random.Generator, grid_n: int) -> list[dict[str, np.ndarray]]:
    """Floor + back wall + one side wall — corner setup."""
    planes = []
    planes.append(make_gaussian_plane(
        center=np.array([0, 0, -0.5]), normal=np.array([0, 0, 1]),
        size=1.0, grid_n=grid_n, color=_pick_surface_color(rng, saturated=False), rng=rng,
    ))
    planes.append(make_gaussian_plane(
        center=np.array([0, 0.5, 0]), normal=np.array([0, -1, 0]),
        size=1.0, grid_n=grid_n, color=_pick_surface_color(rng, saturated=False), rng=rng,
    ))
    side = rng.choice([-1, 1])
    planes.append(make_gaussian_plane(
        center=np.array([side * 0.5, 0, 0]), normal=np.array([-side, 0, 0]),
        size=1.0, grid_n=grid_n, color=_pick_surface_color(rng, saturated=True), rng=rng,
    ))
    return planes


def _make_alcove(rng: np.random.Generator, grid_n: int) -> list[dict[str, np.ndarray]]:
    """Floor + back wall + 2 side walls + ceiling — enclosed alcove."""
    planes = []
    floor_color = _pick_surface_color(rng, saturated=False)
    planes.append(make_gaussian_plane(
        center=np.array([0, 0, -0.5]), normal=np.array([0, 0, 1]),
        size=1.0, grid_n=grid_n, color=floor_color, rng=rng,
    ))
    planes.append(make_gaussian_plane(
        center=np.array([0, 0.5, 0]), normal=np.array([0, -1, 0]),
        size=1.0, grid_n=grid_n, color=_pick_surface_color(rng, saturated=False), rng=rng,
    ))
    planes.append(make_gaussian_plane(
        center=np.array([-0.5, 0, 0]), normal=np.array([1, 0, 0]),
        size=1.0, grid_n=grid_n, color=_pick_surface_color(rng, saturated=True), rng=rng,
    ))
    planes.append(make_gaussian_plane(
        center=np.array([0.5, 0, 0]), normal=np.array([-1, 0, 0]),
        size=1.0, grid_n=grid_n, color=_pick_surface_color(rng, saturated=True), rng=rng,
    ))
    # Ceiling
    ceil_color = np.clip(floor_color + rng.uniform(0.1, 0.3, 3), 0, 1)
    planes.append(make_gaussian_plane(
        center=np.array([0, 0, 0.5]), normal=np.array([0, 0, -1]),
        size=1.0, grid_n=grid_n, color=ceil_color, rng=rng,
    ))
    return planes


# Background layout generators with weights
_BG_LAYOUTS = [
    (_make_cornell_box, 0.30),
    (_make_backdrop,    0.25),
    (_make_corner,      0.20),
    (_make_open_floor,  0.10),
    (_make_alcove,      0.15),
]


def make_background(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Create a randomized background from one of several layout styles."""
    grid_n = 20

    fns, weights = zip(*_BG_LAYOUTS)
    weights = np.array(weights) / sum(weights)
    layout_fn = fns[rng.choice(len(fns), p=weights)]

    planes = layout_fn(rng, grid_n)
    return {
        k: np.concatenate([p[k] for p in planes], axis=0)
        for k in planes[0]
    }


def tint_object(
    data: dict[str, np.ndarray],
    tint_color: np.ndarray,
    tint_strength: float,
) -> dict[str, np.ndarray]:
    """Apply a color tint to an object's Gaussians.

    Blends original color toward tint_color by tint_strength (0=no change, 1=full tint).
    """
    out = {k: v.copy() for k, v in data.items()}
    out["colors"] = (1 - tint_strength) * out["colors"] + tint_strength * tint_color[None, :]
    out["colors"] = np.clip(out["colors"], 0, 1).astype(np.float32)
    return out


def place_object(
    data: dict[str, np.ndarray],
    translation: np.ndarray,
    rotation_euler_deg: np.ndarray,
    scale: float,
) -> dict[str, np.ndarray]:
    """Place an object: scale, rotate (full 3D Euler XYZ), translate."""
    out = {k: v.copy() for k, v in data.items()}

    # Scale
    out["means"] *= scale
    out["scales"] *= scale

    # 3D rotation from Euler angles
    r = Rotation.from_euler("XYZ", rotation_euler_deg, degrees=True)
    rot_mat = r.as_matrix().astype(np.float32)
    out["means"] = out["means"] @ rot_mat.T

    # Compose rotation quaternion with existing Gaussian quaternions
    scipy_quat = r.as_quat()  # (x, y, z, w)
    place_quat = np.array([scipy_quat[3], scipy_quat[0], scipy_quat[1], scipy_quat[2]], dtype=np.float32)  # (w,x,y,z)
    out["rotations"] = quat_multiply(place_quat, out["rotations"])

    # Translate
    out["means"] += translation[None, :]

    return out


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply quaternions (w,x,y,z). q1 can be [4], q2 can be [N,4]."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1).astype(np.float32)


def generate_cameras(
    rng: np.random.Generator,
    num_views: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate camera views looking at the object area from a spherical shell.

    Camera looks at a point slightly below center (where objects sit, z ~ -0.15),
    from the front hemisphere (azimuth biased toward negative y to avoid
    looking at the back wall from behind).

    Returns (c2w_array [V,4,4], fov_array [V]).
    """
    c2ws = []
    fovs = []
    for _ in range(num_views):
        distance = rng.uniform(2.2, 2.6)
        fov = rng.uniform(30, 50)
        elevation = rng.uniform(15, 35)
        # Front hemisphere: camera at negative-y, looking toward back wall (+y)
        # Azimuth 270° = (0, -d, z). Vary ±45° to avoid looking along walls.
        azimuth = rng.uniform(-45, 45) + 270

        el_rad = np.radians(elevation)
        az_rad = np.radians(azimuth)
        x = distance * np.cos(el_rad) * np.cos(az_rad)
        y = distance * np.cos(el_rad) * np.sin(az_rad)
        z = distance * np.sin(el_rad)

        # Look at object area (slightly below center)
        look_target = np.array([0.0, 0.0, -0.15])
        c2w = look_at_to_c2w(np.array([x, y, z]), look_target)
        c2ws.append(c2w)
        fovs.append(fov)

    return np.stack(c2ws).astype(np.float32), np.array(fovs, dtype=np.float32)


# Tint color palette — saturated colors applied to the neutral-gray ModelNet objects
TINT_COLORS = [
    np.array([0.8, 0.2, 0.2]),   # red
    np.array([0.2, 0.7, 0.2]),   # green
    np.array([0.2, 0.3, 0.8]),   # blue
    np.array([0.8, 0.6, 0.1]),   # gold
    np.array([0.7, 0.2, 0.6]),   # magenta
    np.array([0.2, 0.6, 0.7]),   # cyan
    np.array([0.9, 0.4, 0.1]),   # orange
    np.array([0.5, 0.8, 0.3]),   # lime
    np.array([0.6, 0.4, 0.2]),   # brown
    np.array([0.8, 0.8, 0.8]),   # white/light (keeps object mostly gray)
    np.array([0.4, 0.4, 0.6]),   # steel blue
    np.array([0.7, 0.5, 0.5]),   # dusty rose
]


def compose_scene(
    objects: list[dict[str, np.ndarray]],
    rng: np.random.Generator,
    num_views: int,
) -> dict[str, np.ndarray]:
    """Compose a full scene: tint + place objects, add background, generate cameras."""
    parts = []

    n_obj = len(objects)
    if n_obj == 1:
        obj_scale = rng.uniform(0.55, 0.80)
        euler = np.array([rng.uniform(-15, 15), rng.uniform(-15, 15), rng.uniform(0, 360)])
        # Tint with a random color
        tint = TINT_COLORS[rng.integers(len(TINT_COLORS))]
        tint_strength = rng.uniform(0.2, 0.6)
        obj = tint_object(objects[0], tint, tint_strength)
        # Place near center, sitting on floor
        placed = place_object(obj, np.array([0, 0, -0.15]), euler, obj_scale)
        parts.append(placed)
    elif n_obj == 2:
        used_tints = rng.choice(len(TINT_COLORS), size=2, replace=False)
        for i, obj_data in enumerate(objects):
            obj_scale = rng.uniform(0.45, 0.65)
            euler = np.array([rng.uniform(-15, 15), rng.uniform(-15, 15), rng.uniform(0, 360)])
            tint = TINT_COLORS[used_tints[i]]
            tint_strength = rng.uniform(0.2, 0.6)
            obj = tint_object(obj_data, tint, tint_strength)
            # Offset to avoid overlap
            x_offset = -0.18 if i == 0 else 0.18
            y_offset = rng.uniform(-0.08, 0.08)
            placed = place_object(obj, np.array([x_offset, y_offset, -0.15]), euler, obj_scale)
            parts.append(placed)

    # Background (Cornell-box inspired)
    bg = make_background(rng)
    parts.append(bg)

    # Merge all Gaussians
    scene = {
        k: np.concatenate([p[k] for p in parts], axis=0)
        for k in parts[0]
    }

    # Cameras
    c2w, fov = generate_cameras(rng, num_views)

    scene["c2w"] = c2w
    scene["fov"] = fov
    return scene


def save_scene_h5(scene: dict[str, np.ndarray], path: Path):
    """Save a composed scene to H5 in our training format."""
    with h5py.File(path, "w") as f:
        f.create_dataset("means", data=scene["means"])
        f.create_dataset("scales", data=scene["scales"])
        f.create_dataset("rotations", data=scene["rotations"])
        f.create_dataset("colors", data=scene["colors"])
        f.create_dataset("opacities", data=scene["opacities"].reshape(-1, 1))
        f.create_dataset("c2w", data=scene["c2w"])
        f.create_dataset("fov", data=scene["fov"])


def main():
    parser = argparse.ArgumentParser(description="Compose training scenes")
    parser.add_argument("--objects_dir", type=Path, default=Path("data_v2/objects"))
    parser.add_argument("--output_dir", type=Path, default=Path("data_v2/h5s"))
    parser.add_argument("--num_scenes", type=int, default=1000)
    parser.add_argument("--views_per_scene", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Load all processed objects
    npz_files = sorted(args.objects_dir.glob("*.npz"))
    all_objects = []
    for npz_path in npz_files:
        data = dict(np.load(npz_path))
        all_objects.append((npz_path.stem, data))
        print(f"  Loaded {npz_path.stem}: {len(data['means'])} Gaussians")

    print(f"\n{len(all_objects)} objects loaded. Composing {args.num_scenes} scenes...\n")

    for i in range(args.num_scenes):
        # Pick 1-2 objects
        n_obj = 1 if rng.random() < 0.4 else 2
        chosen_idx = rng.choice(len(all_objects), size=n_obj, replace=False)
        chosen = [all_objects[idx][1] for idx in chosen_idx]
        chosen_names = [all_objects[idx][0] for idx in chosen_idx]

        scene = compose_scene(chosen, rng, args.views_per_scene)

        out_path = args.output_dir / f"scene_{i:04d}.h5"
        save_scene_h5(scene, out_path)

        n_gaussians = len(scene["means"])
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i + 1}/{args.num_scenes}] {chosen_names} → "
                  f"{n_gaussians} Gaussians, {args.views_per_scene} views")

    print(f"\nDone. {args.num_scenes} scenes saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
