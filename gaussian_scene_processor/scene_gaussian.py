from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from scene_config_gaussian import SceneConfig


def normalize_points_to_unit_sphere(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Normalize a point cloud to fit in a unit sphere centered at the origin."""
    center: np.ndarray = points.mean(axis=0)
    points_centered: np.ndarray = points - center
    max_dist: float = np.linalg.norm(points_centered, axis=1).max()

    if max_dist == 0:
        return points, center, 1.0

    points_normalized: np.ndarray = points_centered / max_dist
    return points_normalized, center, max_dist


def generate_scene_gaussians(scene_config: SceneConfig, scene_config_dir: Path) -> dict:
    """
    Generate a combined set of Gaussian attributes from a scene configuration.
    It loads PLY files, applies transformations, and returns a dictionary of NumPy arrays.
    """

    # Lists to aggregate attributes from all objects
    all_means: list = []
    all_scales: list = []
    all_quats: list = []
    all_colors: list = []
    all_opacities: list = []

    for obj_key, obj_config in scene_config.objects.items():
        print(f"Processing object: {obj_key}")
        ply_full_path: Path = scene_config_dir / obj_config.ply_path

        try:
            point_cloud = trimesh.load(ply_full_path, process=False)
            if not isinstance(point_cloud, trimesh.PointCloud) or len(point_cloud.vertices) == 0:
                print(f"  Warning: Could not load a valid PointCloud from {ply_full_path}. Skipping.")
                continue
        except Exception as e:
            print(f"  Error loading {ply_full_path}: {e}. Skipping.")
            continue

        # --- 1. Extract initial Gaussian attributes from the PLY ---
        means = point_cloud.vertices.copy()
        num_points: int = len(means)

        # Robustly access custom vertex data from trimesh metadata
        try:
            vertex_data = point_cloud.metadata['_ply_raw']['vertex']['data']
        except KeyError:
            # Fallback if metadata is missing
            print(f"  Warning: No PLY metadata found for {obj_key}. Using defaults.")
            vertex_data = np.zeros(num_points, dtype=[])

        # Default values in case attributes are not found in the PLY file
        scales = np.full((num_points, 3), 0.01)
        quats = np.zeros((num_points, 4))
        quats[:, 0] = 1.0  # Identity quaternion (w, x, y, z)
        colors = np.full((num_points, 3), 0.5)
        opacities = np.full((num_points, 1), 0.8)

        # Check for attributes in the loaded vertex data.
        # These will likely be FALSE if the PLY files are not generated correctly.
        if 'scale_0' in vertex_data.dtype.names:
            scales = np.vstack([vertex_data['scale_0'], vertex_data['scale_1'], vertex_data['scale_2']]).T

        if 'rot_0' in vertex_data.dtype.names:
            quats = np.vstack(
                [vertex_data['rot_0'], vertex_data['rot_1'], vertex_data['rot_2'], vertex_data['rot_3']]).T

        if 'f_dc_0' in vertex_data.dtype.names:
            colors = np.vstack([vertex_data['f_dc_0'], vertex_data['f_dc_1'], vertex_data['f_dc_2']]).T

        if 'opacity' in vertex_data.dtype.names:
            opacities = vertex_data['opacity'][:, np.newaxis]

        # --- 2. Apply transformations from the scene config ---
        transform = obj_config.transform

        # Get transformation components and force the `float32` type to avoid division errors
        obj_translation = np.array(transform.translation, dtype=np.float32)
        obj_scale = np.array(transform.scale, dtype=np.float32)

        # Rotation: Scipy expects (x, y, z, w)
        obj_rotation = R.from_quat(transform.rotation[1:] + [transform.rotation[0]])

        # Apply transformations to each Gaussian
        # Position (mean): Rotate, then scale, then translate
        transformed_means = obj_rotation.apply(means) * obj_scale

        if transform.normalize:
            transformed_means, center, scale_factor = normalize_points_to_unit_sphere(transformed_means)
            # Adjust object scale and translation to reflect normalization
            if scale_factor > 1e-6:
                obj_scale /= scale_factor
                obj_translation = (obj_translation - center) / scale_factor

        transformed_means += obj_translation

        # Scale: Compose object scale with a Gaussian scale
        transformed_scales = scales * obj_scale

        # Reorder quaternion from (w, x, y, z) to (x, y, z, w) for scipy multiplication
        quat_xyzw = np.concatenate([quats[:, 1:], quats[:, :1]], axis=-1)
        initial_quats_scipy = R.from_quat(quat_xyzw)

        composed_rotation = obj_rotation * initial_quats_scipy
        transformed_quats_scipy = composed_rotation.as_quat()
        transformed_quats = transformed_quats_scipy[:, [3, 0, 1, 2]]  # Convert back to (w, x, y, z)

        transformed_colors = colors * np.array(obj_config.material.color_tint)
        transformed_opacities = opacities * obj_config.material.opacity_multiplier

        # --- 3. Aggregate results ---
        all_means.append(transformed_means)
        all_scales.append(transformed_scales)
        all_quats.append(transformed_quats)
        all_colors.append(np.clip(transformed_colors, 0, 1))
        all_opacities.append(np.clip(transformed_opacities, 0, 1))

    if not all_means:
        print("Warning: No objects were processed. Returning empty data.")
        return {}

    # --- 4. Concatenate all attributes into single NumPy arrays ---
    final_data = {
        "means": np.concatenate(all_means, axis=0),
        "scales": np.concatenate(all_scales, axis=0),
        "rotations": np.concatenate(all_quats, axis=0),
        "colors": np.concatenate(all_colors, axis=0),
        "opacities": np.concatenate(all_opacities, axis=0),
    }

    print(f"Scene generation complete. Total Gaussians: {len(final_data['means'])}")
    return final_data
