from pathlib import Path
import h5py
import numpy as np
from scene_config_gaussian import SceneConfig


def look_at_to_c2w(camera_position, target_position=(0.0, 0.0, 0.0), up_dir=(0.0, 0.0, 1.0)) -> np.ndarray:
    """
    Look at transform matrix

    :param camera_position: camera position
    :param target_position: target position, default is origin
    :param up_dir: up vector, default is z-axis up
    :return: camera to world matrix
    """
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
    look_at_transform = np.matmul(rotation_transform, translation_transform)
    return np.linalg.inv(look_at_transform)


def save_gaussians_to_h5(scene_config: SceneConfig, gaussian_data: dict, output_h5_path: Path):
    """
    Saves the aggregated Gaussian data and camera configurations to an HDF5 file.
    """
    if not gaussian_data:
        print("Error: Gaussian data is empty. Cannot save to H5.")
        return

    # Process camera data
    all_c2w = []
    all_fov = []
    for camera_config in scene_config.cameras:
        c2w = look_at_to_c2w(camera_config.position, camera_config.look_at, camera_config.up)
        all_c2w.append(c2w)
        all_fov.append(camera_config.fov)

    all_c2w = np.stack(all_c2w)
    all_fov = np.array(all_fov)

    # Save to HDF5 file
    output_h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_h5_path, "w") as f:
        # Save Gaussian attributes
        f.create_dataset("means", data=gaussian_data["means"].astype(np.float32), compression="gzip")
        f.create_dataset("scales", data=gaussian_data["scales"].astype(np.float32), compression="gzip")
        f.create_dataset("rotations", data=gaussian_data["rotations"].astype(np.float32), compression="gzip")
        f.create_dataset("colors", data=gaussian_data["colors"].astype(np.float32), compression="gzip")
        f.create_dataset("opacities", data=gaussian_data["opacities"].astype(np.float32), compression="gzip")

        # Save camera attributes
        f.create_dataset("c2w", data=all_c2w.astype(np.float32), compression="gzip")
        f.create_dataset("fov", data=all_fov.astype(np.float32), compression="gzip")

    print(f"Successfully saved Gaussian scene and cameras to: {output_h5_path}")
