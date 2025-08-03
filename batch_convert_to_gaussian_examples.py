import json
import shutil
from pathlib import Path

import trimesh
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R


def convert_obj_to_gaussian_ply(obj_path: Path, ply_path: Path):
    """
    Loads an OBJ file and converts its vertices into a point cloud suitable for
    Gaussian Splatting, saving it as a PLY file using the `plyfile` library
    to ensure all custom attributes are correctly written.
    """
    try:
        # Load the mesh using trimesh. We only need the vertices.
        mesh: trimesh.Trimesh = trimesh.load_mesh(obj_path)

        if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
            print(f"Warning: No vertices found in {obj_path}. Skipping conversion.")
            return

        num_points = len(mesh.vertices)
        means = mesh.vertices

        # --- Define Default Gaussian Attributes ---
        scales = np.full((num_points, 3), 0.01, dtype=np.float32)
        rotations = np.zeros((num_points, 4), dtype=np.float32)
        rotations[:, 0] = 1.0  # Identity quaternion (w, x, y, z)
        colors = np.full((num_points, 3), 0.5, dtype=np.float32)
        opacities = np.full((num_points, 1), 0.8, dtype=np.float32)

        # --- Use plyfile to create a PLY file with custom attributes ---
        # Combine all attributes into a single structured numpy array
        dtypes: tuple[tuple[str, str], ...] = (
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
            ('opacity', 'f4')
        )

        elements = np.empty(num_points, dtype=dtypes)
        attributes = np.concatenate((means, scales, rotations, colors, opacities), axis=1)
        elements[:] = [tuple(row) for row in attributes]

        # Create the PlyElement and PlyData objects
        vertex_element = PlyElement.describe(elements, 'vertex')
        ply_data = PlyData([vertex_element])

        # Write to file
        ply_data.write(str(ply_path))

        print(f"Converted {obj_path} -> {ply_path} (with full Gaussian attributes)")

    except Exception as e:
        print(f"Failed to convert {obj_path} to {ply_path}: {e}")


def update_json_for_gaussians(json_path: Path):
    """
    Loads a JSON file formatted for the old mesh pipeline and converts it
    to the new format required for the Gaussian pipeline.
    """
    try:
        with json_path.open("r") as f:
            old_data = json.load(f)

        new_data = {
            "scene_name": old_data.get("scene_name", "unnamed_scene"),
            "version": "1.1-gaussian",
            "cameras": old_data.get("cameras", []),
            "objects": {}
        }

        for obj_key, old_obj_config in old_data.get("objects", {}).items():
            old_transform = old_obj_config.get("transform", {})

            euler_angles = old_transform.get("rotation", [0, 0, 0])
            rotation = R.from_euler('xyz', euler_angles, degrees=True).as_quat()
            quaternion = [rotation[3], rotation[0], rotation[1], rotation[2]]

            new_transform = {
                "translation": old_transform.get("translation", [0.0, 0.0, 0.0]),
                "rotation": quaternion,
                "scale": old_transform.get("scale", [1.0, 1.0, 1.0]),
                "normalize": old_transform.get("normalize", True)
            }

            new_material = {
                "color_tint": [1.0, 1.0, 1.0],
                "opacity_multiplier": 1.0
            }

            if "mesh_path" in old_obj_config:
                ply_path = Path(old_obj_config["mesh_path"]).with_suffix('.ply').as_posix()
            else:
                continue

            new_data["objects"][obj_key] = {
                "ply_path": ply_path,
                "material": new_material,
                "transform": new_transform
            }

        with json_path.open("w") as f:
            json.dump(new_data, f, indent=2)
        print(f"Successfully converted and updated {json_path} to Gaussian pipeline format.")

    except (json.JSONDecodeError, IOError, KeyError) as e:
        print(f"Could not process JSON file {json_path}: {e}")


def process_directory(src: Path, dst: Path):
    """
    Recursively copies a directory, converting .obj files to Gaussian .ply
    and updating .json files to the new format.
    """
    for item in src.iterdir():
        dst_item = dst / item.name
        if item.is_dir():
            dst_item.mkdir(exist_ok=True)
            process_directory(item, dst_item)
        else:
            # Copy the file first to the destination
            shutil.copy2(item, dst_item)

            # If it's an OBJ file, convert the copied file to a PLY
            if item.suffix.lower() == '.obj':
                ply_path = dst_item.with_suffix('.ply')
                convert_obj_to_gaussian_ply(dst_item, ply_path)
                # Remove the copied .obj file
                dst_item.unlink()

            # If it's a JSON file, update the paths in the copied file
            elif item.suffix.lower() == '.json':
                update_json_for_gaussians(dst_item)


def main():
    """
    Main function to set up directories and start the conversion process.
    """
    src_dir: Path = Path("examples")
    dst_dir: Path = Path("gaussian_examples")

    if not src_dir.is_dir():
        print(f"Error: Source directory '{src_dir}' not found.")
        print("Please ensure the 'examples' directory is in the same folder as this script.")
        return

    if dst_dir.exists():
        print(f"Removing existing destination directory: {dst_dir}")
        shutil.rmtree(dst_dir)

    dst_dir.mkdir(parents=True)
    print(f"Created destination directory: {dst_dir}")

    process_directory(src_dir, dst_dir)
    print("\nAll done! Your Gaussian-ready directory is:", dst_dir)


if __name__ == "__main__":
    main()
