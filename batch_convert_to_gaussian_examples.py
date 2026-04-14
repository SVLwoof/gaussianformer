import json
import shutil
from pathlib import Path

import trimesh
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R


def get_rotation_to_align_vectors(vec_a, vec_b):
    """
    Returns a quaternion (w, x, y, z) that rotates vec_a to align with vec_b.
    Typically used to align the Z-axis (0,0,1) with the surface normal.
    """
    # Normalize input vectors
    vec_a = vec_a / np.linalg.norm(vec_a)
    # Ensure vec_b is normalized and has the right shape
    vec_b = vec_b / (np.linalg.norm(vec_b, axis=1, keepdims=True) + 1e-8)

    # We want to find Rotation R such that R * Z_axis = Normal (vec_b)
    # We construct a rotation matrix basis manually to ensure robustness

    z_axis = vec_b

    # Create a temporary vector to help find the X axis via cross-product
    # We choose a vector that is not parallel to Z.
    tmp = np.zeros_like(z_axis)
    tmp[:, 0] = 1

    # If the normal is too close to the global X axis, use global Y axis instead
    mask = np.abs(z_axis[:, 0]) > 0.9
    tmp[mask, 0] = 0
    tmp[mask, 1] = 1

    x_axis = np.cross(tmp, z_axis)
    x_axis /= (np.linalg.norm(x_axis, axis=1, keepdims=True) + 1e-8)

    y_axis = np.cross(z_axis, x_axis)

    # Construct Rotation Matrix [X | Y | Z] (Stack along the last axis)
    rot_matrices = np.stack([x_axis, y_axis, z_axis], axis=2)

    r = R.from_matrix(rot_matrices)

    # Scipy returns (x, y, z, w)
    quats = r.as_quat()

    # Convert to (w, x, y, z) for 3D Gaussian Splatting standard
    return quats[:, [3, 0, 1, 2]]


def convert_obj_to_gaussian_ply(obj_path: Path, ply_path: Path, num_samples: int | None = None):
    """
    Converts an OBJ to a Gaussian PLY by sampling the surface,
    extracting colors, and aligning rotations to normals.
    """
    try:
        # Load mesh. 'force="mesh"' ensures we get a Trimesh object even if it's a scene
        mesh = trimesh.load(obj_path, force='mesh')

        if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
            print(f"Warning: No vertices found in {obj_path}. Skipping.")
            return

        if num_samples is None:
            num_samples = max(100, min(5000, len(mesh.vertices) // 2))

        print(f"  Processing {obj_path.name} ({num_samples} samples)...")

        # --- 1. Surface Sampling ---
        # sample_surface returns (points, face_index)
        # Try to sample with color if the mesh has textures
        try:
            points, face_indices, colors_sampled = trimesh.sample.sample_surface(mesh, num_samples, sample_color=True)
            # trimesh colors are RGBA uint8, convert to float 0-1
            colors = colors_sampled[:, :3] / 255.0
        except Exception:
            # Fallback for meshes without textures or older trimesh versions
            points, face_indices = trimesh.sample.sample_surface(mesh, num_samples)

            # Try to get color from visual attributes if sample_color failed
            if hasattr(mesh.visual, 'face_colors') and len(mesh.visual.face_colors) > 0:
                colors = mesh.visual.face_colors[face_indices][:, :3] / 255.0
            elif hasattr(mesh.visual, 'vertex_colors') and len(mesh.visual.vertex_colors) > 0:
                # Fallback to grey if vertex mapping is too complex for this script
                colors = np.full((len(points), 3), 1.0, dtype=np.float32)
            else:
                # Default grey
                colors = np.full((len(points), 3), 1.0, dtype=np.float32)

        num_points = len(points)
        means = points.astype(np.float32)

        # --- 2. Calculate Scales ---
        # Heuristic: Scale is related to the surface area covered by each point.
        # Radius ~ sqrt(Area / N)
        avg_area_per_point = mesh.area / (num_points + 1e-8)
        radius = np.sqrt(avg_area_per_point) * 1.5  # Slight overlap factor to prevent holes

        # Anisotropic: flat discs on the surface (thin along normal direction).
        # The rotation quaternion aligns local Z to the surface normal,
        # so scale_2 (Z) should be much thinner than scale_0/scale_1 (tangent plane).
        NORMAL_SCALE_FACTOR = 0.1
        scales = np.column_stack([
            np.full(num_points, radius, dtype=np.float32),
            np.full(num_points, radius, dtype=np.float32),
            np.full(num_points, radius * NORMAL_SCALE_FACTOR, dtype=np.float32),
        ])

        # --- 3. Calculate Rotations ---
        # Align the Gaussian's Z-axis with the surface normal
        normals = mesh.face_normals[face_indices]
        rotations = get_rotation_to_align_vectors(np.array([0, 0, 1]), normals).astype(np.float32)

        # --- 4. Opacities ---
        # Solid objects should be opaque.
        opacities = np.full((num_points, 1), 1.0, dtype=np.float32)

        # --- 5. Construct PLY Data ---
        dtypes = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
            ('opacity', 'f4')
        ]

        # Robustly fill elements by column instead of list-of-tuples
        elements = np.empty(num_points, dtype=dtypes)

        elements['x'] = means[:, 0]
        elements['y'] = means[:, 1]
        elements['z'] = means[:, 2]

        elements['scale_0'] = scales[:, 0]
        elements['scale_1'] = scales[:, 1]
        elements['scale_2'] = scales[:, 2]

        elements['rot_0'] = rotations[:, 0]
        elements['rot_1'] = rotations[:, 1]
        elements['rot_2'] = rotations[:, 2]
        elements['rot_3'] = rotations[:, 3]

        elements['f_dc_0'] = colors[:, 0]
        elements['f_dc_1'] = colors[:, 1]
        elements['f_dc_2'] = colors[:, 2]

        elements['opacity'] = opacities[:, 0]

        vertex_element = PlyElement.describe(elements, 'vertex')
        ply_data = PlyData([vertex_element])
        ply_data.write(str(ply_path))

        print(f"  -> Converted to {ply_path.name} ({num_points} Gaussians)")

    except Exception as e:
        print(f"Failed to convert {obj_path}: {e}")


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

            # Convert Euler angles (degrees) to Quaternion (x, y, z, w)
            euler_angles = old_transform.get("rotation", [0, 0, 0])
            rotation = R.from_euler('xyz', euler_angles, degrees=True).as_quat()

            # Reorder to (w, x, y, z) for consistency
            quaternion = [rotation[3], rotation[0], rotation[1], rotation[2]]

            new_transform = {
                "translation": old_transform.get("translation", [0.0, 0.0, 0.0]),
                "rotation": quaternion,
                "scale": old_transform.get("scale", [1.0, 1.0, 1.0]),
                "normalize": old_transform.get("normalize", True)
            }

            old_material = old_obj_config.get("material", {})
            new_material = {
                "color_tint": old_material.get("diffuse", [1.0, 1.0, 1.0]),
                "opacity_multiplier": 1.0
            }

            if "mesh_path" in old_obj_config:
                # Point to the new .ply file instead of .obj
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
        print(f"Updated config: {json_path.name}")

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
                # Remove the raw .obj file from the destination
                dst_item.unlink()

            # If it's a JSON file, update the paths and structure
            elif item.suffix.lower() == '.json':
                update_json_for_gaussians(dst_item)


def main():
    """
    Main function to set up directories and start the conversion process.
    """
    src_dir: Path = Path("training_examples")
    dst_dir: Path = Path("gaussian_training_examples")

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
