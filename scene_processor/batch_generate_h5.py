import json
from pathlib import Path
from dacite import from_dict, Config
from to_h5 import save_to_h5
from scene_mesh import generate_scene_mesh
from scene_config import SceneConfig


PARENT_DIR: Path = Path(__file__).parent
PROJECT_DIR: Path = PARENT_DIR.parent



def process_single_scene(json_path: Path):
    """
    Reads a scene JSON, loads the meshes, applies transforms,
    and saves the result as an .h5 file in the same directory.
    """
    output_h5_path = json_path.with_suffix('.h5')

    print(f"\nProcessing Scene: {json_path.name}")

    with json_path.open('r') as f:
        config_data = json.load(f)

    # Parse config using the standard RenderFormer SceneConfig
    try:
        scene_config = from_dict(
            data_class=SceneConfig,
            data=config_data,
            config=Config(check_types=False)
        )
    except Exception as e:
        print(f"  [Error] Failed to parse config: {e}")
        return

    scene_config_dir = json_path.parent

    # 1. Generate the mesh data (vertices/faces) for RenderFormer
    try:
        # Note: generate_scene_meshes returns a dict of mesh data suitable for the renderer
        scene_data = generate_scene_mesh(scene_config, scene_config_dir)
    except Exception as e:
        print(f"  [Error] Failed to generate meshes: {e}")
        return

    if not scene_data:
        print("  [Warning] No data generated. Skipping save.")
        return

    # 2. Save to HDF5
    try:
        save_to_h5(scene_config, scene_data, output_h5_path)
        print(f"  [Success] Saved to {output_h5_path.name}")
    except Exception as e:
        print(f"  [Error] Failed to write HDF5: {e}")


def main():
    # Directory containing the original training examples
    examples_dir = Path("training_examples")

    if not examples_dir.exists():
        print(f"Directory '{examples_dir}' not found.")
        return

    # Find all JSON files in the directory
    scene_files = list(examples_dir.glob("**/*.json"))

    if not scene_files:
        print(f"No JSON files found in {examples_dir}")
        return

    print(f"Found {len(scene_files)} scenes to process for RenderFormer...")

    for scene_file in scene_files:
        process_single_scene(scene_file)

    print("\nBatch processing complete. You can now run the rendering script.")


if __name__ == "__main__":
    main()
