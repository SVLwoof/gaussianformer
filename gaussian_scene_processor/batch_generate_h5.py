import json
from pathlib import Path
from dacite import from_dict, Config
from to_h5_gaussian import save_gaussians_to_h5
from scene_gaussian import generate_scene_gaussians
from scene_config_gaussian import SceneConfig

PARENT_DIR: Path = Path(__file__).parent
PROJECT_DIR: Path = PARENT_DIR.parent


def process_single_scene(json_path: Path, output_dir: Path):
    """
    Reads a scene JSON, loads the referenced PLYs, applies transforms,
    and saves the result as an .h5 file in the output directory.
    """
    # Create an output filename based on the JSON stem (e.g., 'scene.json' -> 'scene.h5')
    output_h5_path = output_dir / f"{json_path.stem}.h5"

    # Skip if already exists (optional, currently disabled)
    # if output_h5_path.exists():
    #     print(f"Skipping {json_path.name}, .h5 already exists.")
    #     return

    print(f"\nProcessing Scene: {json_path.name}")

    with json_path.open('r') as f:
        config_data = json.load(f)

    # Parse config using the existing class structure
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

    # 1. Generate the raw Gaussian data (numpy arrays)
    try:
        gaussian_data = generate_scene_gaussians(scene_config, scene_config_dir)
    except Exception as e:
        print(f"  [Error] Failed to generate gaussians: {e}")
        return

    if not gaussian_data:
        print("  [Warning] No data generated. Skipping save.")
        return

    # 2. Save to HDF5
    try:
        save_gaussians_to_h5(scene_config, gaussian_data, output_h5_path)
        print(f"  [Success] Saved to {output_h5_path}")
    except Exception as e:
        print(f"  [Error] Failed to write HDF5: {e}")


def main():
    # Directory containing the converted JSONs and PLYs
    examples_dir = PROJECT_DIR / 'gaussian_training_examples'

    # New output directory for H5 files
    output_dir = PROJECT_DIR / 'gaussian_training_h5s'

    if not examples_dir.exists():
        print(f"Directory '{examples_dir}' not found. Did you run batch_convert_to_gaussian_examples.py?")
        return

    # Create the output directory if it doesn't exist
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Output directory exists: {output_dir}")

    # Find all JSON files in the directory
    scene_files = list(examples_dir.glob("**/*.json"))

    if not scene_files:
        print(f"No JSON files found in {examples_dir}")
        return

    print(f"Found {len(scene_files)} scenes to process...")

    for scene_file in scene_files:
        process_single_scene(scene_file, output_dir)

    print("\nBatch processing complete.")


if __name__ == "__main__":
    main()
