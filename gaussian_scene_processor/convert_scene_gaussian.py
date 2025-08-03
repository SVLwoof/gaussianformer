import json
import argparse
from pathlib import Path
from dacite import from_dict, Config

from to_h5_gaussian import save_gaussians_to_h5
from scene_gaussian import generate_scene_gaussians
from scene_config_gaussian import SceneConfig


def main():
    parser = argparse.ArgumentParser(description="Convert a Gaussian scene config to an HDF5 file.")
    parser.add_argument("scene_config_path", type=Path, help="Path to the scene config JSON file.")
    parser.add_argument("--output_h5_path", type=Path,
                        help="Output path for .h5 file. If not provided, it will be saved next to the config file.",
                        default=None)
    args: argparse.Namespace = parser.parse_args()


    scene_config_path: Path = args.scene_config_path
    if not scene_config_path.is_file():
        print(f"Error: Scene config file not found at {scene_config_path}")
        return

    with scene_config_path.open('r') as f:
        config_data = json.load(f)

    try:
        scene_config = from_dict(
            data_class=SceneConfig,
            data=config_data,
            config=Config(check_types=False)
        )
    except Exception as e:
        print(f"Error parsing scene config: {e}")
        return

    scene_config_dir: Path = scene_config_path.parent

    if args.output_h5_path is None:
        output_h5_path = scene_config_path.with_suffix('.h5')
    else:
        output_h5_path = Path(args.output_h5_path)

    # Generate Gaussian Data
    print("Generating Gaussian scene data...")
    gaussian_data: dict = generate_scene_gaussians(scene_config, scene_config_dir)

    # Save Data to HDF5
    if gaussian_data:
        print(f"Saving data to HDF5 file: {output_h5_path}")
        save_gaussians_to_h5(scene_config, gaussian_data, output_h5_path)
        print("Conversion process complete.")
    else:
        print("Skipping HDF5 generation due to errors in processing objects.")


if __name__ == "__main__":
    main()
