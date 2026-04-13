"""
Generate randomized scene descriptor JSONs for RenderFormer training data.

Produces scene JSONs that follow the RenderFormer training-data constraints:
  - Camera distance to scene center: [1.5, 2.0]
  - FOV: [30, 60] degrees
  - Scene bounding box: [-0.5, 0.5] in x, y, z
  - Lights: up to 8 tri.obj sources, scale [2.0, 2.5], distance [2.1, 2.7], emission sum [2500, 5000]
  - Objects normalized into the scene bbox, with small offsets for multi-object scenes

Usage:
    python3 generate_training_data.py
"""

import json
import math
import random
from pathlib import Path


# ─── Directories ───────────────────────────────────────────────────────────────

ROOT_DIR: Path = Path(__file__).parent
EXAMPLES_DIR: Path = ROOT_DIR / "examples"
OBJECTS_DIR: Path = EXAMPLES_DIR / "objects"
OUTPUT_DIR: Path = ROOT_DIR / "training_examples"

# ─── Generation Settings ──────────────────────────────────────────────────────

NUM_SCENES: int = 30
NUM_CAMERAS_RANGE: tuple[int, int] = (1, 3)
NUM_OBJECTS_RANGE: tuple[int, int] = (1, 2)
NUM_LIGHTS_RANGE: tuple[int, int] = (1, 3)

# RenderFormer training-data constraints
CAMERA_DISTANCE_RANGE: tuple[float, float] = (1.5, 2.0)
CAMERA_FOV_RANGE: tuple[float, float] = (30.0, 60.0)
LIGHT_DISTANCE_RANGE: tuple[float, float] = (2.1, 2.7)
LIGHT_SCALE_RANGE: tuple[float, float] = (2.0, 2.5)
# Per-light emission: split the total budget [2500, 5000] across lights
LIGHT_EMISSION_TOTAL_RANGE: tuple[float, float] = (2500.0, 5000.0)


# ─── Object Discovery ─────────────────────────────────────────────────────────

# Objects that are part of pre-composed scenes and shouldn't be used standalone
EXCLUDE_PATTERNS: set[str] = {
    "wall", "plane", "tri",        # templates
    "obj0", "obj1", "obj2", "obj3", # compose parts
    "short-box", "tall-box",        # cbox internals
    "block1", "block2", "block3", "block4",  # veach-mis blocks
    "sphere1", "sphere2", "sphere3",         # veach-mis spheres
    "shell",                        # shader-ball shell (needs ball too)
    "table",                        # room table (context-dependent)
}


def discover_objects() -> list[tuple[str, str]]:
    """
    Find usable standalone .obj files under the objects directory.
    Returns list of (name, relative_path_from_examples_dir).
    """
    results: list[tuple[str, str]] = []
    for obj_path in OBJECTS_DIR.rglob("*.obj"):
        stem = obj_path.stem
        if any(pat in stem.lower() for pat in EXCLUDE_PATTERNS):
            continue
        rel = obj_path.relative_to(EXAMPLES_DIR).as_posix()
        results.append((stem, rel))
    return results


OBJECTS: list[tuple[str, str]] = discover_objects()


# ─── Geometry helpers ──────────────────────────────────────────────────────────

def random_point_on_sphere(r_min: float, r_max: float,
                           elevation_min: float = 0.1,
                           elevation_max: float = 0.9) -> list[float]:
    """
    Sample a random point on a spherical shell at distance [r_min, r_max].
    elevation_min/max control the fraction of the hemisphere (0=equator, 1=pole).
    Z-up coordinate system.
    """
    r = random.uniform(r_min, r_max)
    # Azimuth: full circle
    azimuth = random.uniform(0, 2 * math.pi)
    # Elevation: sample from a restricted band above the equator
    # Use cosine-weighted sampling for uniform area distribution
    cos_min = math.cos(math.pi / 2 * elevation_max)
    cos_max = math.cos(math.pi / 2 * elevation_min)
    cos_elev = random.uniform(cos_min, cos_max)
    elev = math.acos(cos_elev)

    x = r * math.sin(elev) * math.cos(azimuth)
    y = r * math.sin(elev) * math.sin(azimuth)
    z = r * cos_elev
    return [x, y, z]


# ─── Material Generation ──────────────────────────────────────────────────────

def random_material() -> dict:
    """
    Generate a random physically plausible material.
    Ensures diffuse + specular <= 1.0 per channel.
    """
    # Pick a material archetype, then add variation
    archetype = random.choice(["diffuse", "glossy", "metallic", "mixed"])

    if archetype == "diffuse":
        diffuse = [random.uniform(0.05, 0.9) for _ in range(3)]
        spec_val = random.uniform(0.0, 0.05)
        specular = [spec_val] * 3
        roughness = random.uniform(0.6, 1.0)
    elif archetype == "glossy":
        diffuse = [random.uniform(0.05, 0.5) for _ in range(3)]
        spec_val = random.uniform(0.3, min(1.0 - max(diffuse), 0.9))
        specular = [spec_val] * 3
        roughness = random.uniform(0.01, 0.3)
    elif archetype == "metallic":
        diffuse = [random.uniform(0.0, 0.05) for _ in range(3)]
        # Metallic: tinted specular
        spec_base = [random.uniform(0.5, 1.0) for _ in range(3)]
        specular = spec_base
        roughness = random.uniform(0.05, 0.4)
    else:  # mixed
        diffuse = [random.uniform(0.1, 0.6) for _ in range(3)]
        spec_val = random.uniform(0.1, min(1.0 - max(diffuse), 0.6))
        specular = [spec_val] * 3
        roughness = random.uniform(0.1, 0.8)

    # Clamp diffuse + specular <= 1.0
    for i in range(3):
        total = diffuse[i] + specular[i]
        if total > 1.0:
            specular[i] = 1.0 - diffuse[i]

    smooth_shading = random.choice([True, False])
    rand_diffuse_max = random.uniform(0.0, min(max(diffuse), 0.8)) if random.random() < 0.6 else 0.0
    rand_diffuse_type = random.choice(["per-triangle", "per-shading-group"])
    # Only set a seed when random_diffuse_max > 0, otherwise scene_mesh.py hits
    # np.random.randint(0, 0) which raises ValueError
    rand_seed = random.randint(1, 10000) if random.random() < 0.3 and rand_diffuse_max > 0 else None

    return {
        "diffuse": diffuse,
        "specular": specular,
        "roughness": roughness,
        "emissive": [0.0, 0.0, 0.0],
        "smooth_shading": smooth_shading,
        "rand_tri_diffuse_seed": rand_seed,
        "random_diffuse_max": rand_diffuse_max,
        "random_diffuse_type": rand_diffuse_type,
    }


def random_background_material() -> dict:
    """Material for walls/floor -- typically matte with possible color."""
    diffuse = [random.uniform(0.05, 0.8) for _ in range(3)]
    # Walls can occasionally be slightly specular
    spec_val = random.uniform(0.0, 0.3) if random.random() < 0.4 else 0.0
    specular = [min(spec_val, 1.0 - d) for d in diffuse]
    roughness = random.uniform(0.5, 1.0)

    return {
        "diffuse": diffuse,
        "specular": specular,
        "roughness": roughness,
        "emissive": [0.0, 0.0, 0.0],
        "smooth_shading": random.choice([True, False]),
        "rand_tri_diffuse_seed": None,
        "random_diffuse_max": random.uniform(0.0, 0.5) if random.random() < 0.5 else 0.0,
        "random_diffuse_type": random.choice(["per-triangle", "per-shading-group"]),
    }


# ─── Scene Element Builders ───────────────────────────────────────────────────

def build_backgrounds() -> dict[str, dict]:
    """Build the 4 background elements: floor plane + 3 walls."""
    bg_meshes = [
        ("background_0", "templates/backgrounds/plane.obj"),
        ("background_1", "templates/backgrounds/wall0.obj"),
        ("background_2", "templates/backgrounds/wall1.obj"),
        ("background_3", "templates/backgrounds/wall2.obj"),
    ]
    objects: dict[str, dict] = {}
    for name, mesh_path in bg_meshes:
        objects[name] = {
            "mesh_path": mesh_path,
            "transform": {
                "translation": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0],
                "scale": [0.5, 0.5, 0.5],
                "normalize": False,
            },
            "material": random_background_material(),
        }
    return objects


def build_primary_objects(count: int) -> dict[str, dict]:
    """
    Place 1-2 normalized objects in the scene.
    When placing 2 objects, offset them from center to reduce overlap.
    """
    chosen = random.sample(OBJECTS, min(count, len(OBJECTS)))
    objects: dict[str, dict] = {}

    if len(chosen) == 1:
        name, mesh_path = chosen[0]
        # Single object: center it, maybe small Z offset to sit on the floor
        z_offset = random.uniform(-0.05, 0.05)
        obj_scale = random.uniform(0.8, 1.2)
        objects[name] = {
            "mesh_path": mesh_path,
            "material": random_material(),
            "transform": {
                "translation": [0.0, 0.0, z_offset],
                "rotation": [0.0, 0.0, random.uniform(0, 360)],
                "scale": [obj_scale] * 3,
                "normalize": True,
            },
        }
    else:
        # 2 objects: place them on opposite sides of center
        # Random angle for the separation axis
        sep_angle = random.uniform(0, 2 * math.pi)
        sep_dist = random.uniform(0.15, 0.25)  # conservative to stay in bbox

        for i, (name, mesh_path) in enumerate(chosen):
            sign = 1.0 if i == 0 else -1.0
            x_off = sign * sep_dist * math.cos(sep_angle)
            y_off = sign * sep_dist * math.sin(sep_angle)
            z_off = random.uniform(-0.05, 0.05)
            # Smaller scale for multi-object scenes
            obj_scale = random.uniform(0.5, 0.8)
            objects[name] = {
                "mesh_path": mesh_path,
                "material": random_material(),
                "transform": {
                    "translation": [x_off, y_off, z_off],
                    "rotation": [0.0, 0.0, random.uniform(0, 360)],
                    "scale": [obj_scale] * 3,
                    "normalize": True,
                },
            }

    return objects


def _make_light(pos: list[float], rotation: list[float], emission: float) -> dict:
    """Build a single light object dict."""
    scale_val = random.uniform(*LIGHT_SCALE_RANGE)
    return {
        "mesh_path": "templates/lighting/tri.obj",
        "material": {
            "diffuse": [1.0, 1.0, 1.0],
            "specular": [0.0, 0.0, 0.0],
            "roughness": 1.0,
            "emissive": [emission] * 3,
            "smooth_shading": False,
            "rand_tri_diffuse_seed": None,
            "random_diffuse_max": 0.0,
            "random_diffuse_type": "per-shading-group",
        },
        "transform": {
            "translation": pos,
            "rotation": rotation,
            "scale": [scale_val] * 3,
            "normalize": False,
        },
    }


def build_lights(count: int) -> dict[str, dict]:
    """
    Place 1-3 lights. tri.obj faces -Z by default, so lights placed above
    the scene with rotation [0,0,0] naturally illuminate downward.

    Strategy (matching reference scenes):
      - Light 0 is always a "key light" placed high above the scene with
        zero rotation and at least 60% of the emission budget.
      - Additional fill lights are also placed above the scene (high Z)
        with zero rotation, at varied XY offsets.
    """
    total_emission = random.uniform(*LIGHT_EMISSION_TOTAL_RANGE)

    lights: dict[str, dict] = {}

    # Key light: overhead, gets the lion's share of emission
    key_fraction = random.uniform(0.6, 0.85) if count > 1 else 1.0
    key_emission = total_emission * key_fraction

    # Small XY jitter, high Z -- matches cbox/horse/fox patterns
    key_z = random.uniform(2.1, 2.5)
    key_xy_jitter = 0.3
    key_pos = [
        random.uniform(-key_xy_jitter, key_xy_jitter),
        random.uniform(-key_xy_jitter, key_xy_jitter),
        key_z,
    ]
    lights["light_0"] = _make_light(key_pos, [0.0, 0.0, 0.0], key_emission)

    # Fill lights: also placed above but at wider XY offsets
    remaining_emission = total_emission - key_emission
    for i in range(1, count):
        fill_fraction = remaining_emission / (count - i)  # split evenly among remaining
        fill_emission = fill_fraction * random.uniform(0.7, 1.3)
        remaining_emission -= fill_emission

        # Place at wider XY offset but still above the scene
        fill_z = random.uniform(1.5, 2.5)
        fill_xy_range = 1.8
        fill_pos = [
            random.uniform(-fill_xy_range, fill_xy_range),
            random.uniform(-fill_xy_range, fill_xy_range),
            fill_z,
        ]
        # Zero rotation -- tri.obj faces down by default
        lights[f"light_{i}"] = _make_light(fill_pos, [0.0, 0.0, 0.0], max(fill_emission, 100.0))

    return lights


def _random_camera_position(r_min: float, r_max: float) -> list[float]:
    """
    Place camera on a spherical shell at 5-35 degrees above the equator.
    Reference scenes use positions like [0,-2,0] (equator), [-1,-1,1] (~35 deg),
    [0,-2,0.26] (~7 deg). Avoids steep top-down angles that show the dark void.
    """
    r = random.uniform(r_min, r_max)
    azimuth = random.uniform(0, 2 * math.pi)

    # Elevation from equator: 5 to 35 degrees
    elev_deg = random.uniform(5.0, 35.0)
    elev_rad = math.radians(elev_deg)

    x = r * math.cos(elev_rad) * math.cos(azimuth)
    y = r * math.cos(elev_rad) * math.sin(azimuth)
    z = r * math.sin(elev_rad)
    return [x, y, z]


def build_cameras(count: int) -> list[dict]:
    """
    Generate camera views on a spherical shell looking at the scene center.
    """
    cameras: list[dict] = []
    for _ in range(count):
        pos = _random_camera_position(*CAMERA_DISTANCE_RANGE)
        # Slight jitter on look-at target
        look_at = [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)]
        fov = random.uniform(*CAMERA_FOV_RANGE)
        cameras.append({
            "position": pos,
            "look_at": look_at,
            "up": [0.0, 0.0, 1.0],
            "fov": fov,
        })
    return cameras


# ─── Scene Assembly ────────────────────────────────────────────────────────────

def generate_scene(index: int) -> dict:
    """Assemble a complete scene descriptor."""
    num_objects = random.randint(*NUM_OBJECTS_RANGE)
    num_lights = random.randint(*NUM_LIGHTS_RANGE)
    num_cameras = random.randint(*NUM_CAMERAS_RANGE)

    objects: dict[str, dict] = {}
    objects.update(build_backgrounds())
    primary = build_primary_objects(num_objects)
    objects.update(primary)
    objects.update(build_lights(num_lights))

    obj_names = "_".join(primary.keys())
    scene = {
        "scene_name": f"scene_{index:04d}_{obj_names}",
        "version": "1.0",
        "objects": objects,
        "cameras": build_cameras(num_cameras),
    }
    return scene


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not OBJECTS:
        print(f"Error: No objects found in {OBJECTS_DIR}")
        return

    print(f"Discovered {len(OBJECTS)} usable objects:")
    for name, path in sorted(OBJECTS):
        print(f"  {name}: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(NUM_SCENES):
        scene = generate_scene(i)
        output_path = OUTPUT_DIR / f"scene_{i:04d}.json"
        with open(output_path, "w") as f:
            json.dump(scene, f, indent=4)

    print(f"\nGenerated {NUM_SCENES} scenes in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
