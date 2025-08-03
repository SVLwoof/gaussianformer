from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class TransformConfig:
    """
    Configuration for transforming an object in the scene.
    """
    # Translation vector [x, y, z]
    translation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # Rotation as a quaternion [w, x, y, z]
    rotation: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    # Scale factors [x, y, z]
    scale: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    # Whether to normalize the object to a unit sphere.
    # This can still be useful to fit objects into a canonical volume
    normalize: bool = True


@dataclass
class MaterialConfig:
    """
    Simplified material config for Gaussians.
    """
    # We can use this to tint the base colors of the loaded Gaussians
    color_tint: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    # Opacity of the Gaussian.
    opacity_multiplier: float = 1.0


@dataclass
class ObjectConfig:
    """Configuration for a single Gaussian object in the scene."""
    # Relative path to the Gaussian PLY file.
    ply_path: str
    # Material of the object
    material: MaterialConfig = field(default_factory=MaterialConfig)
    # Transform of the object
    transform: TransformConfig = field(default_factory=TransformConfig)


@dataclass
class CameraConfig:
    """
    Camera configuration.
    """
    # Position of the camera
    position: List[float]
    # Look at point of the camera
    look_at: List[float]
    # Up vector of the camera
    up: List[float]
    # Field of view of the camera
    fov: float


@dataclass
class SceneConfig:
    """Top-level configuration for a scene composed of Gaussian objects."""
    # Name of the scene
    scene_name: str
    # Version of the scene description format
    version: str
    # Objects in the scene (including lighting)
    objects: Dict[str, ObjectConfig]
    # Cameras in the scene
    cameras: List[CameraConfig]
