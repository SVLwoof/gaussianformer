"""Scene configurations for external-scene evaluation.

Each SceneConfig captures the normalization, orbit-camera, and pruning settings
that differ across scenes. The eval pipeline in `run_scene.py` is otherwise
scene-agnostic.

Tomatoes (real-world capture from superspl.at) uses a fixed 2.0x scale and a
180-deg X rotation to flip the source PLY's gravity-down convention.

Objaverse scenes (from the ShapeSplats/Objaverse_Splats HF dataset) are already
gravity-up and are normalized so max(|pos|) == 0.45, matching the training
distribution produced by data_v9/process_objaverse.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SceneConfig:
    name: str

    # Normalization. Set exactly one of (norm_scale, norm_target_aabb).
    norm_scale: float | None = None        # fixed scale (tomatoes: 2.0)
    norm_target_aabb: float | None = None  # auto-scale so max(|pos|) == this (objaverse: 0.45)
    median_center: bool = True
    flip_x_rotation: bool = False          # 180-deg X rotation (flips gravity-down to up)

    # Orbit camera (shared across scoring and final-eval renders).
    orbit_radius: float = 1.7
    orbit_fov_deg: float = 45.0
    n_views: int = 14
    resolution: int = 512

    # LightGaussian scoring.
    score_views: int = 64
    score_resolution: int = 512
    gamma: float = 0.1

    # Pruning targets.
    n_variants: tuple[int, ...] = (5000, 10000, 20000, 30000)

    # Optional Objaverse UID + chunk for the prep_objaverse_scene.py downloader.
    objaverse_uid: str | None = None
    objaverse_chunk: str | None = None

    def __post_init__(self) -> None:
        if (self.norm_scale is None) == (self.norm_target_aabb is None):
            raise ValueError(
                f"SceneConfig({self.name!r}): set exactly one of "
                "norm_scale or norm_target_aabb."
            )

    @property
    def scene_dir(self) -> Path:
        return Path("data_external") / self.name

    @property
    def raw_ply(self) -> Path:
        return self.scene_dir / "raw.ply"

    @property
    def h5_dir(self) -> Path:
        return self.scene_dir / "h5"

    @property
    def renders_dir(self) -> Path:
        return self.scene_dir / "renders"

    @property
    def full_gsplat_dir(self) -> Path:
        return self.renders_dir / "gsplat_full"


SCENES: dict[str, SceneConfig] = {
    "tomatoes": SceneConfig(
        name="tomatoes",
        norm_scale=2.0,
        flip_x_rotation=True,
    ),
    "house": SceneConfig(
        name="house",
        norm_target_aabb=0.45,
        objaverse_uid="4272ba78929d45b8ba250079d292c7b8",
        objaverse_chunk="000-031",
    ),
    "dragon": SceneConfig(
        name="dragon",
        norm_target_aabb=0.45,
        objaverse_uid="7cddac29d2644be4abdec7e558450991",
        objaverse_chunk="000-045",
    ),
    "cartoon": SceneConfig(
        name="cartoon",
        norm_target_aabb=0.45,
        objaverse_uid="235e2e4cc56c479ba638b8952c2105a5",
        objaverse_chunk="000-091",
    ),
}
