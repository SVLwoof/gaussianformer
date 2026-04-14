from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    # --- Data paths ---
    gaussian_h5_dir: Path = Path("gaussian_training_h5s")
    renders_dir: Path = Path("training_renders")
    save_dir: Path = Path("checkpoints")

    # --- Weight transfer ---
    renderformer_model_id: str = "microsoft/renderformer-v1-base"

    # --- Phase 1: Input module only (frozen backbone) ---
    phase1_epochs: int = 50
    phase1_lr: float = 1e-3
    phase1_batch_size: int = 1

    # --- Phase 2: Full fine-tune ---
    phase2_epochs: int = 100
    phase2_lr: float = 5e-5
    phase2_batch_size: int = 1

    # --- Common training ---
    resolution: int = 256
    precision: str = "fp32"
    num_workers: int = 0
    grad_clip: float = 1.0
    min_lr_ratio: float = 0.01  # min_lr = lr * min_lr_ratio

    # --- Logging ---
    log_interval: int = 10
    save_interval: int = 10
