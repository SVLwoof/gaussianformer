"""Checkpoint helpers shared by training, LoRA training and every evaluation loader.

A checkpoint written since 2026-09-17 carries `config` (the GaussianFormerConfig it was trained
with), so a loader needs only the path; older checkpoints fall back to the pe_type default plus
whatever `--model_cfg` the caller passes.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from gaussianformer.models.config import GaussianFormerConfig
from gaussianformer.models.gaussianformer import GaussianFormer


def config_from_checkpoint(ckpt: dict, pe_type: str = "rope",
                           overrides: list[str] | None = None) -> GaussianFormerConfig:
    base = GaussianFormerConfig(**ckpt["config"]) if "config" in ckpt else GaussianFormerConfig(pe_type=pe_type)
    return base.with_overrides(overrides)


def load_seed(module: nn.Module, state_dict: dict, *, allow_new: bool = True) -> list[str]:
    """Warm-start `module` from a seed's state dict; returns the keys the seed lacked.

    RoPE frequency tables (*.freqs) are dropped only when their shape changed (a different
    rotary dim); a table the model does not have at all is a real mismatch and raises. Tensors
    the seed lacks (`allow_new`) must be zero-initialised, so the model IS the seed at load.
    """
    own = module.state_dict()
    sd = {k: v for k, v in state_dict.items()
          if not (k.endswith(".freqs") and k in own and own[k].shape != v.shape)}
    missing, unexpected = module.load_state_dict(sd, strict=False)
    assert not unexpected, f"seed has keys the model lacks: {list(unexpected)[:5]}"
    if not allow_new:
        assert all(k.endswith(".freqs") for k in missing), f"seed lacks tensors: {missing[:5]}"
    nonzero = [k for k in missing if not k.endswith(".freqs") and own[k].abs().sum() > 0]
    assert not nonzero, f"missing seed keys are NOT zero-init (would damage the warm init): {nonzero[:5]}"
    return list(missing)


def load_checkpoint(path: Path, pe_type: str = "rope", overrides: list[str] | None = None,
                    allow_new: bool = False) -> tuple[GaussianFormer, dict]:
    """Model built from the checkpoint's stored config (+ overrides) with its weights loaded."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = GaussianFormer(config_from_checkpoint(ckpt, pe_type, overrides))
    load_seed(model, ckpt["model_state_dict"], allow_new=allow_new)
    return model, ckpt
