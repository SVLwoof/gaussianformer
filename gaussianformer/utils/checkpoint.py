from pathlib import Path

import torch
from torch import nn

from gaussianformer.models.config import GaussianFormerConfig
from gaussianformer.models.gaussianformer import GaussianFormer


def load_seed(module: nn.Module, state_dict: dict, allow_new: bool = True) -> list[str]:
    """Warm-start `module` from `state_dict`; returns the keys the state dict lacked.

    New tensors (`allow_new`) must be zero-initialised so the model computes the same function as the
    seed at load; RoPE frequency tables are derived from the config and may be new or change shape.
    """
    own = module.state_dict()
    sd = {k: v for k, v in state_dict.items() if not (k.endswith(".freqs") and k in own and own[k].shape != v.shape)}
    missing, unexpected = module.load_state_dict(sd, strict=False)
    assert not unexpected, f"state dict has keys the model lacks: {list(unexpected)[:5]}"
    new = [k for k in missing if not k.endswith(".freqs")]
    assert allow_new or not new, f"state dict lacks tensors: {new[:5]}"
    nonzero = [k for k in new if own[k].abs().sum() > 0]
    assert not nonzero, f"new tensors must be zero-initialised: {nonzero[:5]}"
    return list(missing)


def load_checkpoint(path: Path, overrides: list[str] | None = None,
                    allow_new: bool = False) -> tuple[GaussianFormer, dict]:
    """Model built from the checkpoint's stored config (plus `key=value` overrides), weights loaded."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    config = GaussianFormerConfig(**ckpt["config"]).with_overrides(overrides)
    model = GaussianFormer(config)
    load_seed(model, ckpt["model_state_dict"], allow_new=allow_new)
    return model, ckpt
