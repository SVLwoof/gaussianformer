"""Minimal LoRA (Hu et al. 2021) for nn.Linear, no external dependency.

W = W0 + (alpha / r) * B @ A, with W0 frozen, A ~ kaiming-uniform, B = 0 so the wrapped
model is exactly the base model at init. Adapter checkpoints hold only A/B plus the
config needed to re-wrap a base checkpoint (`lora_state_dict` / `load_lora`).
"""
from __future__ import annotations

import math
import re

import torch
import torch.nn as nn

DEFAULT_TARGETS = r"\.(in_proj|out_proj|q_proj|k_proj|v_proj)$"
"""Attention projections of both stages (60 matrices in the V18 architecture)."""


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # A/B are fp32 params; under autocast the matmuls run in the autocast dtype, like base.
        return self.base(x) + (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling

    def merged(self) -> nn.Linear:
        lin = nn.Linear(self.base.in_features, self.base.out_features,
                        bias=self.base.bias is not None)
        with torch.no_grad():
            lin.weight.copy_(self.base.weight + self.scaling * (self.lora_B @ self.lora_A))
            if self.base.bias is not None:
                lin.bias.copy_(self.base.bias)
        return lin.to(self.base.weight.device)


def _set_submodule(root: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child, new)


def apply_lora(model: nn.Module, rank: int, alpha: float, targets: str = DEFAULT_TARGETS,
               dropout: float = 0.0) -> list[str]:
    """Wrap every nn.Linear whose qualified name matches `targets`; freeze everything else.

    Returns the wrapped module names. Raises if nothing matched (a silent no-op here would
    train nothing and look like a converged run).
    """
    pat = re.compile(targets)
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and pat.search(n)]
    if not names:
        raise ValueError(f"LoRA targets {targets!r} matched no nn.Linear")
    for p in model.parameters():
        p.requires_grad = False
    for n in names:
        _set_submodule(model, n, LoRALinear(model.get_submodule(n), rank, alpha, dropout))
    return names


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if ".lora_A" in k or ".lora_B" in k}


def lora_param_count(model: nn.Module) -> int:
    return sum(v.numel() for v in lora_state_dict(model).values())


def merge_lora(model: nn.Module) -> nn.Module:
    """Fold every LoRALinear back into a plain nn.Linear (in place); returns `model`."""
    for n, m in list(model.named_modules()):
        if isinstance(m, LoRALinear):
            _set_submodule(model, n, m.merged())
    return model


def load_lora(model: nn.Module, ckpt: dict, merge: bool = True) -> nn.Module:
    """Re-wrap a base model per `ckpt['lora']` and load its adapter; optionally merge."""
    cfg = ckpt["lora"]
    apply_lora(model, cfg["rank"], cfg["alpha"], cfg["targets"], cfg.get("dropout", 0.0))
    missing, unexpected = model.load_state_dict(ckpt["lora_state_dict"], strict=False)
    assert not unexpected, unexpected
    bad = [k for k in missing if ".lora_A" in k or ".lora_B" in k]
    assert not bad, f"adapter keys missing from checkpoint: {bad[:5]}"
    return merge_lora(model) if merge else model
