"""Transfer pretrained RenderFormer weights into a fresh GaussianFormer model."""

import torch

from gaussianformer.models.config import GaussianFormerConfig
from gaussianformer.models.gaussianformer import GaussianFormer
from renderformer.models.config import RenderFormerConfig
from renderformer.models.renderformer import RenderFormer


def _renderformer_param_names() -> set[str]:
    """Names of all params/buffers in a fresh RenderFormer (the shared backbone surface)."""
    return set(RenderFormer(RenderFormerConfig()).state_dict().keys())


def gaussian_specific_param_names(model: GaussianFormer) -> set[str]:
    """Trainable parameter names in `model` that are NOT part of the RenderFormer backbone."""
    rf_names = _renderformer_param_names()
    return {name for name, _ in model.named_parameters() if name not in rf_names}


def transfer_weights(
    renderformer_model_id: str,
    gf_config: GaussianFormerConfig | None = None,
) -> GaussianFormer:
    """
    Create a GaussianFormer model initialized with RenderFormer pretrained weights.

    Shared-name parameters (transformer, view_transformer, reg_tokens, DPT decoder) are
    copied directly. Gaussian-specific input-module parameters keep their random init.
    """
    print(f"Loading RenderFormer from: {renderformer_model_id}")
    from huggingface_hub import hf_hub_download
    import safetensors.torch

    weights_path = hf_hub_download(renderformer_model_id, "model.safetensors")
    rf_state = safetensors.torch.load_file(weights_path)

    if gf_config is None:
        gf_config = GaussianFormerConfig()
    gf_model = GaussianFormer(gf_config)
    gf_state = gf_model.state_dict()

    transferred = 0
    skipped_shape = []
    skipped_missing = []

    for name, rf_param in rf_state.items():
        if name in gf_state:
            if gf_state[name].shape == rf_param.shape:
                gf_state[name] = rf_param
                transferred += 1
            else:
                skipped_shape.append(f"  {name}: RF {list(rf_param.shape)} vs GF {list(gf_state[name].shape)}")
        else:
            skipped_missing.append(name)

    gf_model.load_state_dict(gf_state)

    gaussian_only = sorted(gaussian_specific_param_names(gf_model))

    print(f"Transferred {transferred} parameters from RenderFormer")
    print(f"Gaussian-specific (random init): {gaussian_only}")
    if skipped_shape:
        print(f"Skipped (shape mismatch):")
        for s in skipped_shape:
            print(s)
    if skipped_missing:
        print(f"RenderFormer-only (not in GaussianFormer): {len(skipped_missing)}")

    return gf_model


def freeze_backbone(model: GaussianFormer) -> list[str]:
    """Freeze everything except the Gaussian-specific input module."""
    trainable_names = gaussian_specific_param_names(model)
    trainable = []
    for name, param in model.named_parameters():
        if name in trainable_names:
            param.requires_grad = True
            trainable.append(name)
        else:
            param.requires_grad = False
    return trainable


def unfreeze_all(model: GaussianFormer) -> None:
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
