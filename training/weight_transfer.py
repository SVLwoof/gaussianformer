"""Transfer pretrained RenderFormer weights into a fresh GaussianFormer model."""

import torch

from gaussianformer.models.config import GaussianFormerConfig
from gaussianformer.models.gaussianformer import GaussianFormer
from renderformer.models.renderformer import RenderFormer


GAUSSIAN_SPECIFIC_PARAMS = {
    "gaussian_encoder.weight",
    "gaussian_encoder.bias",
    "gaussian_encoder_norm.weight",
    "gaussian_token",
}


def transfer_weights(renderformer_model_id: str) -> GaussianFormer:
    """
    Create a GaussianFormer model initialized with RenderFormer pretrained weights.

    The 277 shared parameters (transformer, view_transformer, reg_tokens, DPT decoder)
    are copied directly. The 4 Gaussian-specific parameters (gaussian_encoder, gaussian_token)
    keep their random initialization.
    """
    print(f"Loading RenderFormer from: {renderformer_model_id}")
    from renderformer.models.config import RenderFormerConfig
    from huggingface_hub import hf_hub_download
    import safetensors.torch

    # Load config and weights manually (from_pretrained doesn't work with positional config arg)
    renderformer = RenderFormer(RenderFormerConfig())
    weights_path = hf_hub_download(renderformer_model_id, "model.safetensors")
    rf_state = safetensors.torch.load_file(weights_path)

    gf_model = GaussianFormer(GaussianFormerConfig())
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

    gaussian_only = sorted(GAUSSIAN_SPECIFIC_PARAMS & set(gf_state.keys()))

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
    trainable = []
    for name, param in model.named_parameters():
        if name in GAUSSIAN_SPECIFIC_PARAMS:
            param.requires_grad = True
            trainable.append(name)
        else:
            param.requires_grad = False
    return trainable


def unfreeze_all(model: GaussianFormer) -> None:
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
