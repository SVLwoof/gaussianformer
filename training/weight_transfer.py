import safetensors.torch
from huggingface_hub import hf_hub_download

from gaussianformer.models.config import GaussianFormerConfig
from gaussianformer.models.gaussianformer import GaussianFormer

RENDERFORMER = "microsoft/renderformer-v1-base"


def from_renderformer(config: GaussianFormerConfig, model_id: str = RENDERFORMER) -> tuple[GaussianFormer, list[str]]:
    """GaussianFormer with every tensor it shares with RenderFormer (by name and shape) copied from the
    pretrained checkpoint. Returns the model and the names of its own, randomly initialised parameters
    (the Gaussian input head)."""
    rf = safetensors.torch.load_file(hf_hub_download(model_id, "model.safetensors"))
    model = GaussianFormer(config)
    state = model.state_dict()
    state.update({k: v for k, v in rf.items() if k in state and state[k].shape == v.shape})
    model.load_state_dict(state)
    own = [name for name, _ in model.named_parameters() if name not in rf]
    print(f"RenderFormer weights from {model_id}; new parameters: {own}")
    return model, own
