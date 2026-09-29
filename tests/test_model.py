"""CPU tests of the model, the windowed cross-attention and one training step.

  ATTN_IMPL=sdpa uv run python -m tests.test_model
"""
import math

import torch

from gaussianformer.layers.window import build_window
from gaussianformer.models.config import GaussianFormerConfig
from gaussianformer.models.gaussianformer import GaussianFormer
from gaussianformer.pipelines.rendering_pipeline import model_forward
from gaussianformer.utils.checkpoint import load_seed
from gaussianformer.utils.transform import transform_gaussians_to_cam_coord
from training.train import Loss

SMALL = GaussianFormerConfig(num_layers=2, view_transformer_n_layers=4)
FINE = ["proj_rope_2d=true", "patch_size=4", "ray_embed_patch=8"]


def scene(n=200, pad=10):
    torch.manual_seed(0)
    g = torch.rand(1, n, 14)
    g[..., :3] -= 0.5
    g[..., 3:6] = torch.rand(1, n, 3) * 0.05
    g[..., 6], g[..., 7:10] = 1, 0
    mask = torch.ones(1, n, dtype=torch.bool)
    mask[0, n - pad:] = False
    c2w = torch.eye(4)[None, None].clone()
    c2w[0, 0, 2, 3] = 1.7
    return g, mask, c2w, torch.tensor([[45.0]])


def test_train_step():
    for cfg in ([], FINE, FINE + ["xattn_window=4"]):
        model = GaussianFormer(SMALL.with_overrides(cfg)).train()
        g, mask, c2w, fov = scene()
        pred = model_forward(model, g, mask, c2w, fov, 64, tf32_view=True)
        loss, _, _ = Loss(0.0, 0.05)(pred, torch.rand(1, 64, 64, 3))
        loss.backward()
        assert torch.isfinite(loss) and model.gaussian_encoder.weight.grad.abs().sum() > 0, cfg


def test_window_matches_dense_with_huge_margin():
    dense = GaussianFormer(SMALL.with_overrides(FINE)).eval()
    g, mask, c2w, fov = scene()
    with torch.no_grad():
        ref = model_forward(dense, g, mask, c2w, fov, 64)
        for margin, equal in ((1e6, True), (0.5, False)):
            windowed = GaussianFormer(dense.config.with_overrides(["xattn_window=4", f"xattn_margin={margin}"]))
            assert load_seed(windowed.eval(), dense.state_dict()) == []
            diff = (model_forward(windowed, g, mask, c2w, fov, 64) - ref).abs().max().item()
            assert (diff < 1e-4) == equal, f"margin {margin}: max difference {diff}"


def test_window_is_sparse():
    model = GaussianFormer(SMALL.with_overrides(FINE)).eval()
    g, mask, c2w, fov = scene()
    cam = transform_gaussians_to_cam_coord(c2w[:, 0], g)
    fov_rad = torch.tensor([math.radians(45.0)])
    uv, depth, behind, focal = model.view_transformer.project(cam[..., :3], fov_rad, 64, 64)
    radius = 3.0 * cam[..., 3:6].amax(-1) * focal / depth / 4
    window = build_window(uv, radius, mask & ~behind, 0, 16, 16, 4, 0.5)
    assert window["kv_index"].numel() < 16 * int(mask.sum())


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: ok")
