"""CPU check of projection-windowed cross-attention.

  ATTN_IMPL=sdpa PYTHONPATH=. uv run --no-sync python tests/test_window_cpu.py

1. window with an image-sized margin == dense attention (every Gaussian reaches every tile);
2. a tight window drops keys (structure is sparse), output stays finite;
3. one optimizer step through the windowed path."""
import math
import torch
from gaussianformer.models.config import GaussianFormerConfig
from gaussianformer.models.gaussianformer import GaussianFormer
from gaussianformer.utils.checkpoint import load_seed
from gaussianformer.utils.ray_generator import RayGenerator
from gaussianformer.utils.transform import transform_gaussians_to_cam_coord
from gaussianformer.layers.window import build_window
from training.train import training_forward, compute_loss

torch.manual_seed(0)
base = ["proj_rope_2d=true", "patch_size=4", "ray_embed_patch=8"]
dense = GaussianFormer(GaussianFormerConfig(pe_type="rope", num_layers=2, view_transformer_n_layers=4).with_overrides(base)).eval()
g = torch.rand(1, 200, 14); g[..., :3] -= 0.5; g[..., 3:6] = torch.rand(1, 200, 3) * 0.05; g[..., 6] = 1; g[..., 7:10] = 0
mask = torch.ones(1, 200, dtype=torch.bool); mask[0, 190:] = False  # padding
c2w = torch.eye(4)[None]; c2w[0, 2, 3] = 1.7
fov = torch.tensor([[math.radians(45.0)]]); rg = RayGenerator()
gv, c2wv = transform_gaussians_to_cam_coord(c2w, g)
ro, rd = rg(c2wv[:, None], fov, 64)
with torch.no_grad():
    ref = dense(g, mask, ro, rd, gv[:, None], fov=fov)
for margin, expect_equal in ((1e6, True), (0.5, False)):
    m = GaussianFormer(dense.config.with_overrides(["xattn_window=4", f"xattn_margin={margin}"])).eval()
    assert load_seed(m, dense.state_dict()) == []
    with torch.no_grad():
        out = m(g, mask, ro, rd, gv[:, None], fov=fov)
    diff = (out - ref).abs().max().item()
    assert torch.isfinite(out).all()
    if expect_equal:
        assert diff < 1e-4, f"windowed (huge margin) != dense: max diff {diff}"
    else:
        assert diff > 1e-4, "tight window should change the function"
    print(f"margin {margin}: max |windowed - dense| = {diff:.2e}")
# structure: with a tight window the key count is a small fraction of dense
vt = m.view_transformer
pos = gv[..., :3]; shape = gv[..., 3:10]
uv, depth, behind, focal = vt.project(pos, fov, 64, 64)
radius = 3.0 * shape[..., :3].amax(-1) * focal / depth / 4
win = build_window(uv, radius, mask & ~behind, 0, 16, 16, 4, 0.5)
dense_pairs = 16 * 190
print(f"tight window: {win['kv_index'].numel()} (tile, key) pairs vs {dense_pairs} dense, max_k {win['max_k']}")
assert win["kv_index"].numel() < dense_pairs
# training step through the windowed path
cfg = GaussianFormerConfig(pe_type="rope", num_layers=2, view_transformer_n_layers=4).with_overrides(base + ["xattn_window=4"])
mt = GaussianFormer(cfg).train()
opt = torch.optim.AdamW(mt.parameters(), lr=1e-4)
pred = training_forward(mt, rg, g, mask, c2w, torch.tensor([45.0]), 64, cfg)
loss, _, _ = compute_loss(pred, torch.rand(1, 64, 64, 3), 1.0, 0.0, torch.device("cpu"), fg_bg_weight=0.05)
loss.backward(); opt.step()
print("train step ok, loss", float(loss))
print("WINDOW_OK")
