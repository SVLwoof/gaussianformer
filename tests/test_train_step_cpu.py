"""CPU smoke test: one optimizer step through training_forward + compute_loss.

  ATTN_IMPL=sdpa PYTHONPATH=. uv run --no-sync python tests/test_train_step_cpu.py

One real optimizer step on CPU through training_forward + compute_loss, P2 + value RoPE, patch 4."""
import torch
from gaussianformer.models.config import GaussianFormerConfig
from gaussianformer.models.gaussianformer import GaussianFormer
from gaussianformer.utils.ray_generator import RayGenerator
from training.train import training_forward, compute_loss
torch.manual_seed(0)
for cfg in (["proj_rope_2d=true", "value_rope_2d=true"], ["proj_rope_2d=true", "patch_size=4", "ray_embed_patch=8"]):
    c = GaussianFormerConfig(pe_type="rope", num_layers=2, view_transformer_n_layers=4).with_overrides(cfg)
    m = GaussianFormer(c).train()
    rg = RayGenerator()
    g = torch.rand(1, 64, 14); g[..., :3] -= 0.5; g[..., 3:6] = 0.02; g[..., 6] = 1; g[..., 7:10] = 0
    c2w = torch.eye(4)[None].clone(); c2w[0, 2, 3] = 1.7
    target = torch.rand(1, 64, 64, 3)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    pred = training_forward(m, rg, g, torch.ones(1, 64, dtype=torch.bool), c2w, torch.tensor([45.0]), 64, c)
    loss, lg, lp = compute_loss(pred, target, 1.0, 0.0, torch.device("cpu"), fg_bg_weight=0.05)
    loss.backward(); opt.step()
    print(cfg, "step ok, loss", float(loss))
print("TRAINSTEP_OK")
