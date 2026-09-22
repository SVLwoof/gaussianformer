"""log_cov_input: zero-init identity at load, sign/axis invariance, gradient reaches the new layer.

  ATTN_IMPL=sdpa PYTHONPATH=. uv run --no-sync python tests/test_logcov_cpu.py"""
import roma
import torch
from gaussianformer.models.config import GaussianFormerConfig
from gaussianformer.models.gaussianformer import GaussianFormer, log_covariance
from gaussianformer.utils.ray_generator import RayGenerator
from training.train import training_forward, compute_loss

torch.manual_seed(0)
g = torch.rand(1, 64, 14); g[..., :3] -= 0.5; g[..., 3:6] = torch.rand(1, 64, 3) * 0.03 + 1e-4
g[..., 6:10] = roma.random_unitquat(64)[None][..., [3, 0, 1, 2]]

# q and -q, and a relabelled + flipped axis frame, give the same 6-vector
flip = g.clone(); flip[..., 6:10] *= -1
R = roma.unitquat_to_rotmat(g[0, :, [7, 8, 9, 6]])
P = torch.tensor([[0., 1, 0], [0, 0, 1], [1, 0, 0]])  # R' = R P^T with scales permuted by P
Rp = R @ P.T * torch.tensor([1., -1, -1])  # column flips keep det(Rp) = +1
perm = g.clone(); perm[0, :, 3:6] = (P @ g[0, :, 3:6, None])[..., 0]
perm[0, :, 6:10] = roma.rotmat_to_unitquat(Rp)[:, [3, 0, 1, 2]]
L = log_covariance(g)
assert torch.allclose(L, log_covariance(flip), atol=1e-5)
assert torch.allclose(L, log_covariance(perm), atol=1e-4), (L - log_covariance(perm)).abs().max()
assert torch.isfinite(log_covariance(torch.zeros(1, 4, 14))).all()

# zero-init: identical to the baseline with the same weights
base_c = GaussianFormerConfig(pe_type="rope", num_layers=2, view_transformer_n_layers=4).with_overrides(["proj_rope_2d=true"])
c = base_c.with_overrides(["log_cov_input=true"])
base, m = GaussianFormer(base_c).eval(), GaussianFormer(c).eval()
missing, _ = m.load_state_dict(base.state_dict(), strict=False)
assert all(k.startswith("gaussian_logcov_encoder") for k in missing), missing
mask = torch.ones(1, 64, dtype=torch.bool)
assert torch.equal(base.construct_sequence(g, mask)[0], m.construct_sequence(g, mask)[0])

# one training step reaches the new layer
m.train(); rg = RayGenerator()
c2w = torch.eye(4)[None].clone(); c2w[0, 2, 3] = 1.7
pred = training_forward(m, rg, g, mask, c2w, torch.tensor([45.0]), 64, c)
loss, _, _ = compute_loss(pred, torch.rand(1, 64, 64, 3), 1.0, 0.0, torch.device("cpu"), fg_bg_weight=0.05)
loss.backward()
assert m.gaussian_logcov_encoder.weight.grad.abs().sum() > 0
print("LOGCOV_OK")
