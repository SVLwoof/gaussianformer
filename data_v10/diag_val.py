"""Diagnostic: compute the trainer's val log-L1 for an arbitrary checkpoint, to test whether
V16's phase1_epoch_5 is a good init (~V15's 0.0037) or stuck (~0.012).

  uv run --frozen python -m data_v10.diag_val <ckpt.pt> [n_samples] [resolution]
"""
import sys, random
import torch
from torch.utils.data import DataLoader, Subset

from training.train import training_forward, compute_loss
from training.dataset import GaussianRenderDataset, collate_fn
from gaussianformer.utils.ray_generator import RayGenerator
from gaussianformer.models.gaussianformer import GaussianFormer
from gaussianformer.models.config import GaussianFormerConfig

ckpt = sys.argv[1]
n_max = int(sys.argv[2]) if len(sys.argv) > 2 else 400
res = int(sys.argv[3]) if len(sys.argv) > 3 else 256
device = "cuda"

model = GaussianFormer(GaussianFormerConfig(pe_type="rope"))
sd = torch.load(ckpt, map_location="cpu", weights_only=True)["model_state_dict"]
model.load_state_dict(sd)
model.to(device).eval()
mc = model.config
rg = RayGenerator().to(device)

ds = GaussianRenderDataset("data_v10/h5s_20k_rec_val", "data_v10/renders_val", resolution=res)  # no aug
idx = list(range(len(ds))); random.Random(0).shuffle(idx)
dl = DataLoader(Subset(ds, idx[:n_max]), batch_size=1, shuffle=False, num_workers=8, collate_fn=collate_fn)

tot = 0.0; n = 0
with torch.no_grad():
    for b in dl:
        g = b["gaussians"].to(device); m = b["mask"].to(device)
        c = b["c2w"].to(device); f = b["fov"].to(device); t = b["target"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = training_forward(model, rg, g, m, c, f, res, mc)
            loss, _, _ = compute_loss(pred, t, 1.0, 0.0, device)
        tot += loss.item(); n += 1
print(f"DIAG ckpt={ckpt}  res={res}  n={n}  val_logL1={tot/n:.6f}", flush=True)
