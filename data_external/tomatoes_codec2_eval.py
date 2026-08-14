"""Codec verdict: model vs rec-GT rasterization on held-out views incl. zoom extrapolation."""
from __future__ import annotations
import json, numpy as np, torch, imageio.v3 as iio
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from render_compare import load_model, ModelSpec, load_gt
from infer_gaussian import load_single_gaussian_h5_data
from data_external.orbit import c2w_to_viewmat
from data_v10.prune_recovery import rasterize
import lpips as lpips_lib

D = Path("data_external/tomatoes"); device = "cuda"; RES, FOV = 512, 45.0
h5 = Path("experiments/overfit/data/tomatoes_codec2/h5s/tomatoes_codec2.h5")
data = load_single_gaussian_h5_data(h5)
import h5py
with h5py.File(D/"h5"/"tomatoes_rec20000.h5", "r") as f:
    rec = {k: torch.as_tensor(np.array(f[k], np.float32), device=device) for k in ("means","scales","rotations","colors","opacities")}
recp = dict(means=rec["means"], quats=rec["rotations"], scales=rec["scales"], colors=rec["colors"], opacities=rec["opacities"].reshape(-1))
pipe = load_model(ModelSpec(ckpt=Path("checkpoints_tomato_codec2/phase2_epoch_240.pt"), label="codec", pe_type="rope"), device)
lp = lpips_lib.LPIPS(net="alex").to(device).eval()
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
focal = 0.5*RES/np.tan(0.5*np.radians(FOV))
K = torch.as_tensor(np.array([[focal,0,RES/2],[0,focal,RES/2],[0,0,1]], np.float32), device=device)

def psnr(a,b): return 10*np.log10(1/max(((a-b)**2).mean(),1e-12))
def band(img, txt, hot=False):
    im = Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8)); d = ImageDraw.Draw(im)
    d.rectangle([0,0,im.width,30], fill=(0,0,0)); d.text((5,4), txt, fill=(120,255,160) if hot else (255,230,0), font=font)
    return np.asarray(im).astype(np.float32)/255

summary, rows = {}, []
for name in ("novel_rand", "novel_close", "novel_far"):
    c2ws = np.load(D/f"codec_eval_{name}_c2w.npy")
    vm = torch.as_tensor(np.stack([c2w_to_viewmat(c) for c in c2ws]), device=device)
    Ks = K[None].expand(len(c2ws), 3, 3)
    stats = []
    for i in range(len(c2ws)):
        gt = load_gt(D/"renders"/f"codec_eval_{name}"/f"gt_view_{i}.png", RES)
        rg = rasterize(recp, vm, Ks, [i], RES)[0].clamp(0,1).cpu().numpy()
        c2w_t = torch.as_tensor(c2ws[i], device=device)[None]
        with torch.no_grad():
            out = pipe(gaussians=data["gaussians"][None].to(device), mask=data["mask"][None].to(device),
                       c2w=c2w_t[None], fov=torch.tensor([[FOV]], device=device), resolution=RES,
                       torch_dtype=torch.bfloat16)
        md = np.clip(out[0,0].cpu().float().numpy(), 0, 1)
        pm, pr = psnr(md, gt), psnr(rg, gt)
        lm = float(lp(torch.from_numpy(md).permute(2,0,1)[None].to(device)*2-1,
                      torch.from_numpy(gt).permute(2,0,1)[None].to(device)*2-1).item())
        stats.append((pm, pr, lm, pm - pr))
        if i < 3:
            rows.append(np.concatenate([band(gt, f"GT {name} v{i}"), band(rg, f"rec-GT {pr:.1f}"),
                                        band(md, f"codec {pm:.1f} ({pm-pr:+.1f})", hot=pm>=pr)], axis=1))
    a = np.array(stats)
    summary[name] = dict(model_psnr=float(a[:,0].mean()), rec_psnr=float(a[:,1].mean()),
                         model_lpips=float(a[:,2].mean()), delta=float(a[:,3].mean()),
                         beats_rec=f"{int((a[:,3]>0).sum())}/{len(a)}")
iio.imwrite(D/"renders"/"codec2_verdict.png", (np.clip(np.concatenate(rows,0),0,1)*255).astype(np.uint8))
(D/"renders"/"codec2_verdict.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
