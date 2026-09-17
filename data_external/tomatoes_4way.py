"""4-way tomatoes comparison: GT(full 219k) | rec-GT(recovered 20k) | plain V17 | overfit.
Both models fed the SAME recovered h5. Metrics vs full-GT per view + strip."""
from __future__ import annotations
import json, numpy as np, torch, imageio.v3 as iio
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from render_compare import load_model, ModelSpec, render_view, load_gt
from infer_gaussian import load_single_gaussian_h5_data
import lpips as lpips_lib

D = Path("data_external/tomatoes"); device = "cuda"; RES = 512
h5 = D / "h5" / "tomatoes_rec20000.h5"
data = load_single_gaussian_h5_data(h5)
for k in ("gaussians", "mask", "c2w", "fov"): data[k] = data[k].to(device)
lp = lpips_lib.LPIPS(net="alex").to(device).eval()
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)

models = [("V17", Path("checkpoints_v17_512lp/phase2_epoch_36.pt")),
          ("overfit", Path("checkpoints_nsweep_n1_tomatoes_rec/phase2_epoch_26000.pt"))]
pipes = {n: load_model(ModelSpec(ckpt=c, label=n, pe_type="rope"), device) for n, c in models}

def metr(img, gt):
    p = 10 * np.log10(1 / max(((img - gt) ** 2).mean(), 1e-12))
    l = float(lp(torch.from_numpy(img).permute(2,0,1)[None].to(device)*2-1,
                 torch.from_numpy(gt).permute(2,0,1)[None].to(device)*2-1).item())
    return p, l

def band(img, txt):
    im = Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8))
    d = ImageDraw.Draw(im); d.rectangle([0,0,im.width,30], fill=(0,0,0)); d.text((5,4), txt, fill=(255,230,0), font=font)
    return np.asarray(im).astype(np.float32)/255

rows, table = [], {}
for v in range(14):
    gt = load_gt(D/"renders"/"gsplat_full"/f"view_{v:02d}.png", RES)
    rec = load_gt(D/"renders"/"gsplat_rec20000"/f"view_{v:02d}.png", RES)
    cols = [band(gt, f"GT v{v}"), band(rec, "rec-GT  %.1f dB" % metr(rec, gt)[0])]
    for n in ("V17", "overfit"):
        img = render_view(pipes[n], data, v, RES, None)
        p, l = metr(img, gt)
        table.setdefault(n, []).append((p, l))
        cols.append(band(img, f"{n}  {p:.1f} dB  {l:.4f}"))
    table.setdefault("rec", []).append(metr(rec, gt))
    rows.append(np.concatenate(cols, axis=1))
grid = (np.clip(np.concatenate(rows, axis=0),0,1)*255).astype(np.uint8)
out = D/"renders"/"fourway_rec20000.png"; iio.imwrite(out, grid)
summ = {n: dict(psnr=float(np.mean([x[0] for x in t])), lpips=float(np.mean([x[1] for x in t]))) for n, t in table.items()}
(D/"renders"/"fourway_metrics.json").write_text(json.dumps(summ, indent=1))
print(json.dumps(summ, indent=1)); print(f"wrote {out}")
