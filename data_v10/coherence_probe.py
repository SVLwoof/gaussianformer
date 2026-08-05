import json, numpy as np
from pathlib import Path
from render_compare import load_gt
from data_v10.spectrum_probe import native_box, radial_cross

SHOW, REN, RES, NB = Path("data_v10/showcase"), Path("data_v10/renders"), 512, 32
hf = {}
for l in Path("data_v10/ceiling/content_stats.jsonl").read_text().splitlines():
    c = json.loads(l)
    if c["split"] in ("train","unseen2x"): hf[(c["scene"],c["view"])] = c["hf_energy"]

res = {}
for model in ("V14best","V16-LPIPS","V17-ep36"):
    acc = {}
    for f in sorted(SHOW.glob(f"s*_v*_{model}.png")):
        s, v = int(f.name[1:6]), int(f.name[8:10])
        g = REN/f"scene_{s:04d}_view_{v}.png"
        if not g.exists(): continue
        gt = load_gt(g, RES); md = load_gt(f, RES)
        y0,y1,x0,x1 = native_box(gt)
        if min(y1-y0, x1-x0) < 64: continue
        fr, gg, mm, gm = radial_cross(gt[y0:y1,x0:x1], md[y0:y1,x0:x1], NB)
        h = hf.get((s,v), np.nan)
        grp = "ALL" if np.isnan(h) else ("hf HIGH" if h > 0.155 else "hf LOW" if h < 0.105 else "hf MID")
        for k in ("ALL", grp):
            acc.setdefault(k, [np.zeros(NB), np.zeros(NB), np.zeros(NB, dtype=complex)])
            acc[k][0]+=gg; acc[k][1]+=mm; acc[k][2]+=gm
    res[model] = (fr, acc)
    print(f"{model} done", flush=True)

fr = res["V17-ep36"][0]
out = []
out.append(f"{'freq':>6} {'px':>5} | " + " | ".join(f"{m:^21}" for m in res))
out.append(f"{'':>6} {'':>5} | " + " | ".join(f"{'MTF':>10}{'coher':>11}" for _ in res))
for i in range(NB):
    if fr[i] < 0.02 or fr[i] > 0.47: continue
    row = f"{fr[i]:6.3f} {1/fr[i]:5.1f} | "
    for m in res:
        gg, mm, gm = res[m][1]["ALL"]
        row += f"{mm[i]/(gg[i]+1e-20):10.3f}{abs(gm[i])/np.sqrt(gg[i]*mm[i]+1e-20):11.3f} | "
    out.append(row)
out.append("")
out.append("=== V17 by detail content ===")
out.append(f"{'freq':>6} {'px':>5} | " + " | ".join(f"{g:^21}" for g in ("hf LOW","hf MID","hf HIGH")))
for i in range(NB):
    if fr[i] < 0.02 or fr[i] > 0.47: continue
    row = f"{fr[i]:6.3f} {1/fr[i]:5.1f} | "
    for g in ("hf LOW","hf MID","hf HIGH"):
        gg, mm, gm = res["V17-ep36"][1][g]
        row += f"{mm[i]/(gg[i]+1e-20):10.3f}{abs(gm[i])/np.sqrt(gg[i]*mm[i]+1e-20):11.3f} | "
    out.append(row)
Path("meeting_material/coherence_probe.txt").write_text("\n".join(out)+"\n")
print("\n".join(out))
