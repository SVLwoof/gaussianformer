"""Build labelled comparison figures for the Sunday progress meeting.

Reads the already-rendered showcase PNGs (data_v10/showcase/, produced by showcase_versions.py)
plus their GT counterparts in data_v10/renders/, and composes:

  fig1_generations.png  -- 4 unseen objects x {GT, V14best, V16+LPIPS, V17}, cropped to the
                           object so the sharpness difference is actually visible at slide size.
  fig2_detail.png       -- 2x zoom on the highest-frequency region of two of those objects.

Crops are computed ONCE from the GT luminance and applied identically to every model, so no
model gets a framing advantage. Per-tile numbers come from the showcase manifests.

  uv run python data_v10/make_meeting_figures.py --out_dir meeting_material
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

SHOWCASE = pathlib.Path("data_v10/showcase")
RENDERS = pathlib.Path("data_v10/renders")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

MODELS = ["V14best", "V16-LPIPS", "V17-ep36"]
COL_TITLES = ["Ground truth", "V14  (May)", "V16 + LPIPS  (Jul 12)", "V17  (Jul 23)"]

# (scene, view, human label, explicit crop box or None for auto). Chosen for high-frequency
# structure + monotone V14->V16->V17 gain. An explicit box is needed where the lit floor is as
# bright as the object, which inflates the auto box and shrinks the subject (row 1).
SUBJECTS = [
    (14870, 0, "Mountain diorama\n(tiny trees)", (147, 158, 337, 348)),
    (4647, 0, "Hourglass\n(slats, metal bands)", None),
    (14623, 3, "Dinosaur\n(dorsal spines, legs)", None),
    (5056, 7, "Wooden tower\n(grain striping)", None),
]
# Detail figure: (scene, view, label, zoom-box as fractions of the FG crop) -- x0,y0,x1,y1.
DETAILS = [
    (14623, 3, "Dorsal spines", (0.30, 0.22, 0.85, 0.62)),
    (4647, 0, "Upper rim + slats", (0.15, 0.08, 0.90, 0.55)),
]


_LPIPS = None


def crop_metrics(gt: Image.Image, mdl: Image.Image) -> dict[str, float]:
    """PSNR/LPIPS on the crop actually displayed, matching the eval convention in
    model_on_v10.py:_fg_crop -- resize the crop to 512 with NEAREST (no smoothing, which would
    flatter a blurry render) and measure there. The showcase manifests hold WHOLE-IMAGE numbers,
    which is the very metric we argue is misleading, so they must not label a cropped figure.
    """
    global _LPIPS
    if _LPIPS is None:
        import lpips as lpips_lib
        _LPIPS = lpips_lib.LPIPS(net="vgg").eval()

    def prep(im: Image.Image) -> torch.Tensor:
        a = np.asarray(im.resize((512, 512), Image.NEAREST), dtype=np.float32) / 255.0
        return torch.from_numpy(a).permute(2, 0, 1)[None]

    g, m = prep(gt), prep(mdl)
    mse = float(((g - m) ** 2).mean())
    with torch.no_grad():
        lp = float(_LPIPS(m * 2 - 1, g * 2 - 1).item())
    return {"psnr": 10.0 * np.log10(1.0 / (mse + 1e-12)), "lpips": lp}


def load_metrics() -> dict[tuple[int, int], dict[str, dict[str, float]]]:
    out: dict[tuple[int, int], dict[str, dict[str, float]]] = {}
    for p in sorted(SHOWCASE.glob("showcase_0*_manifest.json")):
        for scene in json.loads(p.read_text()):
            for view, per in scene["views"].items():
                out[(scene["scene"], int(view))] = per
    return out


def fg_box(gt: Image.Image, pad: float = 0.10) -> tuple[int, int, int, int]:
    """Tight square box around the lit object, from GT luminance. Identical for all models."""
    a = np.asarray(gt.convert("L"), dtype=np.float32) / 255.0
    ys, xs = np.where(a > 0.04)
    if len(xs) == 0:
        return (0, 0, gt.width, gt.height)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * (0.5 + pad)
    half = max(half, 40)
    x0 = int(max(0, cx - half))
    y0 = int(max(0, cy - half))
    x1 = int(min(gt.width, cx + half))
    y1 = int(min(gt.height, cy + half))
    return (x0, y0, x1, y1)


def tiles_for(scene: int, view: int,
              box: tuple[int, int, int, int] | None = None,
              ) -> tuple[list[Image.Image], tuple[int, int, int, int]]:
    gt_p = RENDERS / f"scene_{scene}_view_{view}.png"
    if not gt_p.exists():                                  # renders/ uses 4-wide zero padding
        gt_p = RENDERS / f"scene_{scene:04d}_view_{view}.png"
    gt = Image.open(gt_p).convert("RGB")
    box = box or fg_box(gt)
    imgs = [gt.crop(box)]
    for m in MODELS:
        p = SHOWCASE / f"s{scene:05d}_v{view:02d}_{m}.png"
        imgs.append(Image.open(p).convert("RGB").crop(box))
    return imgs, box


def fig_generations(metrics, out: pathlib.Path, tile: int = 300) -> None:
    pad, top, left, cap = 10, 104, 210, 34
    W = left + 4 * (tile + pad) + pad
    H = top + len(SUBJECTS) * (tile + cap + pad) + pad
    canvas = Image.new("RGB", (W, H), (17, 17, 19))
    d = ImageDraw.Draw(canvas)
    f_h = ImageFont.truetype(FONT_B, 21)
    f_r = ImageFont.truetype(FONT_B, 17)
    f_c = ImageFont.truetype(FONT, 15)
    f_t = ImageFont.truetype(FONT_B, 25)

    d.text((pad + 4, 12), "GaussianFormer: render quality across training generations",
           font=f_t, fill=(245, 245, 245))
    d.text((pad + 4, 44), "unseen objects, novel views   |   metrics computed ON the crop shown "
                          "(object-space, not whole-image)   |   input: pruned+recovered 20k Gaussians",
           font=f_c, fill=(150, 150, 155))

    for c, t in enumerate(COL_TITLES):
        x = left + c * (tile + pad)
        colour = (120, 220, 150) if c == 3 else (225, 225, 225) if c == 0 else (175, 175, 180)
        d.text((x + 4, top - 26), t, font=f_h, fill=colour)

    for r, (scene, view, label, box) in enumerate(SUBJECTS):
        imgs, _ = tiles_for(scene, view, box)
        y = top + r * (tile + cap + pad)
        for i, line in enumerate(label.split("\n")):
            d.text((pad + 4, y + 8 + i * 22), line, font=f_r if i == 0 else f_c,
                   fill=(235, 235, 235) if i == 0 else (150, 150, 155))
        d.text((pad + 4, y + 56), f"scene {scene}, view {view}", font=f_c, fill=(120, 120, 125))

        for c, im in enumerate(imgs):
            x = left + c * (tile + pad)
            canvas.paste(im.resize((tile, tile), Image.LANCZOS), (x, y))
            if c == 3:
                d.rectangle([x - 2, y - 2, x + tile + 1, y + tile + 1],
                            outline=(120, 220, 150), width=2)
            if c == 0:
                txt = "reference"
            else:
                m = crop_metrics(imgs[0], im)
                txt = f"LPIPS {m['lpips']:.4f}   PSNR {m['psnr']:.1f} dB"
            d.text((x + 3, y + tile + 7), txt, font=f_c,
                   fill=(120, 220, 150) if c == 3 else (170, 170, 175))

    canvas.save(out)
    print(f"wrote {out}  ({canvas.width}x{canvas.height})")


def fig_detail(metrics, out: pathlib.Path, tile: int = 340) -> None:
    pad, top, left, cap = 10, 104, 210, 34
    W = left + 4 * (tile + pad) + pad
    H = top + len(DETAILS) * (tile + cap + pad) + pad
    canvas = Image.new("RGB", (W, H), (17, 17, 19))
    d = ImageDraw.Draw(canvas)
    f_h = ImageFont.truetype(FONT_B, 21)
    f_r = ImageFont.truetype(FONT_B, 17)
    f_c = ImageFont.truetype(FONT, 15)
    f_t = ImageFont.truetype(FONT_B, 25)

    d.text((pad + 4, 12), "Where the gain lives: high-frequency structure",
           font=f_t, fill=(245, 245, 245))
    d.text((pad + 4, 44), "same crops, zoomed further   |   LPIPS computed on the zoom shown   |   "
                          "V14 invents smooth blobs; V17 resolves the actual geometry",
           font=f_c, fill=(150, 150, 155))

    for c, t in enumerate(COL_TITLES):
        x = left + c * (tile + pad)
        colour = (120, 220, 150) if c == 3 else (225, 225, 225) if c == 0 else (175, 175, 180)
        d.text((x + 4, top - 26), t, font=f_h, fill=colour)

    for r, (scene, view, label, zb) in enumerate(DETAILS):
        imgs, _ = tiles_for(scene, view)
        y = top + r * (tile + cap + pad)
        d.text((pad + 4, y + 8), label, font=f_r, fill=(235, 235, 235))
        d.text((pad + 4, y + 32), f"scene {scene}, view {view}", font=f_c, fill=(120, 120, 125))
        zooms = []
        for im in imgs:
            w, h = im.size
            zooms.append(im.crop((int(zb[0] * w), int(zb[1] * h), int(zb[2] * w), int(zb[3] * h))))
        for c, z in enumerate(zooms):
            x = left + c * (tile + pad)
            canvas.paste(z.resize((tile, tile), Image.NEAREST), (x, y))
            if c == 3:
                d.rectangle([x - 2, y - 2, x + tile + 1, y + tile + 1],
                            outline=(120, 220, 150), width=2)
            txt = ("reference" if c == 0
                   else f"LPIPS {crop_metrics(zooms[0], z)['lpips']:.4f}")
            d.text((x + 3, y + tile + 7), txt, font=f_c,
                   fill=(120, 220, 150) if c == 3 else (170, 170, 175))

    canvas.save(out)
    print(f"wrote {out}  ({canvas.width}x{canvas.height})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=pathlib.Path, default=pathlib.Path("meeting_material"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()
    fig_generations(metrics, args.out_dir / "fig1_generations.png")
    fig_detail(metrics, args.out_dir / "fig2_detail.png")


if __name__ == "__main__":
    main()
