"""Make report figures: the best and worst examples, with heat maps and ground truth.

    python -m tamperdoc.figures --pred results_val --gt data/val --out figures
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
from PIL import Image, ImageDraw

from .metrics import pixel_f1


def panel(img, heat, mask, title, width=900):
    ys, xs = np.nonzero((heat > 0.5) | mask)
    if ys.size:
        y0, y1 = max(0, ys.min() - 60), min(img.shape[0], ys.max() + 60)
        x0, x1 = max(0, xs.min() - 120), min(img.shape[1], xs.max() + 120)
    else:
        y0, y1, x0, x1 = 0, img.shape[0], 0, img.shape[1]
    crop = img[y0:y1, x0:x1]
    over = crop.astype(np.float32).copy()
    hh = heat[y0:y1, x0:x1]
    over = over * (1 - 0.6 * hh[..., None]) + np.array([255, 0, 0]) * 0.6 * hh[..., None]
    gt = np.stack([mask[y0:y1, x0:x1] * 255] * 3, -1).astype(np.uint8)
    row = np.concatenate([crop, over.astype(np.uint8), gt], 1)
    im = Image.fromarray(row)
    scale = width / im.width
    im = im.resize((width, max(1, int(im.height * scale))))
    out = Image.new("RGB", (width, im.height + 22), "white")
    out.paste(im, (0, 22))
    ImageDraw.Draw(out).text((4, 4), title + "   [document | heat map overlay | ground truth]", fill="black")
    return out


def stack(panels, path):
    w = max(p.width for p in panels)
    h = sum(p.height for p in panels) + 8 * len(panels)
    s = Image.new("RGB", (w, h), (200, 200, 200))
    y = 0
    for p in panels:
        s.paste(p, (0, y))
        y += p.height + 8
    s.save(path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", default="figures")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(a.gt, "labels.csv"))))
    recs = []
    for r in rows:
        pred = json.load(open(os.path.join(a.pred, r["id"] + ".json")))
        heat = np.asarray(Image.open(os.path.join(a.pred, r["id"] + "_heatmap.png")), np.float32) / 255
        mask = np.asarray(Image.open(os.path.join(a.gt, "masks", r["id"] + ".png"))) > 127
        f1 = pixel_f1(heat, mask) if mask.any() else None
        recs.append((r, pred, f1))
    tamp = [x for x in recs if x[2] is not None]
    good = sorted(tamp, key=lambda x: -x[2])[:3]
    bad_t = [x for x in sorted(tamp, key=lambda x: x[2]) if x[1]["verdict"] in ("authentic", "inconclusive")][:2]
    bad_a = [x for x in recs if x[2] is None and x[1]["verdict"] in ("inpaint_and_rewrite", "crop_and_replace")][:1]
    bad = (bad_t + bad_a + sorted(tamp, key=lambda x: x[2]))[:3]

    def load_img(r):
        return np.asarray(Image.open(os.path.join(a.gt, "images", r["id"] + ".jpg")).convert("RGB"))

    def mk(x):
        r, pred, f1 = x
        heat = np.asarray(Image.open(os.path.join(a.pred, r["id"] + "_heatmap.png")), np.float32) / 255
        mask = np.asarray(Image.open(os.path.join(a.gt, "masks", r["id"] + ".png"))) > 127
        t = (f"{r['id']}  truth={r['label']}  verdict={pred['verdict']} ({pred['confidence']:.2f})  "
             f"pixel-F1={'-' if f1 is None else f'{f1:.2f}'}  {r.get('degradations', '')}")
        return panel(load_img(r), heat, mask, t)

    stack([mk(x) for x in good], os.path.join(a.out, "correct.png"))
    stack([mk(x) for x in bad], os.path.join(a.out, "failures.png"))
    print("wrote", os.path.join(a.out, "correct.png"), os.path.join(a.out, "failures.png"))
    print("failures:", [(x[0]["id"], x[0]["label"], x[1]["verdict"], x[0].get("degradations")) for x in bad])


if __name__ == "__main__":
    main()
