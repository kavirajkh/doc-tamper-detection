"""Demo helper: build side-by-side panels [document | heat-map overlay | ground truth] for a screen share.

    python demo_overlay.py --images data/demo/images --pred results_demo --gt data/demo --out demo_panels

Writes one PNG per document plus contact_sheet.png (all panels stacked). Needs only numpy + Pillow.
Red overlay = where the model thinks the edit is. The right panel is the true tampered region
(black for authentic pages). Verdict and score come from the prediction JSON and are printed on top.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw


def panel(img, heat, mask, title, width=1500):
    h3 = 0.65 * heat[..., None]
    over = img.astype(np.float32) * (1 - h3) + np.array([255, 0, 0], np.float32) * h3
    gt = np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, 2)
    row = Image.fromarray(np.concatenate([img, over.astype(np.uint8), gt], 1))
    row = row.resize((width, int(row.height * width / row.width)))
    out = Image.new("RGB", (width, row.height + 28), "white")
    out.paste(row, (0, 28))
    ImageDraw.Draw(out).text((6, 8), title + "     [document | model heat map | ground truth]", fill="black")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", default="demo_panels")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    labels = {}
    with open(os.path.join(a.gt, "labels.csv")) as fh:
        for line in fh.read().splitlines()[1:]:
            i, lab = line.split(",")[:2]
            labels[i] = lab
    panels = []
    for f in sorted(os.listdir(a.images)):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(f)[0]
        img = np.array(Image.open(os.path.join(a.images, f)).convert("RGB"))
        heat = np.array(Image.open(os.path.join(a.pred, stem + "_heatmap.png")).convert("L"), np.float32) / 255
        mp = os.path.join(a.gt, "masks", stem + ".png")
        mask = np.array(Image.open(mp).convert("L")) > 127 if os.path.exists(mp) else np.zeros(img.shape[:2], bool)
        res = json.load(open(os.path.join(a.pred, stem + ".json")))
        title = f"{stem}  TRUTH: {labels.get(stem, '?')}  |  VERDICT: {res['verdict']}  |  tamper_score {res['tamper_score']:.2f}"
        p = panel(img, heat, mask, title)
        p.save(os.path.join(a.out, stem + "_panel.png"))
        panels.append(p)
        print(title)
    if panels:
        sheet = Image.new("RGB", (panels[0].width, sum(p.height for p in panels)), "white")
        y = 0
        for p in panels:
            sheet.paste(p, (0, y))
            y += p.height
        sheet.save(os.path.join(a.out, "contact_sheet.png"))
        print("wrote", os.path.join(a.out, "contact_sheet.png"))


if __name__ == "__main__":
    main()
