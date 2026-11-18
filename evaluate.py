"""Score predictions against ground truth.

    python evaluate.py --pred ./results --gt ./testset [--json scores.json]

Ground truth folder:
    labels.csv   columns: id, label (authentic | inpaint_and_rewrite | crop_and_replace), optional extra
                 columns such as template or degradations are used for per-slice breakdowns
    masks/<id>.png   255 = tampered pixel (all zero, or missing, for authentic)

Prediction folder (what predict.py writes):
    <id>.json            {"verdict": ..., "confidence": ..., "tamper_score": ...}
    <id>_heatmap.png     grayscale, same size as the image (resized with a warning if not)

Metrics
  Image level
    auc                 ROC AUC of tamper_score, tampered vs authentic
    macro_f1            3-class F1 (authentic / inpaint / crop); "inconclusive" counts as wrong
    detection_recall    share of tampered documents called tampered (either class)
    false_positive_rate share of authentic documents called tampered  <- the costly error
    inconclusive_rate   share of all documents answered "inconclusive"
  Pixel level (fixed threshold 0.5, never tuned per image)
    pixel_f1, pixel_iou mean over tampered documents
    authentic_false_alarm_px  mean share of authentic pixels above 0.5

Only needs numpy and Pillow.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

CLASSES = ["authentic", "inpaint_and_rewrite", "crop_and_replace"]
THR = 0.5


def auc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    if y.min() == y.max():
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # average ranks for ties
        idx = s == v
        ranks[idx] = ranks[idx].mean()
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def macro_f1(true, pred):
    f1s = []
    for c in CLASSES:
        tp = sum(t == c and p == c for t, p in zip(true, pred))
        fp = sum(t != c and p == c for t, p in zip(true, pred))
        fn = sum(t == c and p != c for t, p in zip(true, pred))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def tamper_score(pred: dict) -> float:
    if "tamper_score" in pred:
        return float(pred["tamper_score"])
    v, c = pred.get("verdict"), float(pred.get("confidence", 0.5))
    if v == "authentic":
        return 1 - c
    if v == "inconclusive":
        return 0.5
    return c


def load_heat(path, shape, warn):
    if not os.path.exists(path):
        return np.zeros(shape, np.float32)
    im = Image.open(path).convert("L")
    if im.size != (shape[1], shape[0]):
        warn.append(os.path.basename(path))
        im = im.resize((shape[1], shape[0]), Image.NEAREST)
    return np.asarray(im, np.float32) / 255.0


def score(pred_dir, gt_dir):
    rows = list(csv.DictReader(open(os.path.join(gt_dir, "labels.csv"))))
    missing, resized = [], []
    recs = []
    for r in rows:
        jp = os.path.join(pred_dir, r["id"] + ".json")
        if os.path.exists(jp):
            pred = json.load(open(jp))
        else:
            missing.append(r["id"])
            pred = {"verdict": "authentic", "confidence": 1.0, "tamper_score": 0.0}
        mp = os.path.join(gt_dir, "masks", r["id"] + ".png")
        if os.path.exists(mp):
            gt = np.asarray(Image.open(mp).convert("L")) > 127
        else:
            img = Image.open(os.path.join(gt_dir, "images", r["id"] + ".jpg"))
            gt = np.zeros((img.size[1], img.size[0]), bool)
        heat = load_heat(os.path.join(pred_dir, r["id"] + "_heatmap.png"), gt.shape, resized)
        hb = heat >= THR
        tp, fp, fn = (hb & gt).sum(), (hb & ~gt).sum(), (~hb & gt).sum()
        recs.append(dict(
            row=r, label=r["label"], verdict=pred.get("verdict", "authentic"), score=tamper_score(pred),
            f1=(2 * tp / (2 * tp + fp + fn)) if gt.any() else None,
            iou=(tp / (tp + fp + fn)) if gt.any() else None,
            false_px=float(hb.mean()) if not gt.any() else None))
    return recs, missing, resized


def summarize(recs):
    if not recs:
        return {}
    y = [int(r["label"] != "authentic") for r in recs]
    called = [r["verdict"] in ("inpaint_and_rewrite", "crop_and_replace") for r in recs]
    auth = [c for c, t in zip(called, y) if t == 0]
    tamp = [c for c, t in zip(called, y) if t == 1]
    f1 = [r["f1"] for r in recs if r["f1"] is not None]
    iou = [r["iou"] for r in recs if r["iou"] is not None]
    fpx = [r["false_px"] for r in recs if r["false_px"] is not None]
    nan = float("nan")
    return dict(
        n=len(recs),
        auc=round(auc([r["score"] for r in recs], y), 4) if 0 < sum(y) < len(y) else nan,
        macro_f1=round(macro_f1([r["label"] for r in recs], [r["verdict"] for r in recs]), 4),
        detection_recall=round(float(np.mean(tamp)), 4) if tamp else nan,
        false_positive_rate=round(float(np.mean(auth)), 4) if auth else nan,
        inconclusive_rate=round(float(np.mean([r["verdict"] == "inconclusive" for r in recs])), 4),
        pixel_f1=round(float(np.mean(f1)), 4) if f1 else nan,
        pixel_iou=round(float(np.mean(iou)), 4) if iou else nan,
        authentic_false_alarm_px=round(float(np.mean(fpx)), 5) if fpx else nan,
    )


def slices(recs):
    out = {}
    by_label = defaultdict(list)
    for r in recs:
        by_label[r["label"]].append(r)
    out["by_label"] = {k: summarize(v) for k, v in sorted(by_label.items())}
    if "template" in recs[0]["row"]:
        by_t = defaultdict(list)
        for r in recs:
            by_t[r["row"]["template"]].append(r)
        out["by_template"] = {k: summarize(v) for k, v in sorted(by_t.items(), key=lambda kv: int(kv[0]))}
    if "degradations" in recs[0]["row"]:
        by_c = defaultdict(list)
        for r in recs:
            d = r["row"]["degradations"]
            cap = "phone_photo" if "phone_photo" in d else "print_scan" if "print_scan" in d else "digital"
            by_c[cap].append(r)
            q = int(d.split("jpeg")[-1]) if "jpeg" in d else None
            if q is not None:
                by_c["jpeg<70" if q < 70 else "jpeg70-84" if q < 85 else "jpeg>=85"].append(r)
        out["by_capture_and_quality"] = {k: summarize(v) for k, v in sorted(by_c.items())}
    return out


def table(name, d):
    keys = ["n", "auc", "macro_f1", "detection_recall", "false_positive_rate", "inconclusive_rate", "pixel_f1",
            "pixel_iou"]
    print(f"\n{name}")
    print(f"  {'slice':<22}" + "".join(f"{k[:11]:>12}" for k in keys))
    for s, m in d.items():
        print(f"  {s:<22}" + "".join(f"{m.get(k, float('nan')):>12.3f}" if k != "n" else f"{m['n']:>12d}" for k in keys))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    recs, missing, resized = score(a.pred, a.gt)
    overall = summarize(recs)
    sl = slices(recs)
    print("OVERALL")
    for k, v in overall.items():
        print(f"  {k:<26}{v}")
    for name, d in sl.items():
        table(name, d)
    if missing:
        print(f"\nWARNING: {len(missing)} predictions missing (scored as authentic): {missing[:5]}...")
    if resized:
        print(f"\nWARNING: {len(resized)} heat maps had the wrong size and were resized: {resized[:5]}...")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(dict(overall=overall, **sl, missing=missing, resized=resized), fh, indent=2)


if __name__ == "__main__":
    main()
