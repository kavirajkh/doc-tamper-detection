"""Inference: heat map, independent signals, and the fused verdict.

PRIMARY GOAL
    The single place that turns pixels into a decision. Train-time selection, calibration and the predict.py
    CLI all call into this module, so behaviour can't diverge between them.

MAJOR FUNCTIONS
    load_model(), load_calibration()   read weights/tampernet.pt and weights/calibration.json.
    predict_maps()   pad -> TamperNet -> softmax -> upsample; returns heat = 1 - P(background) and class probs.
    model_score()    image-level evidence = mean of the 2000 hottest heat pixels (~ one edited field).
    ela_score()      independent, non-learned signal: Error Level Analysis on background pixels only.
    fuse()           logistic combination of the signals into p = P(tampered).
    tamper_type()    inpaint vs crop, by which class dominates the hot pixels.
    analyze()        the whole pipeline for one image -> verdict, confidence, tamper_score, signals, reason, heat.
    explain()        human-readable reason string.
"""
from __future__ import annotations

import io
import json
import os
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import pad_to_8
from .model import TamperNet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join(ROOT, "weights", "tampernet.pt")
DEFAULT_CALIB = os.path.join(ROOT, "weights", "calibration.json")


def load_model(path: str = DEFAULT_WEIGHTS) -> TamperNet:
    """Build the architecture and load trained weights (CPU), switch to eval mode (fixes BatchNorm statistics)."""
    m = TamperNet()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m


@torch.no_grad()
def predict_maps(model: TamperNet, img: np.ndarray):
    """Returns (heat HxW in [0,1], class_probs 3xHxW) at the input's full resolution."""
    # CRITICAL: whole-page inference. The network is fully convolutional, so any size works once H, W are multiples of 8.
    padded, (h, w) = pad_to_8(img)
    x = torch.from_numpy(padded).permute(2, 0, 1).float()[None]
    # Logits at 1/2 resolution.
    logits = model(x)
    # Softmax to probabilities, bilinearly upsample to full padded size, then crop the padding off again.
    probs = F.interpolate(torch.softmax(logits, 1), size=padded.shape[:2], mode="bilinear", align_corners=False)
    probs = probs[0, :, :h, :w].numpy()
    # Heat map = probability of ANY tampered class = 1 - P(background).
    return 1.0 - probs[0], probs


def model_score(heat: np.ndarray, k: int = 2000) -> float:
    """Image-level evidence: mean of the k hottest pixels (k=2000 is about one edited field).

    Chosen on the calibration set (templates 0-5) over single-pixel max, blurred max,
    area above threshold and largest-blob statistics; never tuned on held-out templates."""
    # np.partition finds the k largest in O(n) (a full sort is unnecessary). Averaging the top-k is more robust than
    # the single max (noise) and less diluted than the page mean (a small edit).
    flat = heat.ravel()
    k = min(k, flat.size)
    return float(np.partition(flat, -k)[-k:].mean())


def ela_map(img: np.ndarray, quality: int = 90) -> np.ndarray:
    # ELA step 1: re-save at JPEG quality 90.
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=quality)
    re = np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")).astype(np.float32)
    # ELA step 2: per-pixel absolute difference. Regions with a different compression history respond differently.
    return np.abs(img.astype(np.float32) - re).mean(2)


def ela_score(img: np.ndarray) -> float:
    """Error Level Analysis as a weak, model-independent signal.

    Raw ELA lights up every text stroke, so we score only background pixels
    (well away from ink) and compare the most anomalous 16x16 block with the
    document's typical background level."""
    # Raw ELA lights up every text stroke, so score only BACKGROUND: mask out ink (dark pixels dilated by 7x7).
    e = ela_map(img)
    gray = img.mean(2)
    ink = cv2.dilate((gray < 170).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    bg = np.where(ink, np.nan, e)
    h, w = e.shape
    # Average ELA in 16x16 blocks; keep blocks that are mostly background; then ask how far the most anomalous block
    # sits above the page median, measured in robust (MAD) units.
    hb, wb = h // 16, w // 16
    blocks = bg[: hb * 16, : wb * 16].reshape(hb, 16, wb, 16)
    valid = (~np.isnan(blocks)).mean((1, 3)) > 0.5
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # blocks that are all ink give empty slices
        means = np.nanmean(blocks, (1, 3))[valid]
    if means.size < 10:
        return 0.0
    med = np.median(means)
    mad = np.median(np.abs(means - med)) + 1e-3
    # Normalise to 0..1 (the *25 is an empirical scale chosen so typical pages stay low).
    return float(np.clip((np.percentile(means, 99.9) - med) / (mad * 25), 0, 1))


def tamper_type(probs: np.ndarray, heat: np.ndarray) -> str:
    # Which tamper type? Sum P(inpaint) vs P(crop) over the hot pixels (fall back to the hottest 0.1% if none exceed 0.5).
    sel = heat > 0.5
    if not sel.any():
        sel = heat >= np.percentile(heat, 99.9)
    return "inpaint_and_rewrite" if probs[1][sel].sum() >= probs[2][sel].sum() else "crop_and_replace"


def load_calibration(path: str = DEFAULT_CALIB) -> dict:
    with open(path) as fh:
        return json.load(fh)


def fuse(signals: dict, calib: dict) -> float:
    # Logistic regression prediction: sigmoid(bias + sum(weight * signal)). Weights come from calibration.json.
    z = calib["bias"] + sum(calib["coef"][k] * signals[k] for k in calib["features"])
    return float(1 / (1 + np.exp(-z)))


def analyze(img: np.ndarray, model: TamperNet, calib: dict) -> dict:
    # ANALYZE: (1) heat map + probs, (2) two independent signals, (3) fuse to p, (4) apply the decision band.
    heat, probs = predict_maps(model, img)
    signals = {"model": round(model_score(heat), 4), "ela": round(ela_score(img), 4)}
    p = fuse(signals, calib)
    # The decision band from calibration.json: p >= hi -> tampered (type by tamper_type); p <= lo -> authentic;
    # anything between -> inconclusive (a human looks). A false accusation is the costly error, so hi is strict.
    lo, hi = calib["inconclusive_band"]
    if p >= hi:
        verdict = tamper_type(probs, heat)
        conf = p
    elif p <= lo:
        verdict, conf = "authentic", 1 - p
    else:
        verdict, conf = "inconclusive", 1 - abs(p - 0.5) * 2
    reason = explain(verdict, signals, heat, p)
    return dict(verdict=verdict, confidence=round(float(conf), 4), tamper_score=round(p, 4),
                signals=signals, reason=reason, heat=heat)


def explain(verdict: str, s: dict, heat: np.ndarray, p: float) -> str:
    if verdict == "authentic":
        return f"No localized anomaly (model {s['model']:.2f}, ELA {s['ela']:.2f})."
    ys, xs = np.nonzero(heat > 0.5)
    where = (f"around x={int(xs.mean())}, y={int(ys.mean())}" if ys.size else "no clear region")
    if verdict == "inconclusive":
        return f"Weak or conflicting evidence ({where}; tamper score {p:.2f}); send to human review."
    kind = ("smoothed background under re-rendered text" if verdict == "inpaint_and_rewrite"
            else "pasted region with different noise/compression")
    return f"Likely {kind} {where} (model {s['model']:.2f}, ELA {s['ela']:.2f})."
