"""Metrics shared by training, calibration and evaluate.py."""
from __future__ import annotations

import numpy as np


def pixel_f1(heat: np.ndarray, mask: np.ndarray, thr: float = 0.5) -> float:
    pred = heat >= thr
    gt = mask.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    if tp + fp + fn == 0:
        return 1.0
    return float(2 * tp / (2 * tp + fp + fn))


def pixel_iou(heat: np.ndarray, mask: np.ndarray, thr: float = 0.5) -> float:
    pred = heat >= thr
    gt = mask.astype(bool)
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else float(np.logical_and(pred, gt).sum() / union)
