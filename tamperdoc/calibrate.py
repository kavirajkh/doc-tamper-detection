"""Fuse the signals and set the decision thresholds on the calibration set.

PRIMARY GOAL
    Turn raw scores into a DECISION POLICY with an explicit cost target: at most max_fpr (2%) of authentic
    documents may be called tampered, and the 'authentic' verdict must still keep >= min_recall (90%) of
    tampered documents above its threshold. Everything between is 'inconclusive'. Output: weights/calibration.json.

MAJOR FUNCTIONS
    main()   runs the model over a calibration set, fits LogisticRegression on [model, ela], computes the two
             thresholds as quantiles of the authentic / tampered score distributions, writes the JSON.

    python -m tamperdoc.calibrate --data data/calib

Fits a logistic regression on [model, ela] -> P(tampered), then sets:
  * upper threshold: the lowest score at which at most 2% of authentic calibration
    documents would be called tampered (a false accusation is the costly error);
  * lower threshold: the highest score at which at least 90% of tampered calibration
    documents still land above it.
Scores between the two are reported as "inconclusive" and go to human review.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from .data import load, read_labels
from .infer import DEFAULT_CALIB, ela_score, load_model, model_score, predict_maps

FEATURES = ["model", "ela"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/calib")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", default=DEFAULT_CALIB)
    ap.add_argument("--max_fpr", type=float, default=0.02)
    ap.add_argument("--min_recall", type=float, default=0.90)
    a = ap.parse_args(argv)
    # Run the trained model over every calibration page and record the two signals (features) + the label.
    model = load_model(a.weights) if a.weights else load_model()
    X, y = [], []
    for r in read_labels(a.data):
        img, _ = load(a.data, r)
        heat, _ = predict_maps(model, img)
        X.append([model_score(heat), ela_score(img)])
        y.append(int(r["label"] != "authentic"))
    X, y = np.array(X), np.array(y)
    # Fit the fusion: logistic regression maps [model, ela] to P(tampered). C=10 = weak regularisation. A signal
    # that doesn't help gets weight ~0 (check the fitted coef in calibration.json).
    lr = LogisticRegression(C=10.0).fit(X, y)
    p = lr.predict_proba(X)[:, 1]
    # Sort the fused scores of authentic and tampered pages separately; thresholds are quantiles of these.
    auth, tamp = np.sort(p[y == 0]), np.sort(p[y == 1])
    # UPPER threshold: smallest score above which at most max_fpr of authentic pages fall (the 98th percentile of
    # authentic scores). Scores >= hi are called tampered.
    hi = float(auth[int(np.ceil((1 - a.max_fpr) * len(auth))) - 1]) + 1e-4
    # LOWER threshold: the score below which only (1 - min_recall) of tampered pages fall. Scores <= lo are called authentic.
    lo = float(tamp[int(np.floor((1 - a.min_recall) * len(tamp)))]) - 1e-4
    # Safety: keep lo <= hi and both inside [0.01, 0.99].
    lo = min(lo, hi)
    hi, lo = min(hi, 0.99), max(lo, 0.01)
    calib = dict(features=FEATURES, coef=dict(zip(FEATURES, lr.coef_[0].round(4).tolist())),
                 bias=round(float(lr.intercept_[0]), 4), inconclusive_band=[round(lo, 4), round(hi, 4)],
                 fitted_on=a.data, n=int(len(y)))
    with open(a.out, "w") as fh:
        json.dump(calib, fh, indent=2)
    called = np.where(p >= hi, 1, np.where(p <= lo, 0, -1))
    print(json.dumps(calib, indent=2))
    print(f"calibration set: authentic called tampered {np.mean(called[y == 0] == 1):.3f}, "
          f"tampered called tampered {np.mean(called[y == 1] == 1):.3f}, inconclusive {np.mean(called == -1):.3f}")
    print("single-signal AUC  model:", round(_auc(X[:, 0], y), 3), " ela:", round(_auc(X[:, 1], y), 3),
          " fused:", round(_auc(p, y), 3))


def _auc(s, y):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s)


if __name__ == "__main__":
    main()
