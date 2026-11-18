"""Run the tamper detector on a folder of document images.

PRIMARY GOAL
    The command-line CONTRACT with the grader: images in -> per image a JSON verdict and a same-size
    8-bit heat map out. Contains no detection logic itself; it only does file I/O and timing around
    tamperdoc.infer.analyze().

MAJOR FUNCTIONS
    main()   parse args -> load model + calibration ONCE -> for each image: analyze() -> write PNG + JSON.

    python predict.py --input ./images --output ./results

Writes, for every image <name>.<ext>:
    <name>.json          verdict, confidence, tamper_score, signals, reason
    <name>_heatmap.png   8-bit grayscale, same size as the input, 255 = most likely tampered
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tamperdoc.infer import DEFAULT_CALIB, DEFAULT_WEIGHTS, analyze, load_calibration, load_model  # noqa: E402

EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--calibration", default=DEFAULT_CALIB)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for PyTorch (0 = library default)")
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    # Load once, reuse for every image (model load is slow relative to one prediction).
    model, calib = load_model(a.weights), load_calibration(a.calibration)
    os.makedirs(a.output, exist_ok=True)
    # Sorted, so output order is deterministic. Only known image extensions are processed.
    files = sorted(f for f in os.listdir(a.input) if f.lower().endswith(EXTS))
    times = []
    for f in files:
        t = time.time()
        # Force 3-channel RGB (handles greyscale / RGBA / palette images).
        img = np.array(Image.open(os.path.join(a.input, f)).convert("RGB"))
        # CRITICAL: all detection happens here (see tamperdoc/infer.py).
        res = analyze(img, model, calib)
        # Heat map is written as a PNG, not into the JSON: remove it from the dict.
        heat = res.pop("heat")
        stem = os.path.splitext(f)[0]
        # Float 0..1 -> 8-bit grayscale (255 = most likely tampered). Same width/height as the input by construction.
        Image.fromarray((np.clip(heat, 0, 1) * 255).round().astype(np.uint8)).save(
            os.path.join(a.output, stem + "_heatmap.png"))
        with open(os.path.join(a.output, stem + ".json"), "w") as fh:
            json.dump({"image": f, **res}, fh, indent=2)
        times.append(time.time() - t)
        print(f"{f}: {res['verdict']} ({res['confidence']:.2f})  {times[-1]:.2f}s", flush=True)
    if times:
        print(f"\n{len(times)} images, mean {np.mean(times):.2f}s, max {np.max(times):.2f}s per image")


if __name__ == "__main__":
    main()
