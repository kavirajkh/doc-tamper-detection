"""Dataset loading and crop sampling.

PRIMARY GOAL
    Feed the network a balanced stream of 192x192 training crops with per-pixel class targets, mixing
    positives with three kinds of HARD NEGATIVES so the model learns the edit, not the context.

MAJOR FUNCTIONS / OBJECTS
    read_labels()   read labels.csv of a dataset folder.
    load()          read one page + its mask (all-zero mask for authentic pages).
    CropSampler     infinite IterableDataset yielding (crop tensor, class-id target map).
    pad_to_8()      edge-pad an image so H, W are multiples of 8 (needed by the DCT branch at inference).

Crops start on multiples of 8 so the JPEG 8x8 grid stays aligned with the
DCT branch. Cropping at an arbitrary offset would hide exactly the
compression traces the frequency branch is meant to see.
"""
from __future__ import annotations

import csv
import json
import os
import random

import numpy as np
import torch
from PIL import Image

from .model import CLASSES

LABEL_TO_CLASS = {"authentic": 0, "inpaint_and_rewrite": 1, "crop_and_replace": 2}


def read_labels(root: str) -> list[dict]:
    """Rows of <root>/labels.csv as dicts (id, label, template, ...)."""
    with open(os.path.join(root, "labels.csv")) as fh:
        return list(csv.DictReader(fh))


def load(root: str, row: dict):
    """Load one page as RGB uint8 plus its boolean tamper mask (all False when the mask file is absent = authentic)."""
    img = np.array(Image.open(os.path.join(root, "images", row["id"] + ".jpg")).convert("RGB"))
    mpath = os.path.join(root, "masks", row["id"] + ".png")
    mask = np.array(Image.open(mpath)) > 127 if os.path.exists(mpath) else np.zeros(img.shape[:2], bool)
    return img, mask


class CropSampler(torch.utils.data.IterableDataset):
    """Yields (image crop, per-pixel class target).

    Three kinds of crop:
      * positive  - centred on the tampered region;
      * field negative - centred on an untouched field, with field types drawn with the
        SAME weights the forger uses (totals most often, then dates, ...). Without this
        the model learns "the total is tampered" from layout alone;
      * ink negative - centred on any untouched ink (headers, names, descriptions), so large
        bold or coloured text that is rarely forged doesn't become a false alarm;
      * random    - anywhere on the page.
    """

    def __init__(self, root: str, crop: int = 192, p_pos: float = 0.35, p_field_neg: float = 0.30,
                 p_ink_neg: float = 0.25, seed: int = 0):
        self.root, self.crop, self.p_pos, self.p_neg, self.p_ink = root, crop, p_pos, p_field_neg, p_ink_neg
        self.rows = read_labels(root)
        self.seed = seed

    def _fields(self, row):
        path = os.path.join(self.root, "fields", row["id"] + ".json")
        return json.load(open(path)) if os.path.exists(path) else []

    def __iter__(self):
        from .tamper import TARGET_WEIGHTS
        # Per-worker RNG so parallel workers don't yield identical streams.
        rnd = random.Random(self.seed + (torch.utils.data.get_worker_info() or type("w", (), {"id": 0})).id)
        c = self.crop
        # Positives are only drawn from tampered pages.
        tampered = [r for r in self.rows if r["label"] != "authentic"]
        while True:
            # Roll which kind of crop this sample is: pos / field-negative / ink-negative / random.
            u = rnd.random()
            kind = ("pos" if u < self.p_pos else "field" if u < self.p_pos + self.p_neg
                    else "ink" if u < self.p_pos + self.p_neg + self.p_ink else "random")
            # Negatives can come from ANY page (authentic or tampered): a tampered page still has many untouched regions.
            row = rnd.choice(tampered if kind == "pos" else self.rows)
            img, mask = load(self.root, row)
            h, w = mask.shape
            cy = cx = None
            # Choose the crop CENTRE for each kind.
            if kind == "pos" and mask.any():
                # pos: centre on the edit (centroid of the mask).
                ys, xs = np.nonzero(mask)
                cy, cx = int(ys.mean()), int(xs.mean())
            elif kind == "field":
                cands = []
                # field negative: candidate untouched fields, repeated by the forger's own weights
                # (totals most often...). Otherwise layout position alone predicts the label (shortcut #2).
                for f in self._fields(row):
                    base = "amount" if f["name"].startswith("amount") else f["name"]
                    wgt = TARGET_WEIGHTS.get(base, 2 if base == "amount" else 0)
                    x0, y0, x1, y1 = f["bbox"]
                    # Skip fields within 8 px of the edit so a 'negative' never contains tampered pixels.
                    if wgt and not mask[max(0, y0 - 8):y1 + 8, max(0, x0 - 8):x1 + 8].any():
                        cands += [f] * wgt
                if cands:
                    x0, y0, x1, y1 = rnd.choice(cands)["bbox"]
                    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            elif kind == "ink":
                # ink negative: any dark (text) pixel not in the mask, on a 4x subsampled grid for speed. Teaches that
                # headers, names and bold text are not suspicious (shortcut #1).
                ink = (img.mean(2) < 150)[::4, ::4] & ~mask[::4, ::4]
                ys, xs = np.nonzero(ink)
                if len(ys):
                    k = rnd.randrange(len(ys))
                    cy, cx = int(ys[k]) * 4, int(xs[k]) * 4
            if cy is None:
                cy, cx = rnd.randint(c // 2, h - c // 2), rnd.randint(c // 2, w - c // 2)
            else:
                # Jitter the centre so the edit is not always in the middle of the crop.
                cy += rnd.randint(-c // 3, c // 3)
                cx += rnd.randint(-c // 3, c // 3)
            # CRITICAL: snap the crop's top-left corner DOWN to a multiple of 8. JPEG works on an 8x8 grid; an
            # arbitrary offset would misalign the DCT branch's blocks and hide the compression evidence.
            y0 = int(np.clip(cy - c // 2, 0, h - c)) // 8 * 8
            x0 = int(np.clip(cx - c // 2, 0, w - c)) // 8 * 8
            im = img[y0:y0 + c, x0:x0 + c]
            m = mask[y0:y0 + c, x0:x0 + c]
            # Per-pixel class target: 0 background, 1 inpaint_and_rewrite, 2 crop_and_replace (image-level label painted onto the mask).
            target = np.zeros((c, c), np.int64)
            target[m] = LABEL_TO_CLASS[row["label"]]
            yield torch.from_numpy(im).permute(2, 0, 1).float(), torch.from_numpy(target)


def pad_to_8(img: np.ndarray):
    """Edge-pad so H and W are multiples of 8 (the DCT branch needs whole blocks). Returns the padded image and the
    original (h, w) so the output can be cropped back.
    """
    h, w = img.shape[:2]
    H, W = (h + 7) // 8 * 8, (w + 7) // 8 * 8
    if (H, W) == (h, w):
        return img, (h, w)
    out = np.pad(img, ((0, H - h), (0, W - w), (0, 0)), mode="edge")
    return out, (h, w)


__all__ = ["CLASSES", "CropSampler", "load", "read_labels", "pad_to_8", "LABEL_TO_CLASS"]
