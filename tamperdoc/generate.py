"""Generate a labeled dataset of authentic and tampered invoices.

PRIMARY GOAL
    Turn (seed, template list, count) into a reproducible dataset on disk in exactly the grader's
    ground-truth format, so the SAME evaluate.py scores it with no glue code.

MAJOR FUNCTIONS
    make_one()     build ONE document end to end: render -> capture -> (forge) -> degrade -> mask.
    main()         CLI: loops make_one(); writes images/, masks/, labels.csv (and fields/ if asked).
    parse_range()  "0-5,7" -> [0,1,2,3,4,5,7].

Output layout (same as the assignment's ground-truth format):
    <out>/images/<id>.jpg
    <out>/masks/<id>.png      (255 = tampered pixel; all-zero for authentic)
    <out>/labels.csv          id,label,template,field,method,source,q_orig,degradations,q_final

Example:
    python -m tamperdoc.generate --out data/train --n 1800 --templates 0-5 --seed 1 --save_fields
    python -m tamperdoc.generate --out data/val   --n 300  --templates 6-7 --seed 2
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random

import numpy as np
from PIL import Image

from .tamper import capture, crop_and_replace, degrade, inpaint_and_rewrite
from .templates import TEMPLATES, render

LABELS = ["authentic", "inpaint_and_rewrite", "crop_and_replace"]


def parse_range(s: str) -> list[int]:
    """Parse a template list like '0-5' or '0-3,7' into a list of ints."""
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def make_one(idx: int, templates: list[int], seed: int, profile: str, label: str | None = None):
    """Build one document. EVERYTHING random derives from (seed, idx): the same arguments always recreate
    the same image, so generation is reproducible, parallelisable and resumable.

    Pipeline: render clean invoice -> capture (grain + JPEG q 75-95) -> forge per label -> degrade
    (all classes) -> binarise mask. Returns (image, mask, field boxes, metadata row).
    """
    rnd = random.Random(seed * 1_000_003 + idx)
    nrng = np.random.default_rng(seed * 1_000_003 + idx)
    tid = rnd.choice(templates)
    doc = render(tid, rnd)
    q_orig = rnd.randint(75, 95)
    img = capture(doc, nrng, q_orig)
    # Class cycles with the index -> exactly balanced: 1/3 authentic, 1/3 inpaint, 1/3 crop.
    label = label or LABELS[idx % 3]
    info = dict(field="", method="", source="")
    mask = np.zeros(img.shape[:2], np.uint8)
    private = profile == "private"
    if label == "inpaint_and_rewrite":
        # Substitute fonts may only come from the families used by THIS split's templates (prevents the
        # 'unfamiliar font = tampered' shortcut).
        fams = sorted({TEMPLATES[t]["family"] for t in templates})
        img, mask, info = inpaint_and_rewrite(img, doc, rnd, nrng, single_digit=private and rnd.random() < 0.35,
                                              foreign_font_p=0.25 if private else 0.4, families=fams)
    elif label == "crop_and_replace":
        img, mask, info = crop_and_replace(img, doc, rnd, nrng, q_orig,
                                           same_document_p=0.4 if private else 0.0)
    # CRITICAL: degradation runs for ALL labels, after tampering.
    img, mask, ops, q = degrade(img, mask, doc.paper, nrng, rnd, profile)
    # Re-binarise: rotation/warp interpolation can leave grey values at the mask edge.
    mask = (mask > 127).astype(np.uint8) * 255
    fields = [dict(name=f.name, bbox=list(f.bbox)) for f in doc.fields]
    return img, mask, fields, dict(label=label, template=tid, field=info.get("field", ""), method=info.get("method", ""),
                           source=info.get("source", ""), q_orig=q_orig, degradations="|".join(ops), q_final=q)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--templates", default="0-5")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--profile", choices=["public", "private"], default="public",
                    help="'private' adds phone-photo and print-scan capture, harsher JPEG, single-digit edits "
                         "and same-document pastes (grader test set)")
    ap.add_argument("--prefix", default="doc")
    ap.add_argument("--save_fields", action="store_true",
                    help="also write fields/<id>.json (field boxes, used to sample hard negatives in training)")
    a = ap.parse_args(argv)
    tpls = parse_range(a.templates)
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "masks"), exist_ok=True)
    if a.save_fields:
        os.makedirs(os.path.join(a.out, "fields"), exist_ok=True)
    rows = []
    for i in range(a.n):
        img, mask, fields, meta = make_one(i, tpls, a.seed, a.profile)
        name = f"{a.prefix}_{i:05d}"
        # The single final JPEG save, at the quality chosen by degrade().
        Image.fromarray(img).save(os.path.join(a.out, "images", name + ".jpg"), quality=meta["q_final"])
        Image.fromarray(mask).save(os.path.join(a.out, "masks", name + ".png"))
        if a.save_fields:
            with open(os.path.join(a.out, "fields", name + ".json"), "w") as fh:
                json.dump(fields, fh)
        rows.append(dict(id=name, **meta))
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{a.n}", flush=True)
    with open(os.path.join(a.out, "labels.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {len(rows)} documents to {a.out}")


if __name__ == "__main__":
    main()
