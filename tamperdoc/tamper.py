"""Tamper operations and post-capture degradation.

PRIMARY GOALS
    1. Simulate how a real "original" document was captured (grain + one JPEG save).
    2. Forge it in the two ways the assignment defines, returning a pixel-exact mask.
    3. Degrade the result (rotation, blur, noise, phone photo, print-scan, JPEG) identically for
       ALL classes, authentic included, so degradation can't become a label shortcut.

MAJOR FUNCTIONS
    jpeg()                round-trip an image through JPEG at quality q.
    capture()             texture + one JPEG save = the document "as scanned".
    pick_field()          choose which field to forge, weighted like a real fraudster.
    alter_text()          produce the fraudulent new value (change digits, bump leading digit).
    inpaint_and_rewrite() FORGERY 1: erase a field with cv2.inpaint, write new text.
    crop_and_replace()    FORGERY 2: paste a same-type field from another capture.
    degrade()             post-tamper degradation; returns the JPEG quality to save at.
    phone_photo(), print_scan()  harsher capture conditions used by the private profile.

Every tamper function returns (image, mask, info). The mask is uint8, 255 where
pixels were changed by the forger, 0 elsewhere, and is the same size as the
image.
"""
from __future__ import annotations

import io
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .templates import FAMILIES, Doc, add_paper_texture, font, render

TARGET_WEIGHTS = {"total": 5, "subtotal": 2, "tax": 1, "date": 3, "due_date": 2, "invoice_no": 2,
                  "vendor": 1, "bill_to": 1}


def jpeg(img: np.ndarray, q: int) -> np.ndarray:
    """Save to an in-memory JPEG at quality q and decode it back. This is how a region (or a whole page)
    acquires a specific compression history.
    """
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=int(q))
    return np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def capture(doc: Doc, rng: np.random.Generator, q: int) -> np.ndarray:
    """The 'original' as received from a scanner or camera: texture + one JPEG save."""
    # Order matters: grain first, THEN JPEG, exactly as a camera/scanner would produce it.
    return jpeg(add_paper_texture(doc.image, rng), q)


def pick_field(doc: Doc, rnd: random.Random, digits_only: bool = False):
    """Pick the field a forger would edit. Each candidate is repeated `weight` times, so rnd.choice() is a
    weighted draw (total 5, date 3, ...). digits_only restricts to numeric fields.
    """
    cands = []
    for f in doc.fields:
        base = f.name.split("_")[0] if f.name.startswith(("amount", "unit", "qty")) else f.name
        w = TARGET_WEIGHTS.get(base, 2 if base == "amount" else 0)
        if digits_only and not any(c.isdigit() for c in f.text):
            w = 0
        if w:
            # Repeat the field w times = probability proportional to its weight.
            cands += [f] * w
    return rnd.choice(cands)


def alter_text(text: str, rnd: random.Random, single_digit: bool = False) -> str:
    """Create the fraudulent value.
    Numeric text: change 1-3 random digits (exactly 1 in single_digit mode, the hardest case).
    Non-numeric text (names): swap the last word for a company suffix.
    """
    digits = [i for i, c in enumerate(text) if c.isdigit()]
    if digits:
        s = list(text)
        k = 1 if single_digit else rnd.randint(1, min(3, len(digits)))
        for i in rnd.sample(digits, k):
            s[i] = rnd.choice([d for d in "0123456789" if d != s[i]])
        # the classic fraud: make the leading digit bigger
        if not single_digit and rnd.random() < 0.3:
            s[digits[0]] = str(min(9, int(s[digits[0]]) + rnd.randint(1, 5)))
        return "".join(s)
    words = text.split()
    words[-1] = rnd.choice(["Holdings", "Partners", "Group", "Inc.", "LLC", "Services"])
    return " ".join(words)


def f_regular(f_path: str) -> str:
    """Map a bold/regular font file back to its family's regular file (to exclude the original font
    when picking a *different* one).
    """
    for reg, bold in FAMILIES.values():
        if f_path in (reg, bold):
            return reg
    return f_path


def _box(bbox, pad, shape):
    """Expand a bbox by `pad` pixels on every side, clamped to the image bounds."""
    h, w = shape[:2]
    x0, y0, x1, y1 = bbox
    return max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)


def _anchor_right(f) -> bool:
    """True for right-aligned fields (numbers, dates). The forger re-draws them anchored to the right edge
    so the new text lines up with the old.
    """
    return f.name.startswith(("amount", "unit", "qty")) or f.name in ("subtotal", "tax", "total", "date", "due_date", "invoice_no")


def inpaint_and_rewrite(img: np.ndarray, doc: Doc, rnd: random.Random, nrng: np.random.Generator,
                        single_digit: bool = False, method: str | None = None, foreign_font_p: float = 0.4,
                        families: list[str] | None = None):
    """FORGERY 1 - inpaint_and_rewrite: erase a field, then write a different value in its place.

    Steps: (1) pick a field, (2) cv2.inpaint over its bbox (paper grain is smeared away),
    (3) alter the text, (4) draw it in the same font (60%) or another family (40%),
    (5) mask = erased box UNION new-text box (exactly the pixels the forger changed).
    Returns (image, mask uint8 0/255, info dict describing the edit).

    families: font families the forger may substitute. Keep this to families that also
    appear in genuine documents of the same split; otherwise "unfamiliar font" becomes a
    label shortcut and the model flags every document set in that font."""
    f = pick_field(doc, rnd, digits_only=single_digit)
    x0, y0, x1, y1 = _box(f.bbox, 3, img.shape)
    # Hole to erase = the field's bbox padded by 3 px (so no ink fringe survives).
    hole = np.zeros(img.shape[:2], np.uint8)
    hole[y0:y1, x0:x1] = 255
    # Two classic OpenCV inpainting algorithms, chosen at random so the model can't key on one method's smoothing.
    meth = method or rnd.choice(["telea", "ns"])
    flag = cv2.INPAINT_TELEA if meth == "telea" else cv2.INPAINT_NS
    # CRITICAL: erase the text by filling from the surroundings. The result is unnaturally smooth and
    # noise-free there, which is the key trace the noise branch learns.
    out = cv2.inpaint(img, hole, rnd.randint(3, 6), flag)

    # The fraudulent replacement value.
    new = alter_text(f.text, rnd, single_digit)
    # Font choice: usually the original font (careful forger), sometimes a different family (sloppy forger).
    # Substitutes come only from `families` = fonts that also occur in genuine pages (shortcut #3 fix).
    if rnd.random() < foreign_font_p:
        own = f_regular(f.font_path)
        pool = [fam for fam in (families or list(FAMILIES)) if FAMILIES[fam][0] != own] or list(FAMILIES)
        fam = rnd.choice(pool)
        path = FAMILIES[fam][1 if f.bold else 0]
    else:
        path = f.font_path
    size = f.font_size + rnd.choice([0, 0, 0, -1, 1])
    # Small random colour shift: a hand-matched ink colour is never perfect.
    shift = rnd.randint(-20, 20)
    col = tuple(int(np.clip(c + shift + rnd.randint(-6, 6), 0, 255)) for c in f.color)
    pil = Image.fromarray(out)
    d = ImageDraw.Draw(pil)
    fnt = font(path, size)
    if _anchor_right(f):
        xy, anc = (f.bbox[2] + rnd.randint(-1, 1), f.bbox[1] + rnd.randint(-1, 1)), "rt"
    else:
        xy, anc = (f.bbox[0] + rnd.randint(-1, 1), f.bbox[1] + rnd.randint(-1, 1)), "lt"
    # Write the new text, anchored like the original (right edge for numbers, left edge for names).
    d.text(xy, new, font=fnt, fill=col, anchor=anc)
    tb = d.textbbox(xy, new, font=fnt, anchor=anc)
    out = np.array(pil)

    # Ground-truth mask = erased box UNION the bbox of the newly drawn text.
    mask = hole.copy()
    tx0, ty0, tx1, ty1 = _box(tuple(int(v) for v in tb), 2, img.shape)
    mask[ty0:ty1, tx0:tx1] = 255
    return out, mask, dict(field=f.name, old=f.text, new=new, method=f"inpaint_{meth}", font=path)


def crop_and_replace(img: np.ndarray, doc: Doc, rnd: random.Random, nrng: np.random.Generator, q_orig: int,
                     same_document_p: float = 0.0):
    """FORGERY 2 - crop_and_replace: paste a same-type field cut from another capture over the target.

    Patch source: another invoice rendered and "captured" at a JPEG quality 10-25 away from this
    page's (so its noise and compression history differ) or, with probability same_document_p,
    another field of the SAME page (hardest case: similar noise). A +-2 px source jitter imitates
    a hand-placed paste. mask = the pasted rectangle. Returns (image, mask, info).
    """
    f = pick_field(doc, rnd)
    pad = rnd.randint(4, 9)
    tx0, ty0, tx1, ty1 = f.bbox
    tw, th = tx1 - tx0, ty1 - ty0

    # Where does the pasted patch come from?
    if rnd.random() < same_document_p:
        # paste a value from elsewhere in the same document (e.g. a line amount over the total)
        others = [g for g in doc.fields if g is not f and abs((g.bbox[3] - g.bbox[1]) - th) <= 4]
        g = rnd.choice(others) if others else f
        src_img, src_field, origin = img, g, "same_document"
    else:
        # Other-document case: render a donor invoice (75% same template, 25% a different one).
        donor_tid = doc.template_id if rnd.random() < 0.75 else rnd.choice([t for t in range(8) if t != doc.template_id] or [doc.template_id])
        donor = render(donor_tid, random.Random(rnd.random()))
        # Give the donor a DIFFERENT JPEG history (+-10..25 quality): the trace the DCT branch detects.
        q_d = int(np.clip(q_orig + rnd.choice([-1, 1]) * rnd.randint(10, 25), 55, 98))
        # Capture the donor as a scanner would: grain + its own JPEG save.
        src_img = capture(donor, nrng, q_d)
        # take the same kind of field from the donor: same name, else same type (any line amount),
        # else a total. Never fall back to an arbitrary field (e.g. a vendor banner pasted into a table).
        # Take the SAME KIND of field from the donor (otherwise a vendor banner lands in a table: too easy).
        prefix = f.name.split("_")[0]
        same = ([g for g in donor.fields if g.name == f.name]
                or [g for g in donor.fields if g.name.split("_")[0] == prefix]
                or [g for g in donor.fields if g.name in ("total", "subtotal")])
        src_field = rnd.choice(same)
        origin = f"other_document(t{donor_tid},q{q_d})"

    sx0, sy0, sx1, sy1 = src_field.bbox
    w = max(tw, sx1 - sx0) + 2 * pad
    h = max(th, sy1 - sy0) + 2 * pad
    if _anchor_right(f):
        dx1, dy0 = tx1 + pad, ty0 - pad
        dx0 = dx1 - w
        sx1_, sy0_ = sx1 + pad, sy0 - pad
        sx0_ = sx1_ - w
    else:
        dx0, dy0 = tx0 - pad, ty0 - pad
        sx0_, sy0_ = sx0 - pad, sy0 - pad
    H, W = img.shape[:2]
    dx0, dy0 = max(0, dx0), max(0, dy0)
    sx0_, sy0_ = max(0, sx0_), max(0, sy0_)
    w = min(w, W - dx0, src_img.shape[1] - sx0_)
    h = min(h, H - dy0, src_img.shape[0] - sy0_)
    # small misalignment, as a real paste would have
    # Imperfect alignment, as with a real manual paste.
    jx, jy = rnd.randint(-2, 2), rnd.randint(-2, 2)
    sx0_ = int(np.clip(sx0_ + jx, 0, src_img.shape[1] - w))
    sy0_ = int(np.clip(sy0_ + jy, 0, src_img.shape[0] - h))
    patch = src_img[sy0_:sy0_ + h, sx0_:sx0_ + w].copy()
    # A same-page patch already has the page's noise; re-compress it at another quality so it still
    # carries a distinct compression history.
    if origin == "same_document":
        patch = jpeg(patch, int(np.clip(q_orig + rnd.randint(-20, 10), 50, 98)))
    out = img.copy()
    # CRITICAL: the actual paste. Everything outside this rectangle stays byte-identical.
    out[dy0:dy0 + h, dx0:dx0 + w] = patch
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[dy0:dy0 + h, dx0:dx0 + w] = 255
    return out, mask, dict(field=f.name, old=f.text, new=src_field.text, method="crop_replace", source=origin)


# ---------------------------------------------------------------- degradation
def degrade(img: np.ndarray, mask: np.ndarray, paper, nrng: np.random.Generator, rnd: random.Random,
            profile: str = "public"):
    """Realistic capture-condition degradation, applied identically to EVERY class (authentic too).

    profile "public": optional mild rotation / blur / noise / contrast, then a final JPEG (q 60-96).
    profile "private": 35% phone photo, else 30% print-and-scan, else the public ops; JPEG q 45-92.
    Image and mask are warped together (mask with nearest-neighbour so it stays binary).
    The caller saves ONCE at the returned quality: tamper -> degrade -> single final save, which
    matches a customer re-uploading a file.

    Applied identically to every class, authentic included.

    Returns the degraded image (not yet JPEG-saved), the transformed mask, the list
    of operations, and the JPEG quality the caller must save it at."""
    # Record every operation; it goes to labels.csv so evaluate.py can slice results by capture type / JPEG quality.
    ops = []
    h, w = img.shape[:2]
    out = img.astype(np.float32)

    # Capture condition: phone photo (35%), else print-and-scan (30%), else mild digital degradations.
    # Only the private profile uses the first two.
    if profile == "private" and rnd.random() < 0.35:
        out, mask = phone_photo(out, mask, paper, nrng, rnd)
        ops.append("phone_photo")
    elif profile == "private" and rnd.random() < 0.3:
        out, mask = print_scan(out, mask, paper, nrng, rnd)
        ops.append("print_scan")
    else:
        if rnd.random() < 0.3:
            ang = rnd.uniform(-1.5, 1.5)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
            out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=tuple(float(c) for c in paper))
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
            ops.append(f"rotate{ang:.1f}")
        if rnd.random() < 0.3:
            s = rnd.uniform(0.4, 1.1)
            out = cv2.GaussianBlur(out, (0, 0), s)
            ops.append(f"blur{s:.1f}")
        if rnd.random() < 0.3:
            out = out + nrng.normal(0, rnd.uniform(1, 4), out.shape)
            ops.append("noise")
        if rnd.random() < 0.3:
            a, b = rnd.uniform(0.85, 1.1), rnd.uniform(-15, 15)
            out = out * a + b
            ops.append("contrast")
    out = np.clip(out, 0, 255).astype(np.uint8)
    # Final JPEG quality for the whole page. NOT saved here: the caller saves once, so the document has a
    # single, consistent last compression.
    lo, hi = (60, 96) if profile == "public" else (45, 92)
    q = rnd.randint(lo, hi)
    ops.append(f"jpeg{q}")
    return out, mask, ops, q  # caller saves the image once at quality q


def phone_photo(img, mask, paper, nrng, rnd):
    h, w = img.shape[:2]
    # Phone photo: random perspective warp (corners move up to 4% of the size), vignetting, blur and sensor
    # noise. Erases much of the grain/JPEG evidence.
    d = 0.04
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.float32([[rnd.uniform(-d, d) * w, rnd.uniform(-d, d) * h] for _ in range(4)])
    M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(70.0, 70.0, 75.0))
    mask = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = rnd.uniform(0, w), rnd.uniform(0, h)
    light = 1.0 - rnd.uniform(0.1, 0.3) * np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / np.hypot(w, h)
    img = img * light[..., None] + rnd.uniform(-10, 10)
    img = cv2.GaussianBlur(img, (0, 0), rnd.uniform(0.6, 1.4))
    img = img + nrng.normal(0, rnd.uniform(2, 6), img.shape)
    return img, mask


def print_scan(img, mask, paper, nrng, rnd):
    h, w = img.shape[:2]
    # Print-and-scan: blur + toner-like contrast + dust specks + slight rotation. Destroys most compression
    # traces (pixel-F1 collapses here).
    img = cv2.GaussianBlur(img, (0, 0), rnd.uniform(0.8, 1.5))
    img = (img - 128) * rnd.uniform(1.1, 1.35) + 128 + rnd.uniform(5, 20)  # toner contrast, paper brightening
    dots = (nrng.random((h, w, 1)) < 0.003) * -nrng.uniform(30, 90, (h, w, 1))  # specks
    img = img + dots + nrng.normal(0, 3, img.shape)
    ang = rnd.uniform(-2.0, 2.0)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, rnd.uniform(0.98, 1.0))
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(250.0, 250.0, 250.0))
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return img, mask
