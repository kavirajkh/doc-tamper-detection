"""Render synthetic invoices with known field positions.

PRIMARY GOAL
    Produce clean, realistic invoice images AND the exact pixel box of every field on them
    (total, date, line amounts, ...). The forger in tamper.py needs those boxes to know *where*
    to edit, and the ground-truth masks are built from them. No box = no pixel-accurate mask.

MAJOR FUNCTIONS / OBJECTS
    TEMPLATES           dict: template id -> layout recipe (font family, header/table style, colours).
    Field, Doc          dataclasses: one text field (text + bbox + font) and one rendered page.
    Painter.text()      the ONLY way text is drawn; it records the bbox of every named field.
    render()            draws a whole invoice for a template id and returns a Doc.
    add_paper_texture() adds scanner/camera-like grain and shading to a clean render.

Each template id fixes a layout style (fonts, header arrangement, table style,
colours). The content (vendor, items, amounts, dates) is random per document,
so two invoices from the same template look alike but carry different values.

Templates 0-7 are "public" (used for training and the candidate examples).
Templates 8-11 are reserved for the grader's private test set and use fonts
and layouts that never appear in 0-7.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts")

W, H = 850, 1100  # US Letter at 100 dpi

FAMILIES = {
    "dejavu_sans": ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
    "dejavu_serif": ("DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf"),
    "dejavu_cond": ("DejaVuSansCondensed.ttf", "DejaVuSansCondensed-Bold.ttf"),
    "dejavu_mono": ("DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf"),
    "lib_sans": ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"),
    "lib_serif": ("LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf"),
    "lib_mono": ("LiberationMono-Regular.ttf", "LiberationMono-Bold.ttf"),
    "free_sans": ("FreeSans.ttf", "FreeSansBold.ttf"),
    "free_serif": ("FreeSerif.ttf", "FreeSerifBold.ttf"),
    "free_mono": ("FreeMono.ttf", "FreeMonoBold.ttf"),
}

# header: "left" (title left, meta right) | "center" (centred title) | "band" (coloured band)
# table: "grid" | "rules" | "zebra" | "plain"
TEMPLATES = {
    0: dict(family="dejavu_sans", header="left", table="grid", accent=(30, 60, 120), size=15),
    1: dict(family="lib_serif", header="center", table="rules", accent=(20, 20, 20), size=16),
    2: dict(family="lib_sans", header="band", table="zebra", accent=(0, 110, 90), size=15),
    3: dict(family="dejavu_cond", header="left", table="plain", accent=(140, 30, 30), size=16),
    4: dict(family="dejavu_mono", header="center", table="grid", accent=(60, 60, 60), size=14),
    5: dict(family="free_sans", header="band", table="rules", accent=(90, 40, 130), size=15),
    6: dict(family="dejavu_serif", header="left", table="zebra", accent=(10, 80, 150), size=15),
    7: dict(family="lib_mono", header="band", table="plain", accent=(120, 80, 0), size=14),
    # private (grader) templates: fonts/layout combos unseen above
    8: dict(family="free_serif", header="center", table="zebra", accent=(150, 20, 60), size=17),
    9: dict(family="free_mono", header="left", table="rules", accent=(30, 30, 90), size=15),
    10: dict(family="lib_sans", header="center", table="plain", accent=(0, 0, 0), size=13, two_col=True),
    11: dict(family="dejavu_serif", header="band", table="grid", accent=(70, 110, 30), size=16, two_col=True),
}
PUBLIC_TEMPLATES = list(range(0, 8))
PRIVATE_TEMPLATES = list(range(8, 12))

VENDORS = ["Northwind Supply Co.", "Blue Harbor Logistics", "Sierra Office Goods", "Granite Peak Labs",
           "Redwood Medical Supply", "Atlas Print & Copy", "Cedar Lane Catering", "Pioneer Tech Services",
           "Summit Electrical", "Lakeside Janitorial", "Orchid Analytics", "Ironclad Hardware",
           "Maple Street Bakery", "Horizon Freight", "Quarry Stone Works", "Bright Path Consulting"]
ITEMS = ["Printer paper (case)", "Toner cartridge", "Consulting hours", "Network switch", "Desk chair",
         "Delivery fee", "Cleaning service", "Software license", "Lab reagents", "Catering - lunch",
         "Cable kit", "Maintenance visit", "Shipping pallets", "Safety gloves (box)", "Monitor 27in",
         "Annual support", "Training session", "Storage boxes", "Label printer", "Coffee supplies"]
STREETS = ["Main St", "Oak Ave", "Pine Rd", "Market St", "Elm Blvd", "Harbor Way", "Lake Dr", "Hill Rd"]
CITIES = ["Sacramento, CA", "Austin, TX", "Denver, CO", "Portland, OR", "Raleigh, NC", "Phoenix, AZ"]


@dataclass
class Field:
    """One text item on the page. `name` is its role ("total", "date", "amount_3", ...); it is used to choose
    forgery targets and training negatives.
    """
    name: str
    text: str
    # Where the text sits on the page: (x0, y0, x1, y1) in pixels, as measured by Pillow at draw time.
    bbox: tuple  # x0, y0, x1, y1 (pixels, inclusive-exclusive)
    font_path: str
    font_size: int
    color: tuple
    bold: bool = False


@dataclass
class Doc:
    """One rendered invoice: clean image, the list of Field boxes, the template that made it, its paper colour."""
    image: np.ndarray  # H x W x 3 uint8, clean render (no texture)
    fields: list = field(default_factory=list)
    template_id: int = 0
    paper: tuple = (255, 255, 255)


def money(v: float) -> str:
    """Format a number as US currency, e.g. 1250 -> '$1,250.00'."""
    return f"${v:,.2f}"


def rand_date(rng: random.Random) -> str:
    """Random date string in one of two formats (MM/DD/YYYY or YYYY-MM-DD)."""
    m, d = rng.randint(1, 12), rng.randint(1, 28)
    y = rng.choice([2024, 2025, 2026])
    return rng.choice([f"{m:02d}/{d:02d}/{y}", f"{y}-{m:02d}-{d:02d}"])


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a bundled TTF at a given size. Fonts ship in fonts/, so rendering is identical on every machine."""
    return ImageFont.truetype(os.path.join(FONT_DIR, path), size)


class Painter:
    """Wraps a Pillow image and records every named text draw as a Field (the 'bounding-box ledger')."""
    def __init__(self, img: Image.Image, fam: str, base: int):
        self.d = ImageDraw.Draw(img)
        self.reg, self.bold = FAMILIES[fam]
        self.base = base
        self.fields: list[Field] = []

    def text(self, xy, s, size=None, bold=False, color=(20, 20, 20), anchor="la", name=None):
        """Draw `s` at `xy`; if `name` is given, also register its bounding box.

        CRITICAL: all text goes through here. Because the bbox is measured with the very same
        font and anchor used to draw, boxes can never drift from what is actually on the page.
        """
        size = size or self.base
        path = self.bold if bold else self.reg
        f = font(path, size)
        self.d.text(xy, s, font=f, fill=color, anchor=anchor)
        # Measure where the text really landed (same font + anchor as the draw above).
        bb = self.d.textbbox(xy, s, font=f, anchor=anchor)
        # Only fields a forger might edit get a name; decorative text (headers, labels) is not recorded.
        if name:
            self.fields.append(Field(name, s, tuple(int(v) for v in bb), path, size, color, bold))
        return bb


def render(template_id: int, rng: random.Random) -> Doc:
    """Render one invoice.

    Flow: paper tint -> header (band/center/left) -> meta block (invoice #, dates, bill-to)
    -> 3-7 line items -> subtotal/tax/total -> footer. Content is random per call (rng); layout
    and fonts are fixed by the template id. Amounts are genuinely computed (qty*unit, sums), so
    the document is internally consistent until a forger changes something.

    Returns a Doc: the clean image (no grain yet, not JPEG-saved) plus every named Field box.
    """
    t = TEMPLATES[template_id]
    paper = tuple(int(c) for c in rng.choice([(255, 255, 255), (252, 250, 244), (248, 248, 246), (255, 253, 248)]))
    img = Image.new("RGB", (W, H), paper)
    p = Painter(img, t["family"], t["size"])
    d = p.d
    acc = t["accent"]
    ink = (20, 20, 20)
    s = t["size"]

    vendor = rng.choice(VENDORS)
    inv_no = f"INV-{rng.randint(10000, 99999)}"
    addr = f"{rng.randint(10, 9999)} {rng.choice(STREETS)}, {rng.choice(CITIES)}"
    bill_to = rng.choice([v for v in VENDORS if v != vendor])

    # ---- header
    y = 50
    if t["header"] == "band":
        d.rectangle([0, 0, W, 120], fill=acc)
        p.text((50, 35), vendor, size=s + 12, bold=True, color=(255, 255, 255), name="vendor")
        p.text((50, 75), addr, size=s - 2, color=(235, 235, 235))
        p.text((W - 50, 40), "INVOICE", size=s + 14, bold=True, color=(255, 255, 255), anchor="ra")
        y = 150
    elif t["header"] == "center":
        p.text((W // 2, 40), "INVOICE", size=s + 16, bold=True, color=acc, anchor="ma")
        p.text((W // 2, 90), vendor, size=s + 4, bold=True, color=ink, anchor="ma", name="vendor")
        p.text((W // 2, 118), addr, size=s - 2, color=(80, 80, 80), anchor="ma")
        d.line([50, 148, W - 50, 148], fill=acc, width=2)
        y = 170
    else:  # left
        d.rectangle([50, 40, 110, 100], outline=acc, width=3)  # logo box
        d.ellipse([62, 52, 98, 88], fill=acc)
        p.text((125, 45), vendor, size=s + 8, bold=True, color=acc, name="vendor")
        p.text((125, 82), addr, size=s - 2, color=(80, 80, 80))
        p.text((W - 50, 45), "INVOICE", size=s + 12, bold=True, color=ink, anchor="ra")
        y = 140

    # ---- meta block
    issue, due = rand_date(rng), rand_date(rng)
    lx, vx = (W - 330, W - 50) if not t.get("two_col") else (50, 260)
    meta = [("Invoice #", inv_no, "invoice_no"), ("Date", issue, "date"), ("Due date", due, "due_date")]
    for i, (k, v, nm) in enumerate(meta):
        yy = y + i * (s + 12)
        p.text((lx, yy), k + ":", bold=True, color=ink)
        p.text((vx, yy), v, color=ink, anchor="ra" if not t.get("two_col") else "la", name=nm)
    bx = 50 if not t.get("two_col") else W - 330
    p.text((bx, y), "Bill to:", bold=True, color=ink)
    p.text((bx, y + s + 12), bill_to, color=ink, name="bill_to")
    p.text((bx, y + 2 * (s + 12)), f"Acct {rng.randint(100000, 999999)}", color=(90, 90, 90))
    y += 3 * (s + 12) + 40

    # ---- line items
    n = rng.randint(3, 7)
    cols = [50, 470, 560, 680, W - 50]  # desc | qty | unit | amount (right edge)
    row_h = s + 16
    head_y = y
    if t["table"] in ("grid", "zebra"):
        d.rectangle([cols[0], y, cols[-1], y + row_h], fill=acc)
        hc = (255, 255, 255)
    else:
        hc = acc
    for txt, x, anc in [("Description", cols[0] + 8, "la"), ("Qty", cols[2] - 8, "ra"),
                        ("Unit price", cols[3] - 8, "ra"), ("Amount", cols[4] - 8, "ra")]:
        p.text((x, y + 7), txt, bold=True, color=hc, anchor=anc)
    if t["table"] == "rules":
        d.line([cols[0], y + row_h, cols[-1], y + row_h], fill=acc, width=2)
    y += row_h
    subtotal = 0.0
    for i in range(n):
        desc = rng.choice(ITEMS)
        qty = rng.randint(1, 20)
        unit = round(rng.uniform(5, 900), 2)
        # Line amount = qty x unit price, so totals add up on authentic pages.
        amt = round(qty * unit, 2)
        subtotal += amt
        if t["table"] == "zebra" and i % 2 == 0:
            d.rectangle([cols[0], y, cols[-1], y + row_h], fill=(235, 238, 240))
        p.text((cols[0] + 8, y + 7), desc, color=ink)
        p.text((cols[2] - 8, y + 7), str(qty), color=ink, anchor="ra", name=f"qty_{i}")
        p.text((cols[3] - 8, y + 7), money(unit), color=ink, anchor="ra", name=f"unit_{i}")
        p.text((cols[4] - 8, y + 7), money(amt), color=ink, anchor="ra", name=f"amount_{i}")
        if t["table"] == "grid":
            d.rectangle([cols[0], y, cols[-1], y + row_h], outline=(150, 150, 150))
        elif t["table"] == "rules":
            d.line([cols[0], y + row_h, cols[-1], y + row_h], fill=(200, 200, 200))
        y += row_h
    if t["table"] == "grid":
        for x in cols[1:-1]:
            d.line([x if x != cols[1] else cols[2] - 60, head_y, x if x != cols[1] else cols[2] - 60, y], fill=(150, 150, 150))

    # ---- totals
    # Totals block: subtotal, tax (random rate) and total. 'total' is the most commonly forged field.
    tax = round(subtotal * rng.choice([0.0, 0.0725, 0.0825, 0.06]), 2)
    total = subtotal + tax
    y += 25
    for k, v, nm, b in [("Subtotal", subtotal, "subtotal", False), ("Tax", tax, "tax", False),
                        ("Total due", total, "total", True)]:
        p.text((cols[3] - 8, y), k + ":", bold=True, color=ink, anchor="ra")
        p.text((cols[4] - 8, y), money(v), bold=b, color=ink, anchor="ra", name=nm, size=s + (3 if b else 0))
        y += s + 14

    # ---- footer
    y = max(y + 40, H - 160)
    d.line([50, y, W - 50, y], fill=(190, 190, 190))
    p.text((50, y + 15), "Payment terms: Net 30. Please include the invoice number with your payment.",
           size=s - 3, color=(90, 90, 90))
    p.text((50, y + 15 + s + 4), f"Questions? billing@{vendor.split()[0].lower()}.example", size=s - 3,
           color=(90, 90, 90))
    # Hand back the pixels plus the ledger of field boxes recorded by Painter.text().
    return Doc(np.array(img), p.fields, template_id, paper)


def add_paper_texture(img: np.ndarray, rng: np.random.Generator, strength: float | None = None) -> np.ndarray:
    """Paper grain + low-frequency shading, as a scanner or camera would capture."""
    h, w, _ = img.shape
    strength = strength if strength is not None else rng.uniform(2.0, 6.0)
    # Per-pixel Gaussian grain (same value on all 3 channels): the sensor/paper noise the noise branch later learns from.
    grain = rng.normal(0, strength, (h, w, 1))
    # Low-frequency shading: a coarse random grid (one value per ~50 px) enlarged smoothly = uneven lighting / paper tone.
    low = rng.normal(0, 1, (h // 50 + 2, w // 50 + 2)).astype(np.float32)
    import cv2
    low = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)[..., None] * rng.uniform(1.0, 4.0)
    # Add both, then clip back to the valid 0..255 range.
    out = img.astype(np.float32) + grain + low
    return np.clip(out, 0, 255).astype(np.uint8)
