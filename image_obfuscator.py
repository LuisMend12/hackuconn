#!/usr/bin/env python3
"""
image_obfuscator.py (NO-COMMAND / DIRECTORY DEFAULT)

Runs with:
    python3 image_obfuscator.py

It automatically:
- Reads images recursively from INPUT_DIR
- Writes processed images into OUTPUT_DIR

This is a SAFE obfuscator (watermark / optional blur / optional pixelation).
It is NOT data poisoning.

Edit the CONFIG section below to change behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# =======================
# CONFIG (EDIT THESE)
# =======================
INPUT_DIR = Path("/workspaces/hackuconn/images")     # folder to read images from
OUTPUT_DIR = Path("output")    # folder to write outputs to

ADD_WATERMARK = True
WATERMARK_TEXT = "DO NOT TRAIN"
WATERMARK_OPACITY = 70         # 0..255 (higher = more visible)
WATERMARK_SCALE = 0.06         # relative to image width
WATERMARK_MARGIN = 16          # px

PIXELATE = False               # set True to enable
PIXELATE_FACTOR = 10           # higher = stronger pixelation

BLUR = False                   # set True to enable
BLUR_RADIUS = 1.2              # light blur

STRIP_METADATA = True          # saves without EXIF
# =======================

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def iter_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if is_image_file(p)])


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def apply_pixelation(im: Image.Image, factor: int) -> Image.Image:
    factor = max(2, int(factor))
    w, h = im.size
    small = im.resize((max(1, w // factor), max(1, h // factor)), Image.Resampling.NEAREST)
    return small.resize((w, h), Image.Resampling.NEAREST)


def add_watermark(im: Image.Image, text: str, opacity: int, scale: float, margin: int) -> Image.Image:
    if not text:
        return im

    im = im.convert("RGBA")
    w, h = im.size

    font_size = max(12, int(w * max(0.02, float(scale))))
    font = safe_font(font_size)

    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    x = max(margin, w - tw - margin)
    y = max(margin, h - th - margin)

    pad = max(6, font_size // 6)
    box = (x - pad, y - pad, x + tw + pad, y + th + pad)
    draw.rectangle(box, fill=(0, 0, 0, min(180, int(opacity) + 80)))
    draw.text((x, y), text, font=font, fill=(255, 255, 255, int(opacity)))

    out = Image.alpha_composite(im, overlay).convert("RGB")
    return out


def save_image(im: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Saving without passing exif strips metadata by default.
    # For JPEGs we add quality/optimize.
    suf = out_path.suffix.lower()
    if suf in {".jpg", ".jpeg"}:
        im.save(out_path, quality=95, optimize=True)
    else:
        im.save(out_path)


def process_one(in_path: Path, out_path: Path) -> Optional[str]:
    try:
        im = Image.open(in_path).convert("RGB")
    except Exception as e:
        return f"FAILED open: {in_path} ({e})"

    if PIXELATE:
        im = apply_pixelation(im, PIXELATE_FACTOR)

    if BLUR:
        im = im.filter(ImageFilter.GaussianBlur(radius=float(BLUR_RADIUS)))

    if ADD_WATERMARK:
        im = add_watermark(im, WATERMARK_TEXT, WATERMARK_OPACITY, WATERMARK_SCALE, WATERMARK_MARGIN)

    save_image(im, out_path)
    return None


def main():
    ensure_dir(OUTPUT_DIR)

    imgs = iter_images(INPUT_DIR)
    if not imgs:
        print(f"No images found under: {INPUT_DIR.resolve()}")
        print("Create an 'images/' folder and put images inside it.")
        return

    print(f"Input:  {INPUT_DIR.resolve()}")
    print(f"Output: {OUTPUT_DIR.resolve()}")
    print(f"Found {len(imgs)} image(s). Processing...")

    ok = 0
    fail = 0

    for p in imgs:
        # Flat output: keep filename only (simple for demos)
        out_path = OUTPUT_DIR / p.name
        err = process_one(p, out_path)
        if err:
            fail += 1
            print(err)
        else:
            ok += 1

    print(f"Done ✅  OK={ok}  FAILED={fail}")
    print(f"Outputs are in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()