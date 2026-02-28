#!/usr/bin/env python3
"""
image_obfuscator.py (SAFE)

Supports:
1) Streamlit / Python import usage:
      from image_obfuscator import obfuscate_file
      obfuscate_file("in.png", "out.png", watermark_text="DO NOT TRAIN")

2) Optional directory mode when running directly:
      python3 image_obfuscator.py
   (reads from INPUT_DIR, writes to OUTPUT_DIR)

This is a SAFE obfuscator (watermark / optional blur / optional pixelation).
It is NOT data poisoning.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont, ImageFilter


# =======================
# DEFAULT CONFIG
# (Used for directory mode)
# =======================
INPUT_DIR = Path("/workspaces/hackuconn/images")
OUTPUT_DIR = Path("/workspaces/hackuconn/output")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# -----------------------
# Helpers
# -----------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def iter_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if is_image_file(p)])

def safe_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()

def apply_pixelation(im: Image.Image, factor: int) -> Image.Image:
    factor = max(2, int(factor))
    w, h = im.size
    small = im.resize((max(1, w // factor), max(1, h // factor)), Image.Resampling.NEAREST)
    return small.resize((w, h), Image.Resampling.NEAREST)

def add_watermark(
    im: Image.Image,
    text: str,
    opacity: int = 70,
    scale: float = 0.06,
    margin: int = 16,
) -> Image.Image:
    if not text:
        return im.convert("RGB")

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
    suf = out_path.suffix.lower()
    if suf in {".jpg", ".jpeg"}:
        im.save(out_path, quality=95, optimize=True)
    else:
        im.save(out_path)


# -----------------------
# Core API (Streamlit will use this)
# -----------------------

def obfuscate_image(
    im: Image.Image,
    *,
    watermark_text: str = "DO NOT TRAIN",
    watermark_opacity: int = 70,
    watermark_scale: float = 0.06,
    watermark_margin: int = 16,
    pixelate: bool = False,
    pixelate_factor: int = 10,
    blur: bool = False,
    blur_radius: float = 1.2,
) -> Image.Image:
    """
    Takes a PIL Image and returns an obfuscated PIL Image.
    Safe transformations only (watermark/pixelate/blur).
    """
    im = im.convert("RGB")

    if pixelate:
        im = apply_pixelation(im, pixelate_factor)

    if blur:
        im = im.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))

    if watermark_text:
        im = add_watermark(
            im,
            watermark_text,
            opacity=max(0, min(255, int(watermark_opacity))),
            scale=float(watermark_scale),
            margin=max(0, int(watermark_margin)),
        )

    return im


def obfuscate_file(
    input_path: str | Path,
    output_path: str | Path,
    **kwargs,
) -> None:
    """
    Convenience wrapper: reads an image file and writes an obfuscated output file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    im = Image.open(input_path).convert("RGB")
    out = obfuscate_image(im, **kwargs)
    save_image(out, output_path)


# -----------------------
# Optional: directory mode
# -----------------------

def main():
    ensure_dir(OUTPUT_DIR)
    imgs = iter_images(INPUT_DIR)

    if not imgs:
        print(f"No images found under: {INPUT_DIR.resolve()}")
        return

    print(f"Input:  {INPUT_DIR.resolve()}")
    print(f"Output: {OUTPUT_DIR.resolve()}")
    print(f"Found {len(imgs)} image(s). Processing...")

    ok = 0
    fail = 0

    for p in imgs:
        out_path = OUTPUT_DIR / p.name
        try:
            obfuscate_file(p, out_path, watermark_text="DO NOT TRAIN")
            ok += 1
        except Exception as e:
            fail += 1
            print(f"FAILED {p}: {e}")

    print(f"Done ✅  OK={ok}  FAILED={fail}")
    print(f"Outputs are in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()