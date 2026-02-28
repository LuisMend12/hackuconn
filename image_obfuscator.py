#!/usr/bin/env python3
"""
image_obfuscator.py (SAFE)

Now supports invisible watermarking (LSB steganographic marker) so the image
does NOT have a visibly readable watermark, while still embedding a marker.

Modes:
- watermark_mode="visible": draws text watermark (old behavior)
- watermark_mode="invisible": embeds an invisible marker (default)
- watermark_mode="none": no watermark at all

This is SAFE obfuscation (optional blur/pixelation + watermarking).
It is NOT data poisoning.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont, ImageFilter


# =======================
# DEFAULT CONFIG
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

def add_watermark_visible(
    im: Image.Image,
    text: str,
    opacity: int = 70,
    scale: float = 0.06,
    margin: int = 16,
) -> Image.Image:
    """Visible watermark (old behavior)."""
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
# Invisible marker (LSB)
# -----------------------

def _to_bits(data: bytes) -> List[int]:
    bits: List[int] = []
    for b in data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits

def _from_bits(bits: List[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | (bits[i + j] & 1)
        out.append(byte)
    return bytes(out)

def embed_invisible_marker(
    im: Image.Image,
    marker_text: str,
    *,
    channels: str = "rgb",
) -> Image.Image:
    """
    Embed marker_text into the LSBs of pixels (in RGB channels by default).
    Produces an image that looks the same to humans (near-imperceptible change).

    Note: If you save as JPEG afterward, you may lose the marker due to compression.
    Prefer PNG/WebP lossless for reliable extraction.
    """
    if not marker_text:
        return im.convert("RGB")

    im = im.convert("RGB")
    w, h = im.size
    px = im.load()

    payload = marker_text.encode("utf-8")
    length = len(payload)

    # 4-byte length prefix (big endian) + payload
    header = length.to_bytes(4, byteorder="big")
    data = header + payload
    bits = _to_bits(data)

    # Determine which channels to use
    chan_idxs = []
    channels = channels.lower()
    if "r" in channels: chan_idxs.append(0)
    if "g" in channels: chan_idxs.append(1)
    if "b" in channels: chan_idxs.append(2)
    if not chan_idxs:
        chan_idxs = [0, 1, 2]

    capacity = w * h * len(chan_idxs)
    if len(bits) > capacity:
        raise ValueError(
            f"Marker too large for image capacity. Need {len(bits)} bits, have {capacity} bits."
        )

    bit_i = 0
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            rgb = [r, g, b]
            for ci in chan_idxs:
                if bit_i >= len(bits):
                    break
                rgb[ci] = (rgb[ci] & 0xFE) | bits[bit_i]
                bit_i += 1
            px[x, y] = tuple(rgb)
            if bit_i >= len(bits):
                break
        if bit_i >= len(bits):
            break

    return im

def extract_invisible_marker(
    im: Image.Image,
    *,
    channels: str = "rgb",
) -> Optional[str]:
    """
    Extract the embedded marker text (if present) from LSBs.
    Returns None if extraction fails.
    """
    im = im.convert("RGB")
    w, h = im.size
    px = im.load()

    chan_idxs = []
    channels = channels.lower()
    if "r" in channels: chan_idxs.append(0)
    if "g" in channels: chan_idxs.append(1)
    if "b" in channels: chan_idxs.append(2)
    if not chan_idxs:
        chan_idxs = [0, 1, 2]

    bits: List[int] = []
    for y in range(h):
        for x in range(w):
            rgb = px[x, y]
            for ci in chan_idxs:
                bits.append(rgb[ci] & 1)

    # First 32 bits = length
    if len(bits) < 32:
        return None

    length_bytes = _from_bits(bits[:32])
    length = int.from_bytes(length_bytes, byteorder="big", signed=False)

    total_bits = (4 + length) * 8
    if length < 0 or total_bits > len(bits):
        return None

    payload_bits = bits[32:total_bits]
    payload = _from_bits(payload_bits)

    try:
        return payload.decode("utf-8")
    except Exception:
        return None


# -----------------------
# Core API
# -----------------------

def obfuscate_image(
    im: Image.Image,
    *,
    watermark_text: str = "DO NOT TRAIN",
    watermark_mode: str = "invisible",   # <-- default: no visible watermark
    watermark_opacity: int = 70,
    watermark_scale: float = 0.06,
    watermark_margin: int = 16,
    invisible_channels: str = "rgb",
    pixelate: bool = False,
    pixelate_factor: int = 10,
    blur: bool = False,
    blur_radius: float = 1.2,
) -> Image.Image:
    """
    Takes a PIL Image and returns an obfuscated PIL Image.
    Safe transformations only (watermark/pixelate/blur).

    watermark_mode:
      - "none": no watermark
      - "visible": drawn text watermark
      - "invisible": embeds an LSB marker (recommended if you want no visible text)
    """
    im = im.convert("RGB")

    if pixelate:
        im = apply_pixelation(im, pixelate_factor)

    if blur:
        im = im.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))

    mode = (watermark_mode or "invisible").lower().strip()

    if mode == "visible":
        if watermark_text:
            im = add_watermark_visible(
                im,
                watermark_text,
                opacity=max(0, min(255, int(watermark_opacity))),
                scale=float(watermark_scale),
                margin=max(0, int(watermark_margin)),
            )
    elif mode == "invisible":
        if watermark_text:
            im = embed_invisible_marker(im, watermark_text, channels=invisible_channels)
    elif mode == "none":
        pass
    else:
        raise ValueError(f"Unknown watermark_mode: {watermark_mode!r}")

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
            # Default is invisible marker (no visible watermark)
            obfuscate_file(
                p,
                out_path,
                watermark_text="DO NOT TRAIN",
                watermark_mode="invisible",
            )
            ok += 1
        except Exception as e:
            fail += 1
            print(f"FAILED {p}: {e}")

    print(f"Done ✅  OK={ok}  FAILED={fail}")
    print(f"Outputs are in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()