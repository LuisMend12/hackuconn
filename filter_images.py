#!/usr/bin/env python3
"""
filter_images.py

Image filter that produces a TXT report AND estimates whether images are AI-generated.

What it does:
- Loads images from a folder (recursive)
- Computes quality metrics (corrupt, size, blur, brightness, aspect, duplicates)
- Adds CLIP-based "AI-generated vs real photo vs screenshot" scoring (probabilistic)
- Outputs a HUMAN-READABLE TXT report (not JSONL)
- Optionally copies images into keep/quarantine/drop folders

Dependencies:
  pip install opencv-python-headless pillow tqdm numpy torch open_clip_torch
"""

from __future__ import annotations

import shutil
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
from PIL import Image, UnidentifiedImageError
import cv2
from tqdm import tqdm

# --- CLIP (AI detection) ---
import torch
import open_clip


# -----------------------------
# Config / thresholds
# -----------------------------

@dataclass
class FilterConfig:
    # Minimum acceptable size
    min_width: int = 256
    min_height: int = 256

    # Blur detection (variance of Laplacian).
    blur_drop_threshold: float = 30.0
    blur_quarantine_threshold: float = 80.0

    # Brightness thresholds (0..255). Using mean luminance.
    dark_drop_threshold: float = 15.0
    dark_quarantine_threshold: float = 30.0

    bright_drop_threshold: float = 245.0
    bright_quarantine_threshold: float = 230.0

    # Aspect ratio bounds (w/h)
    aspect_min: float = 0.25
    aspect_max: float = 4.0

    # Duplicate detection
    dup_hamming_threshold: int = 6

    # If image is corrupt => DROP
    drop_on_corrupt: bool = True

    # AI detection threshold (probability of AI-generated)
    ai_prob_yes_threshold: float = 0.60


# -----------------------------
# Utilities
# -----------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def to_uint64_from_bits(bits: np.ndarray) -> int:
    x = 0
    for bit in bits.astype(np.uint8).tolist():
        x = (x << 1) | int(bit)
    return x

def compute_phash_bgr(img_bgr: np.ndarray) -> int:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    small = np.float32(small)
    dct = cv2.dct(small)
    dct_low = dct[:8, :8].copy()
    vals = dct_low.flatten()
    vals_no_dc = vals[1:]
    med = np.median(vals_no_dc)
    bits = (vals_no_dc > med)
    bits64 = np.concatenate([np.array([0], dtype=bool), bits])
    return to_uint64_from_bits(bits64)

def laplacian_variance(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())

def mean_luminance(gray: np.ndarray) -> float:
    return float(np.mean(gray))

def load_image_bgr(path: Path) -> Tuple[Optional[np.ndarray], Optional[str]]:
    try:
        with Image.open(path) as im:
            im.verify()
    except (UnidentifiedImageError, OSError) as e:
        return None, f"corrupt_or_unreadable: {e}"

    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            rgb = np.array(im)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr, None
    except Exception as e:
        return None, f"decode_failed: {e}"


# -----------------------------
# AI detector (CLIP)
# -----------------------------

class AIDetector:
    """
    CLIP-based probabilistic detector.
    This is NOT perfect; it gives an estimate based on similarity to prompts.
    """
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model = self.model.to(self.device)
        self.model.eval()

        # Prompts: include "screenshot" because many real images in ML pipelines are screenshots
        self.prompts = [
            "an AI generated image",
            "a real photograph",
            "a screenshot of a computer screen",
            "a real camera photo",
            "a synthetic computer generated picture",
        ]
        self.text = self.tokenizer(self.prompts).to(self.device)

        with torch.no_grad():
            self.text_features = self.model.encode_text(self.text)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def score(self, pil_rgb: Image.Image) -> Dict[str, float]:
        with torch.no_grad():
            img = self.preprocess(pil_rgb).unsqueeze(0).to(self.device)
            img_feat = self.model.encode_image(img)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)

            sims = (img_feat @ self.text_features.T).squeeze(0)  # shape: (num_prompts,)
            probs = torch.softmax(sims, dim=-1).detach().cpu().numpy()

        return {p: float(pr) for p, pr in zip(self.prompts, probs)}

    def ai_probability(self, scores: Dict[str, float]) -> float:
        # AI probability: combine the two “synthetic” prompts
        return float(scores.get("an AI generated image", 0.0) + scores.get("a synthetic computer generated picture", 0.0))


# -----------------------------
# Result schema
# -----------------------------

@dataclass
class ImageMetrics:
    width: int
    height: int
    aspect_ratio: float
    blur_var_lap: float
    mean_luma: float
    phash: int

@dataclass
class ImageDecision:
    decision: str
    reasons: List[str]
    risk: float

@dataclass
class AIResult:
    ai_prob: float
    ai_label: str   # YES / NO
    scores: Dict[str, float]

@dataclass
class ImageRecord:
    path: str
    metrics: Optional[ImageMetrics]
    decision: ImageDecision
    ai: Optional[AIResult]
    error: Optional[str] = None


# -----------------------------
# Core filtering logic
# -----------------------------

class ImageFilter:
    def __init__(self, cfg: Optional[FilterConfig] = None, ai_detector: Optional[AIDetector] = None):
        self.cfg = cfg or FilterConfig()
        self._seen: List[Tuple[str, int]] = []
        self.ai = ai_detector  # can be None (disable AI detection)

    def analyze(self, img_path: Path) -> ImageRecord:
        img_bgr, err = load_image_bgr(img_path)
        if img_bgr is None:
            decision = ImageDecision(
                decision="DROP" if self.cfg.drop_on_corrupt else "QUARANTINE",
                reasons=["corrupt_or_unreadable"],
                risk=1.0 if self.cfg.drop_on_corrupt else 0.7,
            )
            return ImageRecord(path=str(img_path), metrics=None, decision=decision, ai=None, error=err)

        h, w = img_bgr.shape[:2]
        aspect = (w / h) if h > 0 else 0.0

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = laplacian_variance(gray)
        luma = mean_luminance(gray)
        phash = compute_phash_bgr(img_bgr)

        metrics = ImageMetrics(
            width=w,
            height=h,
            aspect_ratio=float(aspect),
            blur_var_lap=float(blur),
            mean_luma=float(luma),
            phash=int(phash),
        )

        decision = self._decide(img_path, metrics)

        # AI detection (probabilistic)
        ai_res: Optional[AIResult] = None
        if self.ai is not None:
            # Convert to PIL RGB for CLIP preprocessing
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            scores = self.ai.score(pil)
            ai_prob = self.ai.ai_probability(scores)
            ai_label = "YES" if ai_prob >= self.cfg.ai_prob_yes_threshold else "NO"
            ai_res = AIResult(ai_prob=float(ai_prob), ai_label=ai_label, scores=scores)

        return ImageRecord(path=str(img_path), metrics=metrics, decision=decision, ai=ai_res, error=None)

    def _decide(self, img_path: Path, m: ImageMetrics) -> ImageDecision:
        reasons: List[str] = []
        risk = 0.0

        if m.width < self.cfg.min_width or m.height < self.cfg.min_height:
            reasons.append(f"too_small({m.width}x{m.height})")
            risk = max(risk, 0.75)

        if m.aspect_ratio < self.cfg.aspect_min or m.aspect_ratio > self.cfg.aspect_max:
            reasons.append(f"weird_aspect({m.aspect_ratio:.3f})")
            risk = max(risk, 0.55)

        if m.blur_var_lap < self.cfg.blur_drop_threshold:
            reasons.append(f"very_blurry(varLap={m.blur_var_lap:.1f})")
            risk = max(risk, 0.95)
        elif m.blur_var_lap < self.cfg.blur_quarantine_threshold:
            reasons.append(f"blurry(varLap={m.blur_var_lap:.1f})")
            risk = max(risk, 0.60)

        if m.mean_luma <= self.cfg.dark_drop_threshold:
            reasons.append(f"too_dark(meanLuma={m.mean_luma:.1f})")
            risk = max(risk, 0.95)
        elif m.mean_luma <= self.cfg.dark_quarantine_threshold:
            reasons.append(f"dark(meanLuma={m.mean_luma:.1f})")
            risk = max(risk, 0.60)

        if m.mean_luma >= self.cfg.bright_drop_threshold:
            reasons.append(f"too_bright(meanLuma={m.mean_luma:.1f})")
            risk = max(risk, 0.95)
        elif m.mean_luma >= self.cfg.bright_quarantine_threshold:
            reasons.append(f"bright(meanLuma={m.mean_luma:.1f})")
            risk = max(risk, 0.60)

        dup_of = self._find_duplicate(m.phash)
        if dup_of is not None:
            reasons.append(f"duplicate_of({dup_of})")
            risk = max(risk, 0.85)

        if any(r.startswith("duplicate_of(") for r in reasons):
            decision = "DROP"
        elif risk >= 0.85:
            decision = "DROP"
        elif risk >= 0.35:
            decision = "QUARANTINE"
        else:
            decision = "KEEP"

        self._seen.append((str(img_path), m.phash))
        return ImageDecision(decision=decision, reasons=reasons, risk=float(risk))

    def _find_duplicate(self, phash: int) -> Optional[str]:
        for path, seen_hash in self._seen:
            if hamming_distance(phash, seen_hash) <= self.cfg.dup_hamming_threshold:
                return path
        return None


# -----------------------------
# Traversal + TXT output
# -----------------------------

def iter_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and is_image_file(p)])

def write_txt_report(path: Path, records: List[ImageRecord], counts: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("IMAGE FILTER REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total: {sum(counts.values())}\n")
        f.write(f"KEEP: {counts.get('KEEP', 0)}\n")
        f.write(f"QUARANTINE: {counts.get('QUARANTINE', 0)}\n")
        f.write(f"DROP: {counts.get('DROP', 0)}\n")
        f.write("=" * 60 + "\n\n")

        for rec in records:
            f.write("=" * 60 + "\n")
            f.write(f"Image: {rec.path}\n")

            if rec.error:
                f.write(f"ERROR: {rec.error}\n")
            else:
                m = rec.metrics
                assert m is not None
                f.write("Metrics:\n")
                f.write(f"  Width x Height: {m.width} x {m.height}\n")
                f.write(f"  Aspect Ratio: {m.aspect_ratio:.3f}\n")
                f.write(f"  Blur (Var Laplacian): {m.blur_var_lap:.2f}\n")
                f.write(f"  Mean Luminance: {m.mean_luma:.2f}\n")
                f.write(f"  pHash: {m.phash}\n")

            # --- AI section ---
            if rec.ai is not None:
                f.write("\nAI Detection:\n")
                f.write(f"  AI Generated?: {rec.ai.ai_label}\n")
                f.write(f"  AI Probability: {rec.ai.ai_prob:.3f}\n")
                f.write("  Scores:\n")
                # Print top 3 scores
                top = sorted(rec.ai.scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
                for k, v in top:
                    f.write(f"    - {k}: {v:.3f}\n")

            f.write("\nDecision:\n")
            f.write(f"  Class: {rec.decision.decision}\n")
            f.write(f"  Risk: {rec.decision.risk:.2f}\n")

            if rec.decision.reasons:
                f.write("  Reasons:\n")
                for r in rec.decision.reasons:
                    f.write(f"    - {r}\n")

            f.write("\n")

def maybe_copy(src: Path, dst_dir: Optional[Path]) -> None:
    if dst_dir is None:
        return
    safe_mkdir(dst_dir)
    dst = dst_dir / src.name
    if dst.exists():
        stem, suf = src.stem, src.suffix
        k = 1
        while True:
            cand = dst_dir / f"{stem}__{k}{suf}"
            if not cand.exists():
                dst = cand
                break
            k += 1
    shutil.copy2(src, dst)


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Image filter (TXT report) + CLIP-based AI detection.")
    ap.add_argument("--in_dir", required=True, help="Folder containing images (recursive).")
    ap.add_argument("--out_txt", required=True, help="Output TXT report path.")
    ap.add_argument("--copy_keep_dir", default=None)
    ap.add_argument("--copy_quarantine_dir", default=None)
    ap.add_argument("--copy_drop_dir", default=None)

    # Quality overrides
    ap.add_argument("--min_width", type=int, default=256)
    ap.add_argument("--min_height", type=int, default=256)
    ap.add_argument("--blur_drop", type=float, default=30.0)
    ap.add_argument("--blur_quarantine", type=float, default=80.0)
    ap.add_argument("--dup_hamming", type=int, default=6)

    # AI overrides
    ap.add_argument("--ai_threshold", type=float, default=0.60, help="AI probability threshold for YES/NO (default=0.60).")
    ap.add_argument("--disable_ai", action="store_true", help="Disable AI detection (quality filter only).")

    args = ap.parse_args()

    cfg = FilterConfig(
        min_width=args.min_width,
        min_height=args.min_height,
        blur_drop_threshold=args.blur_drop,
        blur_quarantine_threshold=args.blur_quarantine,
        dup_hamming_threshold=args.dup_hamming,
        ai_prob_yes_threshold=args.ai_threshold,
    )

    ai_detector = None if args.disable_ai else AIDetector()
    filt = ImageFilter(cfg, ai_detector=ai_detector)

    in_dir = Path(args.in_dir)
    out_txt = Path(args.out_txt)

    keep_dir = Path(args.copy_keep_dir) if args.copy_keep_dir else None
    quar_dir = Path(args.copy_quarantine_dir) if args.copy_quarantine_dir else None
    drop_dir = Path(args.copy_drop_dir) if args.copy_drop_dir else None

    images = iter_images(in_dir)
    if not images:
        print(f"No images found under: {in_dir}")
        write_txt_report(out_txt, [], {"KEEP": 0, "QUARANTINE": 0, "DROP": 0})
        return

    records: List[ImageRecord] = []
    counts = {"KEEP": 0, "QUARANTINE": 0, "DROP": 0}

    for p in tqdm(images, desc="Filtering images"):
        rec = filt.analyze(p)
        records.append(rec)
        counts[rec.decision.decision] += 1

        if rec.decision.decision == "KEEP":
            maybe_copy(p, keep_dir)
        elif rec.decision.decision == "QUARANTINE":
            maybe_copy(p, quar_dir)
        else:
            maybe_copy(p, drop_dir)

    write_txt_report(out_txt, records, counts)

    print("Done.")
    print(f"Total: {len(images)} | KEEP: {counts['KEEP']} | QUARANTINE: {counts['QUARANTINE']} | DROP: {counts['DROP']}")
    print(f"Wrote TXT report to: {out_txt}")


if __name__ == "__main__":
    main()