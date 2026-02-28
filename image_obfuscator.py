#!/usr/bin/env python3
"""
AI Image Obfuscator (Nightshade-style)

Perturbs an image so that during AI model training it is associated with a
*destination concept* instead of its true content. Visually the change is
minimal (slight gloss/tint); to the model it acts as a training-time poison.

Usage:
    python image_obfuscator.py --input art.png --output art_obfuscated.png --concept "cat"
    python image_obfuscator.py --input art.png --output art_obfuscated.png --concept "landscape" --strength 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Optional: PyTorch + OpenCLIP for full poison optimization
try:
    import torch
    import open_clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False


# =============================================================================
# Constants
# =============================================================================

# Default CLIP preprocessing size (ViT-B/32)
CLIP_SIZE = 224
# CLIP normalization (unnormalize for saving)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
# Perturbation bound (L_inf) in *normalized* space; small = minimal visible change
DEFAULT_EPS = 0.02
# Optimization steps for poison
DEFAULT_STEPS = 100
# Learning rate for gradient ascent on perturbation
DEFAULT_LR = 0.1


# =============================================================================
# Fallback: subtle visual-only obfuscation (no CLIP)
# =============================================================================

def obfuscate_fallback(image: np.ndarray, concept: str, strength: float = 0.3) -> np.ndarray:
    """
    Add a minimal visible "gloss" or tint when CLIP is not available.
    Does NOT change model associations; use full pipeline for that.
    """
    out = image.astype(np.float64) / 255.0
    # Use concept string as a seed for reproducible tint
    seed = hash(concept) % (2**32)
    rng = np.random.default_rng(seed)
    # Very subtle per-channel tint (simulate "slight gloss")
    tint = (rng.uniform(-1, 1, 3) * strength * 0.02).reshape(1, 1, 3)
    out = out + tint
    # Optional: barely visible high-freq noise (can confuse naive hashing)
    h, w = out.shape[:2]
    noise = rng.normal(0, strength * 0.005, (*out.shape[:2], 1))
    out = out + np.broadcast_to(noise, out.shape)
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)


# =============================================================================
# CLIP-based poison optimization
# =============================================================================

def _get_clip_model(device: str, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


def _load_image_for_clip(path: Path, preprocess, device: str):
    pil = Image.open(path).convert("RGB")
    img_tensor = preprocess(pil).unsqueeze(0).to(device)
    return img_tensor, np.array(pil)


def _unnormalize_clip(t: "torch.Tensor") -> "torch.Tensor":
    """From CLIP normalized (C,H,W) to pixel [0,1] (C,H,W)."""
    mean = torch.tensor(CLIP_MEAN, device=t.device, dtype=t.dtype).view(3, 1, 1)
    std = torch.tensor(CLIP_STD, device=t.device, dtype=t.dtype).view(3, 1, 1)
    return t * std + mean


def _tensor_to_pil(t: "torch.Tensor", is_normalized: bool = True) -> Image.Image:
    """Convert tensor (batch) to PIL. If is_normalized, unnormalize from CLIP space first."""
    t = t.squeeze(0).detach().cpu()
    if is_normalized:
        t = _unnormalize_clip(t)
    t = t.permute(1, 2, 0)
    t = torch.clamp(t, 0, 1)
    arr = (t.numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def optimize_poison(
    image_tensor: "torch.Tensor",
    concept: str,
    model,
    tokenizer,
    device: str,
    eps: float = DEFAULT_EPS,
    steps: int = DEFAULT_STEPS,
    lr: float = DEFAULT_LR,
) -> "torch.Tensor":
    """
    Optimize a small perturbation so that CLIP associates the image with
    the destination concept. Perturbation is clamped to [-eps, eps] for
    minimal visible change.
    """
    image_tensor = image_tensor.clone().detach().requires_grad_(False)
    delta = torch.zeros_like(image_tensor, device=device, requires_grad=True)

    # Destination concept text embedding (fixed)
    text_tokens = tokenizer([concept]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    optimizer = torch.optim.Adam([delta], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        # Perturbed image in same normalized space as CLIP expects
        perturbed = image_tensor + delta
        image_features = model.encode_image(perturbed)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # Maximize similarity with destination concept => minimize negative cosine sim
        loss = -(image_features @ text_features.T).squeeze()
        loss.backward()
        optimizer.step()
        # Project perturbation to L_inf ball
        with torch.no_grad():
            delta.clamp_(-eps, eps)

    perturbed = image_tensor + delta
    return perturbed


def obfuscate_with_clip(
    image_path: Path,
    concept: str,
    output_path: Path,
    eps: float = DEFAULT_EPS,
    steps: int = DEFAULT_STEPS,
    lr: float = DEFAULT_LR,
    strength: float = 1.0,
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
) -> None:
    """
    Load image, run CLIP-based poison optimization, then blend result with
    original at full resolution and save.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, tokenizer = _get_clip_model(device, model_name, pretrained)

    img_tensor, pil_rgb = _load_image_for_clip(image_path, preprocess, device)
    orig_size = pil_rgb.shape[:2][::-1]  # (W, H) for PIL

    # Scale epsilon by strength (optional)
    effective_eps = eps * strength

    poisoned_tensor = optimize_poison(
        img_tensor, concept, model, tokenizer, device,
        eps=effective_eps, steps=steps, lr=lr,
    )
    poisoned_pil_small = _tensor_to_pil(poisoned_tensor, is_normalized=True)

    # Upscale poison to original resolution and blend with original for minimal visual change
    poisoned_pil = poisoned_pil_small.resize(orig_size, Image.Resampling.LANCZOS)
    orig_pil = Image.fromarray(pil_rgb)
    # Blend: more strength => more poison, less strength => closer to original
    out_arr = np.clip(
        (1 - strength) * np.array(orig_pil) + strength * np.array(poisoned_pil),
        0, 255
    ).astype(np.uint8)
    out_pil = Image.fromarray(out_arr)
    out_pil.save(output_path, quality=95)
    print(f"Saved obfuscated image to {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Obfuscate an image so AI training associates it with a destination concept (Nightshade-style)."
    )
    parser.add_argument("--input", "-i", required=True, type=Path, help="Input image path")
    parser.add_argument("--output", "-o", required=True, type=Path, help="Output image path")
    parser.add_argument(
        "--concept", "-c",
        default="cat",
        help="Destination concept (e.g. 'cat', 'landscape'). Model will drift toward this when trained on the image.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Poison strength 0–1. Higher = stronger association with concept, slightly more visible change (default: 1.0).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=DEFAULT_EPS,
        help=f"Max perturbation in CLIP normalized space (L_inf). Default {DEFAULT_EPS} for minimal visibility.",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Optimization steps")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate for perturbation")
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Use fallback (tint only, no CLIP) even if CLIP is available.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    strength = max(0.0, min(1.0, args.strength))

    if not HAS_CLIP or args.fallback:
        if HAS_CLIP and args.fallback:
            print("Using fallback (tint-only) mode as requested.")
        else:
            print("CLIP not available (install: pip install open-clip-torch torch pillow). Using fallback tint-only mode.")
        pil = Image.open(args.input).convert("RGB")
        arr = np.array(pil)
        out_arr = obfuscate_fallback(arr, args.concept, strength)
        Image.fromarray(out_arr).save(args.output, quality=95)
        print(f"Saved (fallback) to {args.output}")
        return

    obfuscate_with_clip(
        args.input,
        args.concept,
        args.output,
        eps=args.eps,
        steps=args.steps,
        lr=args.lr,
        strength=strength,
    )


if __name__ == "__main__":
    main()
