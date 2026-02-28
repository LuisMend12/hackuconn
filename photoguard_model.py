"""
PhotoGuard model loading: Stable Diffusion img2img pipeline for the attack and optional demo.
"""
import torch
from diffusers import StableDiffusionImg2ImgPipeline

MODEL_ID = "runwayml/stable-diffusion-v1-5"
_pipe = None


def get_pipe(device: str | None = None):
    """Load and cache the Stable Diffusion img2img pipeline. Uses CUDA if available."""
    global _pipe
    if _pipe is None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        _pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
        )
        _pipe = _pipe.to(device)
    return _pipe
