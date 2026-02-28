"""
PhotoGuard utilities: image preprocessing and recovery for the adversarial protection pipeline.
"""
from PIL import Image
import torch
import torchvision.transforms as T


# Standard size for Stable Diffusion v1.x
IMG_SIZE = 512

_resize = T.Resize(IMG_SIZE)
_center_crop = T.CenterCrop(IMG_SIZE)
_to_tensor = T.ToTensor()


def preprocess(pil_image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL image to a batched tensor in [-1, 1] range, resized and center-cropped to 512x512.
    Suitable for feeding into Stable Diffusion's VAE.
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    pil_image = _center_crop(_resize(pil_image))
    x = _to_tensor(pil_image).unsqueeze(0)  # (1, 3, H, W), [0, 1]
    x = x * 2.0 - 1.0  # [-1, 1]
    return x


def recover_image(
    adv_image: Image.Image,
    init_image: Image.Image,
    mask_image: Image.Image | None = None,
    background: bool = False,
) -> Image.Image:
    """
    Optionally recover regions of the adversarial image using the original and a mask.
    If no mask or background=False, returns adv_image unchanged.
    """
    if mask_image is None or not background:
        return adv_image
    import numpy as np
    adv = np.array(adv_image.convert("RGB"))
    init = np.array(init_image.convert("RGB").resize(adv_image.size))
    mask = np.array(mask_image.convert("L").resize(adv_image.size), dtype=np.float32) / 255.0
    mask = mask[:, :, np.newaxis]
    out = (adv * (1 - mask) + init * mask).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)
