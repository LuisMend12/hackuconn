"""
PhotoGuard PGD attack: perturbs the image in VAE latent space so that img2img editing is disrupted.
"""
import torch
from tqdm import tqdm


def pgd(
    X: torch.Tensor,
    model,
    eps: float = 0.1,
    step_size: float = 0.015,
    iters: int = 40,
    clamp_min: float = 0,
    clamp_max: float = 1,
    mask: torch.Tensor | None = None,
    device: str | None = None,
) -> torch.Tensor:
    """
    Projected Gradient Descent attack on the VAE encoder.
    Maximizes the latent norm so that downstream img2img fails to follow the source image.
    """
    if device is None:
        device = next(model.parameters()).device
    X = X.to(device)
    X_adv = X.clone().detach() + (torch.rand(*X.shape, device=device) * 2 * eps - eps)
    pbar = tqdm(range(iters), desc="PhotoGuard attack")
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        X_adv = X_adv.to(device)
        X_adv.requires_grad_(True)
        loss = (model(X_adv).latent_dist.mean).norm()
        pbar.set_postfix(loss=f"{loss.item():.5f}", step_size=f"{actual_step_size:.4f}")
        grad, = torch.autograd.grad(loss, [X_adv])
        X_adv = X_adv.detach() - grad.sign() * actual_step_size
        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        if mask is not None:
            X_adv = X_adv * mask
    return X_adv
