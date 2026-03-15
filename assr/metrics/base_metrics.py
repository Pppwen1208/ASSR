from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

_LPIPS_MODELS: dict[str, Any] = {}


def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"expected [B,C,H,W] or [C,H,W], got {tuple(x.shape)}")
    return x


def _prepare_valid_mask(
    valid_mask: torch.Tensor | None,
    ref: torch.Tensor,
) -> torch.Tensor | None:
    if valid_mask is None:
        return None
    m = _ensure_4d(valid_mask).to(device=ref.device, dtype=ref.dtype)
    if m.shape[1] != 1:
        m = m.mean(dim=1, keepdim=True)
    if m.shape[-2:] != ref.shape[-2:]:
        m = F.interpolate(m, size=ref.shape[-2:], mode="nearest")
    return (m > 0.5).to(ref.dtype)


def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> float:
    pred = _ensure_4d(pred)
    target = _ensure_4d(target)
    mask = _prepare_valid_mask(valid_mask, pred)
    if mask is None:
        mse = F.mse_loss(pred, target).item()
    else:
        diff2 = (pred - target) ** 2
        num = (diff2 * mask).sum().item()
        den = (mask.sum().item() * pred.shape[1]) + eps
        mse = num / den
    if mse < eps:
        return 100.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse + eps))


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    c1: float = 0.01**2,
    c2: float = 0.03**2,
) -> float:
    pred = _ensure_4d(pred)
    target = _ensure_4d(target)
    mask = _prepare_valid_mask(valid_mask, pred)
    mu_x = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
    sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y

    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim_map = num / den.clamp_min(1e-8)
    if mask is None:
        return float(ssim_map.mean().item())
    v = (ssim_map * mask).sum() / (mask.sum() * pred.shape[1] + 1e-8)
    return float(v.item())


def _get_lpips_model(device: torch.device) -> Any | None:
    key = str(device)
    if key in _LPIPS_MODELS:
        return _LPIPS_MODELS[key]
    try:
        import lpips

        model = lpips.LPIPS(net="vgg").eval().to(device)
        _LPIPS_MODELS[key] = model
        return model
    except Exception:
        return None


def lpips_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> float:
    pred = _ensure_4d(pred)
    target = _ensure_4d(target)
    mask = _prepare_valid_mask(valid_mask, pred)
    if mask is not None:
        pred = pred * mask
        target = target * mask
    model = _get_lpips_model(pred.device)
    if model is None:
        return float("nan")
    with torch.no_grad():
        p = pred * 2.0 - 1.0
        t = target * 2.0 - 1.0
        d = model(p, t).mean().item()
    return float(d)
