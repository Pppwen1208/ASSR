from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F


def _to_gray_np_batch(x: torch.Tensor) -> list[np.ndarray]:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"expected [B,C,H,W] or [C,H,W], got {tuple(x.shape)}")
    x = x.detach().float().cpu().clamp(0.0, 1.0)
    if x.shape[1] == 3:
        x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    elif x.shape[1] != 1:
        x = x.mean(dim=1, keepdim=True)
    out: list[np.ndarray] = []
    for i in range(x.shape[0]):
        out.append(x[i, 0].numpy())
    return out


def _prepare_valid_mask(
    valid_mask: torch.Tensor | None,
    ref: torch.Tensor,
) -> torch.Tensor | None:
    if valid_mask is None:
        return None
    m = valid_mask
    if m.ndim == 3:
        m = m.unsqueeze(0)
    if m.ndim != 4:
        raise ValueError(f"expected valid mask [B,1,H,W] or [1,H,W], got {tuple(m.shape)}")
    m = m.to(device=ref.device, dtype=ref.dtype)
    if m.shape[1] != 1:
        m = m.mean(dim=1, keepdim=True)
    if m.shape[-2:] != ref.shape[-2:]:
        m = F.interpolate(m, size=ref.shape[-2:], mode="nearest")
    return (m > 0.5).to(ref.dtype)


def _canny_cv2(img: np.ndarray, low: int, high: int) -> np.ndarray:
    import cv2

    u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    e = cv2.Canny(u8, threshold1=low, threshold2=high, L2gradient=True)
    return (e > 0).astype(np.float32)


def _canny_skimage(img: np.ndarray, low: int, high: int, sigma: float) -> np.ndarray:
    from skimage.feature import canny

    e = canny(
        img,
        sigma=sigma,
        low_threshold=float(low) / 255.0,
        high_threshold=float(high) / 255.0,
    )
    return e.astype(np.float32)


def _sobel_fallback(img: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(img, dtype=np.float32)
    gy = np.zeros_like(img, dtype=np.float32)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    g = np.sqrt(gx * gx + gy * gy + 1e-8)
    thr = float(g.mean() + 0.5 * g.std())
    return (g > thr).astype(np.float32)


def canny_edges(
    x: torch.Tensor,
    low_threshold: int = 100,
    high_threshold: int = 200,
    sigma: float = 1.2,
    backend: Literal["auto", "cv2", "skimage", "sobel"] = "auto",
) -> torch.Tensor:
    imgs = _to_gray_np_batch(x)
    edges: list[np.ndarray] = []
    for img in imgs:
        e: np.ndarray | None = None
        if backend in ("auto", "cv2"):
            try:
                e = _canny_cv2(img, low=low_threshold, high=high_threshold)
            except Exception:
                e = None
        if e is None and backend in ("auto", "skimage"):
            try:
                e = _canny_skimage(
                    img,
                    low=low_threshold,
                    high=high_threshold,
                    sigma=sigma,
                )
            except Exception:
                e = None
        if e is None:
            e = _sobel_fallback(img)
        edges.append(e)

    e_np = np.stack(edges, axis=0)[:, None, ...]
    return torch.from_numpy(e_np)


def edge_f1(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    low_threshold: int = 100,
    high_threshold: int = 200,
    sigma: float = 1.2,
    tolerance: int = 1,
    backend: Literal["auto", "cv2", "skimage", "sobel"] = "auto",
    eps: float = 1e-8,
) -> float:
    pred_e = canny_edges(
        pred,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        sigma=sigma,
        backend=backend,
    ).to(dtype=torch.float32)
    targ_e = canny_edges(
        target,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        sigma=sigma,
        backend=backend,
    ).to(dtype=torch.float32)

    mask = _prepare_valid_mask(valid_mask, pred_e)
    if mask is not None:
        pred_e = pred_e * mask
        targ_e = targ_e * mask

    if tolerance > 0:
        k = 2 * tolerance + 1
        pred_d = F.max_pool2d(pred_e, kernel_size=k, stride=1, padding=tolerance)
        targ_d = F.max_pool2d(targ_e, kernel_size=k, stride=1, padding=tolerance)
    else:
        pred_d = pred_e
        targ_d = targ_e

    tp_prec = (pred_e * targ_d).sum().item()
    tp_rec = (targ_e * pred_d).sum().item()
    pred_sum = pred_e.sum().item()
    targ_sum = targ_e.sum().item()
    if pred_sum < eps and targ_sum < eps:
        return 1.0
    if pred_sum < eps or targ_sum < eps:
        return 0.0
    precision = tp_prec / (pred_sum + eps)
    recall = tp_rec / (targ_sum + eps)
    return float((2.0 * precision * recall) / (precision + recall + eps))
