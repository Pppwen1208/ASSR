from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def fix_random_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, p)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


def tensor_to_numpy_image(
    x: torch.Tensor,
    clamp: bool = True,
    channel_first: bool = True,
) -> np.ndarray:
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"expected [C,H,W] or [1,C,H,W], got {tuple(x.shape)}")
    t = x.detach().float().cpu()
    if clamp:
        t = t.clamp(0.0, 1.0)
    if channel_first:
        arr = t.permute(1, 2, 0).numpy()
    else:
        arr = t.numpy()
    return arr


def tensor_to_pil_image(x: torch.Tensor, clamp: bool = True) -> Image.Image:
    arr = tensor_to_numpy_image(x, clamp=clamp, channel_first=True)
    if arr.shape[2] == 1:
        u8 = (arr[:, :, 0] * 255.0).astype(np.uint8)
        return Image.fromarray(u8, mode="L")
    u8 = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(u8, mode="RGB")

