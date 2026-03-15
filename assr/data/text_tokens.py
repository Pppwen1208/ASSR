from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch


class TextEmbeddingAdapter:
    """
    Converts offline token ids or precomputed vectors into fixed-size token embeddings.
    This keeps the training/inference path independent from external caption models.
    """

    def __init__(self, max_len: int = 16, embed_dim: int = 768) -> None:
        self.max_len = max_len
        self.embed_dim = embed_dim
        self._basis = torch.linspace(0.01, 1.0, embed_dim)

    def _id_to_vec(self, token_id: int) -> torch.Tensor:
        # Deterministic sinusoid encoding for lightweight offline token handling.
        t = float(token_id) + 1.0
        vec = torch.sin(self._basis * t) + torch.cos(self._basis * (t * 0.5))
        return vec

    def from_ids(self, token_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        embed = torch.zeros(self.max_len, self.embed_dim)
        mask = torch.zeros(self.max_len, dtype=torch.bool)
        used = min(len(token_ids), self.max_len)
        for i in range(used):
            embed[i] = self._id_to_vec(int(token_ids[i]))
            mask[i] = True
        return embed, mask

    def from_array(self, arr: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(arr, np.ndarray):
            arr_t = torch.from_numpy(arr).float()
        else:
            arr_t = arr.float()
        if arr_t.ndim != 2:
            raise ValueError("text embedding array must have shape [L, D]")
        l, d = arr_t.shape
        if d != self.embed_dim:
            raise ValueError(f"expected text embed dim {self.embed_dim}, got {d}")

        embed = torch.zeros(self.max_len, self.embed_dim)
        mask = torch.zeros(self.max_len, dtype=torch.bool)
        used = min(l, self.max_len)
        embed[:used] = arr_t[:used]
        mask[:used] = True
        return embed, mask

    def from_path(self, path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
        p = Path(path)
        if p.suffix.lower() == ".npy":
            arr = np.load(p)
            return self.from_array(arr)
        raise ValueError(f"unsupported text embedding file: {path}")

