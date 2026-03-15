from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ResizeMeta:
    kappa: float = 1.0
    eta: float = 0.0


class MetaFourierEncoder(nn.Module):
    def __init__(self, in_dim: int, bands: int) -> None:
        super().__init__()
        freqs = 2.0 ** torch.arange(bands).float() * math.pi
        self.register_buffer("freqs", freqs, persistent=False)
        self.in_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Q, in_dim]
        xb = x.unsqueeze(-1) * self.freqs
        sin = torch.sin(xb)
        cos = torch.cos(xb)
        out = torch.cat([x.unsqueeze(-1), sin, cos], dim=-1)
        return out.reshape(x.shape[0], -1)


class MetaUpsampler(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        out_channels: int = 3,
        kernel_size: int = 3,
        fourier_bands: int = 8,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.channels = channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.k2 = kernel_size * kernel_size

        self.meta_ff = MetaFourierEncoder(in_dim=5, bands=fourier_bands)
        meta_dim = 5 * (1 + 2 * fourier_bands)
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

        self.kernel_gen = nn.Sequential(
            nn.Linear(self.k2 * channels + channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, self.k2),
        )
        self.rgb_head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, out_channels),
        )

    @staticmethod
    def _query_geometry(
        h: int,
        w: int,
        scale: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        oh = max(1, int(round(h * scale)))
        ow = max(1, int(round(w * scale)))

        ys = torch.arange(oh, device=device, dtype=dtype)
        xs = torch.arange(ow, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        src_y = (yy + 0.5) / scale - 0.5
        src_x = (xx + 0.5) / scale - 0.5
        cell_y = torch.floor(src_y)
        cell_x = torch.floor(src_x)

        du = src_x - cell_x - 0.5
        dv = src_y - cell_y - 0.5

        return src_y, src_x, du, dv, oh, ow

    def _sample_local_patches(
        self,
        feat: torch.Tensor,
        src_y: torch.Tensor,
        src_x: torch.Tensor,
    ) -> torch.Tensor:
        # feat: [1, C, H, W], output: [Q, k2, C]
        _, c, h, w = feat.shape
        pad = self.kernel_size // 2
        padded = F.pad(feat, (pad, pad, pad, pad), mode="reflect")
        unfolded = F.unfold(padded, kernel_size=self.kernel_size)  # [1, C*k2, H*W]
        unfolded = unfolded.squeeze(0).transpose(0, 1).contiguous()  # [H*W, C*k2]

        cy = torch.floor(src_y).clamp(0, h - 1).long().reshape(-1)
        cx = torch.floor(src_x).clamp(0, w - 1).long().reshape(-1)
        idx = cy * w + cx  # [Q]

        gathered = unfolded[idx]  # [Q, C*k2]
        return gathered.view(gathered.shape[0], self.k2, c)

    def _forward_single(
        self,
        feat: torch.Tensor,
        scale: float,
        meta: ResizeMeta | None = None,
    ) -> torch.Tensor:
        # feat: [1, C, H, W]
        if meta is None:
            meta = ResizeMeta(kappa=max(0.1, 1.0 / scale), eta=min(1.0, 0.15 * scale))

        _, _, h, w = feat.shape
        src_y, src_x, du, dv, oh, ow = self._query_geometry(
            h, w, scale, feat.device, feat.dtype
        )
        patches = self._sample_local_patches(feat, src_y, src_x)  # [Q, k2, C]
        q = patches.shape[0]

        svec = torch.full((q,), float(scale), device=feat.device, dtype=feat.dtype)
        kappa = torch.full((q,), float(meta.kappa), device=feat.device, dtype=feat.dtype)
        eta = torch.full((q,), float(meta.eta), device=feat.device, dtype=feat.dtype)
        m = torch.stack([svec, du.reshape(-1), dv.reshape(-1), kappa, eta], dim=1)
        m_emb = self.meta_mlp(self.meta_ff(m))

        patches_flat = patches.reshape(q, -1)
        weights = self.kernel_gen(torch.cat([patches_flat, m_emb], dim=1))
        weights = torch.softmax(weights, dim=1)  # [Q, k2]

        f_q = (weights.unsqueeze(-1) * patches).sum(dim=1)  # [Q, C]
        rgb = self.rgb_head(f_q)  # [Q, 3]
        rgb = rgb.view(oh, ow, self.out_channels).permute(2, 0, 1).unsqueeze(0)
        return rgb

    def forward(
        self,
        feat: torch.Tensor,
        scale: torch.Tensor | float,
        metas: list[ResizeMeta] | None = None,
    ) -> torch.Tensor:
        # feat: [B, C, H, W]
        b = feat.shape[0]
        if isinstance(scale, torch.Tensor):
            if scale.ndim == 0:
                scale_vals = [float(scale.item()) for _ in range(b)]
            else:
                scale_vals = [float(v.item()) for v in scale]
        else:
            scale_vals = [float(scale) for _ in range(b)]

        if metas is None:
            metas = [None for _ in range(b)]

        outs = []
        for i in range(b):
            outs.append(
                self._forward_single(
                    feat=feat[i : i + 1],
                    scale=scale_vals[i],
                    meta=metas[i],
                )
            )
        return torch.cat(outs, dim=0)
