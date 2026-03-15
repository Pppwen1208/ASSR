from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, bands: int) -> None:
        super().__init__()
        freqs = 2.0 ** torch.arange(bands).float() * math.pi
        self.register_buffer("freqs", freqs, persistent=False)
        self.in_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim]
        xb = x.unsqueeze(-1) * self.freqs
        sin = torch.sin(xb)
        cos = torch.cos(xb)
        out = torch.cat([x.unsqueeze(-1), sin, cos], dim=-1)
        return out.flatten(start_dim=1)


class ScaleToken(nn.Module):
    def __init__(self, out_dim: int, fourier_bands: int) -> None:
        super().__init__()
        self.ff = FourierFeatures(in_dim=1, bands=fourier_bands)
        ff_dim = 1 + 2 * fourier_bands
        self.mlp = nn.Sequential(
            nn.Linear(ff_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, scale: torch.Tensor) -> torch.Tensor:
        # scale: [B]
        s = scale.view(-1, 1)
        return self.mlp(self.ff(s))


class TextScaleCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        text_embed_dim: int = 768,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
        token_dropout: float = 0.1,
        ff_expand: int = 2,
        fourier_bands: int = 8,
        adaln_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.token_dropout = token_dropout

        self.text_proj = nn.Linear(text_embed_dim, channels)
        self.scale_token = ScaleToken(out_dim=channels, fourier_bands=fourier_bands)

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.attn_drop = nn.Dropout(attn_dropout)

        self.post_attn_ln = nn.LayerNorm(channels, eps=adaln_eps)
        self.ada_mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels * 2),
        )
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * ff_expand),
            nn.GELU(),
            nn.Linear(channels * ff_expand, channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        text_embed: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if text_embed is None:
            return x

        b, c, h, w = x.shape
        n = h * w
        x_seq = x.flatten(2).transpose(1, 2)  # [B, N, C]

        if scale is None:
            scale = x.new_ones(b)
        elif scale.ndim == 0:
            scale = scale.expand(b)

        scale_tok = self.scale_token(scale).unsqueeze(1)  # [B, 1, C]
        text_ctx = self.text_proj(text_embed)  # [B, L, C]

        if self.training and self.token_dropout > 0:
            keep = (
                torch.rand(b, text_ctx.shape[1], device=x.device) > self.token_dropout
            ).float()
            text_ctx = text_ctx * keep.unsqueeze(-1)
            if text_mask is not None:
                text_mask = text_mask & keep.bool()

        ctx = torch.cat([text_ctx, scale_tok], dim=1)

        if text_mask is None:
            text_mask = torch.ones(
                b, text_ctx.shape[1], device=x.device, dtype=torch.bool
            )
        scale_mask = torch.ones(b, 1, device=x.device, dtype=torch.bool)
        ctx_mask = torch.cat([text_mask, scale_mask], dim=1)  # [B, L+1]

        q = self.q_proj(x_seq).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(ctx).view(
            b, ctx.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(ctx).view(
            b, ctx.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_mask = (~ctx_mask).unsqueeze(1).unsqueeze(2)  # [B,1,1,L+1]
        attn = attn.masked_fill(attn_mask, -1e4)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, c)
        x_attn = x_seq + self.out_proj(out)
        x_attn = self.post_attn_ln(x_attn)

        ctx_weight = ctx_mask.float().unsqueeze(-1)
        ctx_pool = (ctx * ctx_weight).sum(dim=1) / ctx_weight.sum(dim=1).clamp_min(1.0)
        gamma_beta = self.ada_mlp(ctx_pool)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        gamma = torch.sigmoid(gamma).unsqueeze(1)
        beta = beta.unsqueeze(1)

        x_norm = self.post_attn_ln(x_attn)
        x_mod = x_norm * (1.0 + gamma) + beta
        x_out = x_attn + self.ffn(x_mod)

        return x_out.transpose(1, 2).view(b, c, h, w)

