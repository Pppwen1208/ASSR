from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from assr.config.schema import ModelConfig
from assr.models.attention import ScaleToken, TextScaleCrossAttentionBlock
from assr.models.meta_upsampler import MetaUpsampler, ResizeMeta
from assr.models.rrdb import RRDBBackbone


class ASSR(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        s2_channels: int = 3,
        s1_channels: int = 1,
        use_s1: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_s1 = use_s1

        c = cfg.channels
        self.s2_stem = nn.Conv2d(s2_channels, c, 3, 1, 1)
        self.s1_stem = nn.Conv2d(s1_channels, c, 3, 1, 1) if use_s1 else None
        self.scale_gate_token = ScaleToken(out_dim=c, fourier_bands=cfg.fourier_bands)
        self.text_gate_proj = nn.Linear(cfg.text_embed_dim, c)
        self.gate_conv = nn.Conv2d(c * 4, c, kernel_size=1, stride=1, padding=0)

        self.backbone = RRDBBackbone(
            channels=cfg.channels,
            num_blocks=cfg.rrdb_blocks,
            growth_channels=cfg.growth_channels,
        )
        self.cross_blocks = nn.ModuleList(
            [
                TextScaleCrossAttentionBlock(
                    channels=cfg.channels,
                    text_embed_dim=cfg.text_embed_dim,
                    num_heads=cfg.num_heads,
                    attn_dropout=cfg.attn_dropout,
                    token_dropout=cfg.token_dropout,
                    ff_expand=cfg.ff_expand,
                    fourier_bands=cfg.fourier_bands,
                    adaln_eps=cfg.adaln_eps,
                ),
                TextScaleCrossAttentionBlock(
                    channels=cfg.channels,
                    text_embed_dim=cfg.text_embed_dim,
                    num_heads=cfg.num_heads,
                    attn_dropout=cfg.attn_dropout,
                    token_dropout=cfg.token_dropout,
                    ff_expand=cfg.ff_expand,
                    fourier_bands=cfg.fourier_bands,
                    adaln_eps=cfg.adaln_eps,
                ),
            ]
        )
        self.mid_idx = cfg.rrdb_blocks // 2
        self.late_idx = max(cfg.rrdb_blocks - 2, 0)

        self.ps_pre = nn.Conv2d(c, c * (cfg.pixelshuffle_scale**2), 3, 1, 1)
        self.ps = nn.PixelShuffle(cfg.pixelshuffle_scale)
        self.ps_post = nn.Conv2d(c, cfg.rgb_out_channels, 3, 1, 1)

        self.meta_upsampler = MetaUpsampler(
            channels=cfg.channels,
            out_channels=cfg.rgb_out_channels,
            kernel_size=cfg.meta_kernel_size,
            fourier_bands=cfg.fourier_bands,
        )

    @staticmethod
    def _log_homomorphic_norm(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.log1p(torch.abs(x))

    @staticmethod
    def _as_scale_tensor(scale: float | torch.Tensor, batch: int, ref: torch.Tensor) -> torch.Tensor:
        if isinstance(scale, torch.Tensor):
            if scale.ndim == 0:
                return scale.expand(batch).to(device=ref.device, dtype=ref.dtype)
            return scale.to(device=ref.device, dtype=ref.dtype).view(batch)
        return torch.full((batch,), float(scale), device=ref.device, dtype=ref.dtype)

    @staticmethod
    def _all_close_to_x4(scale: torch.Tensor, atol: float = 1e-6) -> bool:
        return bool(torch.all(torch.abs(scale - 4.0) <= atol).item())

    def _pool_semantic_prior(
        self,
        text_embed: torch.Tensor | None,
        text_mask: torch.Tensor | None,
        batch: int,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        c = self.cfg.channels
        if text_embed is None:
            return torch.zeros(batch, c, device=ref.device, dtype=ref.dtype)

        if text_embed.ndim == 2:
            text_embed = text_embed.unsqueeze(0)
        if text_embed.shape[0] != batch:
            text_embed = text_embed.expand(batch, -1, -1)
        text = self.text_gate_proj(text_embed)
        if text_mask is None:
            mask = torch.ones(
                text.shape[0],
                text.shape[1],
                device=text.device,
                dtype=torch.bool,
            )
        else:
            if text_mask.ndim == 1:
                text_mask = text_mask.unsqueeze(0)
            if text_mask.shape[0] != batch:
                text_mask = text_mask.expand(batch, -1)
            mask = text_mask.to(device=text.device, dtype=torch.bool)

        weight = mask.float().unsqueeze(-1)
        pooled = (text * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        return pooled.to(dtype=ref.dtype)

    def _dynamic_gated_fusion(
        self,
        s2_feat: torch.Tensor,
        s1_feat_norm: torch.Tensor,
        scale: torch.Tensor,
        text_embed: torch.Tensor | None,
        text_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        b, _, h, w = s2_feat.shape
        e_s = self.scale_gate_token(scale)
        e_c = self._pool_semantic_prior(
            text_embed=text_embed,
            text_mask=text_mask,
            batch=b,
            ref=s2_feat,
        )
        e_s_map = e_s.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        e_c_map = e_c.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        gate_in = torch.cat([s2_feat, s1_feat_norm, e_s_map, e_c_map], dim=1)
        gate = torch.sigmoid(self.gate_conv(gate_in))
        return s2_feat + gate * s1_feat_norm

    @staticmethod
    def _to_resize_meta(
        resize_meta: list[dict[str, float]] | list[ResizeMeta] | None,
        batch: int,
        scale: torch.Tensor,
    ) -> list[ResizeMeta]:
        if resize_meta is None:
            metas: list[ResizeMeta] = []
            for i in range(batch):
                s = float(scale[i].item())
                metas.append(ResizeMeta(kappa=max(0.1, 1.0 / s), eta=min(1.0, 0.15 * s)))
            return metas

        out: list[ResizeMeta] = []
        for i in range(batch):
            item = resize_meta[i] if i < len(resize_meta) else resize_meta[-1]
            if isinstance(item, ResizeMeta):
                out.append(item)
            else:
                out.append(
                    ResizeMeta(
                        kappa=float(item.get("kappa", max(0.1, 1.0 / float(scale[i].item())))),
                        eta=float(item.get("eta", min(1.0, 0.15 * float(scale[i].item())))),
                    )
                )
        return out

    def _backbone_forward(
        self,
        fused: torch.Tensor,
        text_embed: torch.Tensor | None,
        text_mask: torch.Tensor | None,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        mod_blocks = {
            self.mid_idx: self.cross_blocks[0],
            self.late_idx: self.cross_blocks[1],
        }
        return self.backbone(
            fused,
            modulation_blocks=mod_blocks,
            modulation_kwargs={
                "text_embed": text_embed,
                "text_mask": text_mask,
                "scale": scale,
            },
        )

    @staticmethod
    def _normalize_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # Normalize per-sample to [0, 1]
        b = x.shape[0]
        x_flat = x.view(b, -1)
        min_v = x_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        max_v = x_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        return (x - min_v) / (max_v - min_v + eps)

    def _compute_risk_outputs(
        self,
        lr_feat: torch.Tensor,
        s2_feat: torch.Tensor,
        s1_feat: torch.Tensor | None,
        out_h: int,
        out_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # A lightweight uncertainty proxy for deployment diagnostics.
        feat_var = torch.var(lr_feat, dim=1, keepdim=True, unbiased=False)
        risk_lr = self._normalize_map(feat_var)

        if s1_feat is not None:
            modality_gap = torch.mean(torch.abs(s2_feat - s1_feat), dim=1, keepdim=True)
            modality_gap = self._normalize_map(modality_gap)
            risk_lr = 0.65 * risk_lr + 0.35 * modality_gap

        risk_hr = F.interpolate(
            risk_lr,
            size=(out_h, out_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0.0, 1.0)
        semantic_consistency = (1.0 - risk_hr.mean(dim=(1, 2, 3))).clamp(0.0, 1.0)
        return risk_hr, semantic_consistency

    def forward(
        self,
        s2_lr: torch.Tensor,
        s1_lr: torch.Tensor | None = None,
        scale: float | torch.Tensor = 4.0,
        text_embed: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        resize_meta: list[dict[str, float]] | list[ResizeMeta] | None = None,
        enable_risk_gate: bool = False,
        risk_gate_strength: float = 0.35,
    ) -> dict[str, torch.Tensor]:
        b = s2_lr.shape[0]
        scale_t = self._as_scale_tensor(scale, batch=b, ref=s2_lr)

        s2_feat = self.s2_stem(s2_lr)
        fused = s2_feat
        s1_feat_norm = None

        if self.use_s1 and self.s1_stem is not None and s1_lr is not None:
            s1_feat = self.s1_stem(s1_lr)
            s1_feat_norm = self._log_homomorphic_norm(s1_feat)
            fused = self._dynamic_gated_fusion(
                s2_feat=s2_feat,
                s1_feat_norm=s1_feat_norm,
                scale=scale_t,
                text_embed=text_embed,
                text_mask=text_mask,
            )

        lr_feat = self._backbone_forward(
            fused=fused, text_embed=text_embed, text_mask=text_mask, scale=scale_t
        )

        if self._all_close_to_x4(scale_t):
            out = self.ps_post(self.ps(self.ps_pre(lr_feat)))
        else:
            metas = self._to_resize_meta(resize_meta, b, scale_t)
            out = self.meta_upsampler(lr_feat, scale=scale_t, metas=metas)

        out = out.clamp(0.0, 1.0)
        risk_map, semantic_consistency = self._compute_risk_outputs(
            lr_feat=lr_feat,
            s2_feat=s2_feat,
            s1_feat=s1_feat_norm,
            out_h=out.shape[-2],
            out_w=out.shape[-1],
        )

        if enable_risk_gate:
            gate = risk_map.clamp(0.0, 1.0)
            smooth = F.avg_pool2d(out, kernel_size=3, stride=1, padding=1)
            out = out * (1.0 - risk_gate_strength * gate) + smooth * (
                risk_gate_strength * gate
            )
            out = out.clamp(0.0, 1.0)

        return {
            "sr": out,
            "risk_map": risk_map,
            "semantic_consistency": semantic_consistency,
        }

    def load_pretrained(self, path: str, strict: bool = True) -> None:
        state = torch.load(path, map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=strict)

    def export_state(self) -> dict[str, Any]:
        return {"model": self.state_dict()}
