from __future__ import annotations

import math

import torch


def _hann2d(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    wy = torch.hann_window(h, periodic=False, device=device, dtype=dtype).clamp_min(1e-3)
    wx = torch.hann_window(w, periodic=False, device=device, dtype=dtype).clamp_min(1e-3)
    return (wy[:, None] * wx[None, :]).unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def tiled_infer(
    model: torch.nn.Module,
    s2_lr: torch.Tensor,
    s1_lr: torch.Tensor | None,
    scale: float,
    tile_size: int = 256,
    hr_overlap: int = 16,
    text_embed: torch.Tensor | None = None,
    text_mask: torch.Tensor | None = None,
    enable_risk_gate: bool = False,
    risk_gate_strength: float = 0.35,
    return_aux: bool = False,
) -> torch.Tensor | dict[str, torch.Tensor]:
    b, _, h, w = s2_lr.shape
    if b != 1:
        raise ValueError("tiled_infer currently supports batch size 1")

    oh = max(1, int(round(h * scale)))
    ow = max(1, int(round(w * scale)))
    out = torch.zeros(1, 3, oh, ow, device=s2_lr.device, dtype=s2_lr.dtype)
    risk_out = torch.zeros(1, 1, oh, ow, device=s2_lr.device, dtype=s2_lr.dtype)
    norm = torch.zeros(1, 1, oh, ow, device=s2_lr.device, dtype=s2_lr.dtype)

    lr_overlap = int(round(hr_overlap / max(scale, 1e-6)))
    stride = max(1, tile_size - lr_overlap)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1 = min(y + tile_size, h)
            x1 = min(x + tile_size, w)
            tile_s2 = s2_lr[:, :, y:y1, x:x1]
            tile_s1 = s1_lr[:, :, y:y1, x:x1] if s1_lr is not None else None

            tile_out = model(
                s2_lr=tile_s2,
                s1_lr=tile_s1,
                scale=scale,
                text_embed=text_embed,
                text_mask=text_mask,
                enable_risk_gate=enable_risk_gate,
                risk_gate_strength=risk_gate_strength,
            )
            tile_sr = tile_out["sr"]
            tile_risk = tile_out["risk_map"]
            _, _, th, tw = tile_sr.shape

            oy = int(round(y * scale))
            ox = int(round(x * scale))
            oy1 = min(oy + th, oh)
            ox1 = min(ox + tw, ow)
            tile_sr = tile_sr[:, :, : oy1 - oy, : ox1 - ox]
            tile_risk = tile_risk[:, :, : oy1 - oy, : ox1 - ox]

            wmap = _hann2d(tile_sr.shape[2], tile_sr.shape[3], out.device, out.dtype)
            out[:, :, oy:oy1, ox:ox1] += tile_sr * wmap
            risk_out[:, :, oy:oy1, ox:ox1] += tile_risk * wmap
            norm[:, :, oy:oy1, ox:ox1] += wmap

    sr = out / norm.clamp_min(1e-6)
    risk_map = (risk_out / norm.clamp_min(1e-6)).clamp(0.0, 1.0)
    if not return_aux:
        return sr
    semantic_consistency = (1.0 - risk_map.mean(dim=(1, 2, 3))).clamp(0.0, 1.0)
    return {"sr": sr, "risk_map": risk_map, "semantic_consistency": semantic_consistency}
