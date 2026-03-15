from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from assr.metrics.base_metrics import lpips_distance, ssim
from assr.metrics.edge_metrics import edge_f1


def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"expected [B,C,H,W] or [C,H,W], got {tuple(x.shape)}")
    return x


def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(
        x,
        size=ref.shape[-2:],
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )


def project_between_scales(
    image: torch.Tensor,
    src_scale: float,
    dst_scale: float,
) -> torch.Tensor:
    """
    P_{a->b}(z): anti-aliased projection from scale a to b.
    """
    image = _ensure_4d(image)
    if abs(src_scale - dst_scale) < 1e-8:
        return image

    _, _, h, w = image.shape
    base_h = max(1.0, h / max(src_scale, 1e-8))
    base_w = max(1.0, w / max(src_scale, 1e-8))
    out_h = max(1, int(round(base_h * dst_scale)))
    out_w = max(1, int(round(base_w * dst_scale)))
    return F.interpolate(
        image,
        size=(out_h, out_w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )


def build_neighbor_reference(
    preds_by_scale: Mapping[float, torch.Tensor],
) -> dict[float, torch.Tensor]:
    scales = sorted(float(s) for s in preds_by_scale.keys())
    refs: dict[float, torch.Tensor] = {}
    if len(scales) == 0:
        return refs

    if len(scales) == 1:
        s = scales[0]
        refs[s] = _ensure_4d(preds_by_scale[s])
        return refs

    for i, s in enumerate(scales):
        cur = _ensure_4d(preds_by_scale[s])
        if i == 0:
            sn = scales[i + 1]
            ref = project_between_scales(_ensure_4d(preds_by_scale[sn]), sn, s)
        elif i == len(scales) - 1:
            sp = scales[i - 1]
            ref = project_between_scales(_ensure_4d(preds_by_scale[sp]), sp, s)
        else:
            sp = scales[i - 1]
            sn = scales[i + 1]
            ref_p = project_between_scales(_ensure_4d(preds_by_scale[sp]), sp, s)
            ref_n = project_between_scales(_ensure_4d(preds_by_scale[sn]), sn, s)
            ref = 0.5 * (ref_p + ref_n)
        refs[s] = _resize_like(ref, cur)
    return refs


def sas_scores(
    preds_by_scale: Mapping[float, torch.Tensor],
    metric: str = "ssim",
) -> dict[float, float]:
    refs = build_neighbor_reference(preds_by_scale)
    out: dict[float, float] = {}
    for s, pred in preds_by_scale.items():
        p = _ensure_4d(pred)
        r = _resize_like(_ensure_4d(refs[float(s)]), p)
        if metric == "ssim":
            out[float(s)] = ssim(p, r)
        elif metric in ("edge", "edge_f1"):
            out[float(s)] = edge_f1(p, r)
        else:
            raise ValueError(f"unsupported SAS metric: {metric}")
    return out


def sce_scores(
    preds_by_scale: Mapping[float, torch.Tensor],
    metric: str = "l1",
) -> dict[float, float]:
    refs = build_neighbor_reference(preds_by_scale)
    out: dict[float, float] = {}
    for s, pred in preds_by_scale.items():
        p = _ensure_4d(pred)
        r = _resize_like(_ensure_4d(refs[float(s)]), p)
        if metric == "l1":
            out[float(s)] = float(torch.mean(torch.abs(p - r)).item())
        elif metric == "lpips":
            out[float(s)] = lpips_distance(p, r)
        else:
            raise ValueError(f"unsupported SCE metric: {metric}")
    return out


def aggregate_scale_metric(
    per_scale_scores: Mapping[float, float],
    weights: Mapping[float, float] | None = None,
) -> float:
    if len(per_scale_scores) == 0:
        return float("nan")
    if weights is None:
        vals = [v for v in per_scale_scores.values() if v == v]
        return float(sum(vals) / max(len(vals), 1))

    num = 0.0
    den = 0.0
    for s, v in per_scale_scores.items():
        if v != v:
            continue
        w = float(weights.get(float(s), 1.0))
        num += w * float(v)
        den += w
    return float(num / max(den, 1e-8))


def evaluate_sas_sce(
    preds_by_scale: Mapping[float, torch.Tensor],
    scale_weights: Mapping[float, float] | None = None,
) -> dict[str, float | dict[float, float]]:
    sas_ssim = sas_scores(preds_by_scale, metric="ssim")
    sas_edge = sas_scores(preds_by_scale, metric="edge_f1")
    sce_l1 = sce_scores(preds_by_scale, metric="l1")
    sce_lpips = sce_scores(preds_by_scale, metric="lpips")

    return {
        "sas_ssim_per_scale": sas_ssim,
        "sas_edge_f1_per_scale": sas_edge,
        "sce_l1_per_scale": sce_l1,
        "sce_lpips_per_scale": sce_lpips,
        "sas_ssim_global": aggregate_scale_metric(sas_ssim, weights=scale_weights),
        "sas_edge_f1_global": aggregate_scale_metric(sas_edge, weights=scale_weights),
        "sce_l1_global": aggregate_scale_metric(sce_l1, weights=scale_weights),
        "sce_lpips_global": aggregate_scale_metric(sce_lpips, weights=scale_weights),
    }
