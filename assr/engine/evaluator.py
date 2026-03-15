from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from assr.metrics.image_metrics import edge_f1, lpips_distance, psnr, ssim
from assr.metrics.scale_metrics import evaluate_sas_sce


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    use_amp: bool = True,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    stats: dict[str, list[float]] = defaultdict(list)

    for batch_idx, batch in enumerate(tqdm(loader, desc="eval", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        for sample in batch:
            s2_lr = sample["s2_lr"].unsqueeze(0).to(device)
            s1_lr = (
                sample["s1_lr"].unsqueeze(0).to(device)
                if sample["s1_lr"] is not None
                else None
            )
            hr = sample["s2_hr"].unsqueeze(0).to(device)
            valid_mask = (
                sample["valid_mask"].unsqueeze(0).to(device)
                if sample.get("valid_mask") is not None
                else None
            )
            scale = sample["scale"].to(device)
            text_embed = sample["text_embed"].unsqueeze(0).to(device)
            text_mask = sample["text_mask"].unsqueeze(0).to(device)

            with torch.autocast(
                device_type=device.type,
                enabled=use_amp and device.type == "cuda",
            ):
                pred_out = model(
                    s2_lr=s2_lr,
                    s1_lr=s1_lr,
                    scale=scale,
                    text_embed=text_embed,
                    text_mask=text_mask,
                    resize_meta=[sample["resize_meta"]],
                    enable_risk_gate=False,
                )
                pred = pred_out["sr"]
            if pred.shape[-2:] != hr.shape[-2:]:
                pred = F.interpolate(
                    pred,
                    size=hr.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
            if valid_mask is not None and valid_mask.shape[-2:] != pred.shape[-2:]:
                valid_mask = F.interpolate(valid_mask, size=pred.shape[-2:], mode="nearest")

            pred = pred.clamp(0.0, 1.0)
            stats["psnr"].append(psnr(pred, hr, valid_mask=valid_mask))
            stats["ssim"].append(ssim(pred, hr, valid_mask=valid_mask))
            stats["edge_f1"].append(edge_f1(pred, hr, valid_mask=valid_mask))
            stats["lpips"].append(lpips_distance(pred, hr, valid_mask=valid_mask))

    out = {}
    for k, v in stats.items():
        if len(v) == 0:
            out[k] = float("nan")
        else:
            valid = [x for x in v if x == x]
            out[k] = float(sum(valid) / max(len(valid), 1))
    return out


@torch.no_grad()
def evaluate_scale_stability(
    model: torch.nn.Module,
    loader,
    scales: list[float],
    device: torch.device,
    use_amp: bool = True,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    if len(scales) == 0:
        return {}

    stats: dict[str, list[float]] = defaultdict(list)
    for batch_idx, batch in enumerate(tqdm(loader, desc="sas_sce", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        for sample in batch:
            s2_lr = sample["s2_lr"].unsqueeze(0).to(device)
            s1_lr = (
                sample["s1_lr"].unsqueeze(0).to(device)
                if sample["s1_lr"] is not None
                else None
            )
            text_embed = sample["text_embed"].unsqueeze(0).to(device)
            text_mask = sample["text_mask"].unsqueeze(0).to(device)
            resize_meta = [sample["resize_meta"]]

            preds: dict[float, torch.Tensor] = {}
            for s in scales:
                with torch.autocast(
                    device_type=device.type,
                    enabled=use_amp and device.type == "cuda",
                ):
                    pred = model(
                        s2_lr=s2_lr,
                        s1_lr=s1_lr,
                        scale=torch.tensor([s], device=device, dtype=s2_lr.dtype),
                        text_embed=text_embed,
                        text_mask=text_mask,
                        resize_meta=resize_meta,
                        enable_risk_gate=False,
                    )["sr"]
                preds[float(s)] = pred.detach()

            score_dict = evaluate_sas_sce(preds)
            stats["sas_ssim"].append(float(score_dict["sas_ssim_global"]))
            stats["sas_edge_f1"].append(float(score_dict["sas_edge_f1_global"]))
            stats["sce_l1"].append(float(score_dict["sce_l1_global"]))
            stats["sce_lpips"].append(float(score_dict["sce_lpips_global"]))

    out = {}
    for k, values in stats.items():
        valid = [v for v in values if v == v]
        out[k] = float(sum(valid) / max(len(valid), 1))
    return out
