from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from assr.config.schema import TrainConfig


def project_scale(
    img: torch.Tensor,
    src_scale: torch.Tensor,
    dst_scale: torch.Tensor,
) -> torch.Tensor:
    b, _, h, w = img.shape
    out = []
    for i in range(b):
        s_src = float(src_scale[i].item())
        s_dst = float(dst_scale[i].item())
        base_h = max(1.0, h / max(s_src, 1e-6))
        base_w = max(1.0, w / max(s_src, 1e-6))
        th = max(1, int(round(base_h * s_dst)))
        tw = max(1, int(round(base_w * s_dst)))
        yi = F.interpolate(
            img[i : i + 1],
            size=(th, tw),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        out.append(yi)
    return torch.cat(out, dim=0)


class ASSRReconstructionLoss(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.lambda_consist = cfg.lambda_consist
        self.lambda_lr = cfg.lambda_lr
        self.lambda_pair = cfg.lambda_pair
        self.l1 = nn.L1Loss()

    def forward(
        self,
        pred_hr: torch.Tensor,
        target_hr: torch.Tensor,
        lr_ref: torch.Tensor,
        scale: torch.Tensor,
        pair_pred_hr: torch.Tensor | None = None,
        pair_scale: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pix = self.l1(pred_hr, target_hr)

        one_scale = torch.ones_like(scale)
        pred_lr = project_scale(pred_hr, src_scale=scale, dst_scale=one_scale)
        lr_term = self.l1(pred_lr, lr_ref)

        pair_term = pred_hr.new_tensor(0.0)
        if pair_pred_hr is not None and pair_scale is not None:
            proj = project_scale(pred_hr, src_scale=scale, dst_scale=pair_scale)
            if pair_pred_hr.shape[-2:] != proj.shape[-2:]:
                pair_pred_hr = F.interpolate(
                    pair_pred_hr,
                    size=proj.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
            pair_term = self.l1(proj, pair_pred_hr)

        consist = self.lambda_lr * lr_term + self.lambda_pair * pair_term
        total = pix + self.lambda_consist * consist
        return {
            "total": total,
            "pix": pix.detach(),
            "consist": consist.detach(),
            "lr_term": lr_term.detach(),
            "pair_term": pair_term.detach(),
        }
