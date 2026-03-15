from __future__ import annotations

import torch
import torch.nn as nn


class VGGPerceptualLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        try:
            from torchvision.models import VGG19_Weights, vgg19

            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:35]
            for p in vgg.parameters():
                p.requires_grad = False
            self.backbone = vgg.eval()
            self.enabled = True
        except Exception:
            self.backbone = nn.Identity()
            self.enabled = False
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return pred.new_tensor(0.0)
        fp = self.backbone(pred)
        ft = self.backbone(target)
        return self.l1(fp, ft)

