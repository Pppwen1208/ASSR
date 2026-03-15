from __future__ import annotations

import torch


def hinge_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    loss_real = torch.relu(1.0 - real_logits).mean()
    loss_fake = torch.relu(1.0 + fake_logits).mean()
    return loss_real + loss_fake


def hinge_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


def sigmoid_ramp(step: int, start: int, tau: int) -> float:
    x = (step - start) / max(tau, 1)
    return float(1.0 / (1.0 + torch.exp(torch.tensor(-x)).item()))

