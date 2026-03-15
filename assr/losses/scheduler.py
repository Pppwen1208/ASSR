from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LossScheduler:
    """
    Smooth ramp for optional perceptual/adversarial terms:
    lambda(t) = lambda_max * sigma((t - t0) / tau), sigma(u)=1/(1+exp(-u))
    """

    lambda_perc_max: float = 0.0
    lambda_adv_max: float = 0.0
    t0: int = 0
    tau: int = 1

    def _sigma(self, step: int) -> float:
        u = (float(step) - float(self.t0)) / float(max(self.tau, 1))
        s = 1.0 / (1.0 + math.exp(-u))
        return max(0.0, float(s))

    def perc_weight(self, step: int, enabled: bool = True) -> float:
        if not enabled or self.lambda_perc_max <= 0:
            return 0.0
        return float(self.lambda_perc_max * self._sigma(step))

    def adv_weight(self, step: int, enabled: bool = True) -> float:
        if not enabled or self.lambda_adv_max <= 0:
            return 0.0
        return float(self.lambda_adv_max * self._sigma(step))

    def weights(self, step: int, use_perceptual: bool, use_gan: bool) -> tuple[float, float]:
        return self.perc_weight(step, enabled=use_perceptual), self.adv_weight(
            step, enabled=use_gan
        )

