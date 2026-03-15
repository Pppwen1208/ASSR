from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from assr.config.schema import DegradationConfig


@dataclass
class DegradationOutput:
    lr: torch.Tensor
    lr_ref: torch.Tensor
    scale: float
    kappa: float
    eta: float


def _anisotropic_gaussian_kernel(
    kernel_size: int,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    c, s = math.cos(theta), math.sin(theta)
    xr = c * xx + s * yy
    yr = -s * xx + c * yy
    kernel = torch.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2))
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    return kernel


def _conv_blur(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    c = x.shape[0]
    k = kernel.shape[0]
    w = kernel.view(1, 1, k, k).repeat(c, 1, 1, 1)
    y = F.conv2d(x.unsqueeze(0), w, padding=k // 2, groups=c)
    return y.squeeze(0)


def _usm(x: torch.Tensor, sigma: float, amount: float) -> torch.Tensor:
    k = 5
    device, dtype = x.device, x.dtype
    kernel = _anisotropic_gaussian_kernel(
        kernel_size=k,
        sigma_x=sigma,
        sigma_y=sigma,
        theta=0.0,
        device=device,
        dtype=dtype,
    )
    blur = _conv_blur(x, kernel)
    out = x + amount * (x - blur)
    return out.clamp(0.0, 1.0)


class ScaleAwareDegradation:
    def __init__(self, cfg: DegradationConfig) -> None:
        self.cfg = cfg

    def _sample_blur_params(self, scale: float) -> tuple[float, float, float]:
        sigma_base = self.cfg.sigma_min + (
            (self.cfg.sigma_max - self.cfg.sigma_min) * (scale - 1.5) / (6.0 - 1.5)
        )
        sigma_base = max(self.cfg.sigma_min, min(self.cfg.sigma_max, sigma_base))
        ratio = random.uniform(self.cfg.anisotropy_min, self.cfg.anisotropy_max)
        sigma_x = sigma_base
        sigma_y = sigma_base / ratio
        theta = random.uniform(0.0, math.pi)
        return sigma_x, sigma_y, theta

    def _anti_alias_resize(self, x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        return F.interpolate(
            x.unsqueeze(0),
            size=(out_h, out_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(0)

    def _poisson_gaussian_noise(self, x: torch.Tensor, scale: float) -> tuple[torch.Tensor, float]:
        shot = random.uniform(0.0, self.cfg.shot_noise_max) * (0.5 + 0.5 * scale / 6.0)
        read = random.uniform(0.0, self.cfg.read_noise_max) * (0.5 + 0.5 * scale / 6.0)
        std = torch.sqrt((x.clamp_min(0.0) * shot) + (read * read))
        noise = torch.randn_like(x) * std
        y = (x + noise).clamp(0.0, 1.0)
        eta = float((shot + read) * 0.5)
        return y, eta

    def _sar_speckle(self, x: torch.Tensor, scale: float) -> tuple[torch.Tensor, float]:
        speckle_std = 0.02 + 0.08 * (scale - 1.5) / (6.0 - 1.5)
        speckle = torch.randn_like(x) * speckle_std + 1.0
        y = (x * speckle).clamp(0.0, 1.0)
        y = torch.log1p(y)
        eta = float(speckle_std)
        return y, eta

    def _light_compression(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        if random.random() > self.cfg.compression_prob:
            return x, 0.0
        q = random.uniform(
            float(self.cfg.compression_quality_min), float(self.cfg.compression_quality_max)
        )
        # Differentiable approximation of compression via low-bit quantization.
        levels = int(max(16, round(2 + q / 3)))
        y = torch.round(x * levels) / levels
        y = y.clamp(0.0, 1.0)
        eta = float((100.0 - q) / 100.0)
        return y, eta

    def degrade_s2(self, hr: torch.Tensor, scale: float) -> DegradationOutput:
        x = _usm(hr, sigma=self.cfg.usm_sigma, amount=self.cfg.usm_amount)
        sigma_x, sigma_y, theta = self._sample_blur_params(scale)
        kernel = _anisotropic_gaussian_kernel(
            kernel_size=self.cfg.kernel_size,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            theta=theta,
            device=x.device,
            dtype=x.dtype,
        )
        x = _conv_blur(x, kernel)

        out_h = max(1, int(round(hr.shape[1] / scale)))
        out_w = max(1, int(round(hr.shape[2] / scale)))
        x = self._anti_alias_resize(x, out_h=out_h, out_w=out_w)
        # Eq.(2): deterministic scale-consistent LR reference before stochastic corruption.
        x_ref = x.clone()
        x, eta_noise = self._poisson_gaussian_noise(x, scale=scale)
        x, eta_comp = self._light_compression(x)

        kappa = 1.0 / max(scale, 1e-6)
        eta = min(1.0, eta_noise + eta_comp)
        return DegradationOutput(
            lr=x,
            lr_ref=x_ref,
            scale=scale,
            kappa=float(kappa),
            eta=float(eta),
        )

    def degrade_s1(self, s1: torch.Tensor, scale: float) -> DegradationOutput:
        sigma_x, sigma_y, theta = self._sample_blur_params(scale)
        kernel = _anisotropic_gaussian_kernel(
            kernel_size=self.cfg.kernel_size,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            theta=theta,
            device=s1.device,
            dtype=s1.dtype,
        )
        x = _conv_blur(s1, kernel)
        out_h = max(1, int(round(s1.shape[1] / scale)))
        out_w = max(1, int(round(s1.shape[2] / scale)))
        x = self._anti_alias_resize(x, out_h=out_h, out_w=out_w)
        x_ref = x.clone()
        x, eta = self._sar_speckle(x, scale=scale)
        return DegradationOutput(
            lr=x,
            lr_ref=x_ref,
            scale=scale,
            kappa=float(1.0 / scale),
            eta=float(eta),
        )
