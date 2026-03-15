from __future__ import annotations

import torch
import torch.nn as nn


def snconv(
    in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1
) -> nn.Module:
    return nn.utils.spectral_norm(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
    )


class SNUNetDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels

        self.enc1 = nn.Sequential(snconv(in_channels, c), nn.LeakyReLU(0.2, True))
        self.enc2 = nn.Sequential(snconv(c, c * 2, stride=2), nn.LeakyReLU(0.2, True))
        self.enc3 = nn.Sequential(
            snconv(c * 2, c * 4, stride=2), nn.LeakyReLU(0.2, True)
        )
        self.bottleneck = nn.Sequential(
            snconv(c * 4, c * 4), nn.LeakyReLU(0.2, True), snconv(c * 4, c * 4)
        )

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1)
        self.dec2 = nn.Sequential(snconv(c * 4, c * 2), nn.LeakyReLU(0.2, True))
        self.up1 = nn.ConvTranspose2d(c * 2, c, 4, 2, 1)
        self.dec1 = nn.Sequential(snconv(c * 2, c), nn.LeakyReLU(0.2, True))

        self.out_head = snconv(c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out_head(d1)

