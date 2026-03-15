from __future__ import annotations

import torch
import torch.nn as nn


class ResidualDenseBlock(nn.Module):
    def __init__(self, channels: int, growth_channels: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(
            channels + growth_channels * 2, growth_channels, 3, 1, 1
        )
        self.conv4 = nn.Conv2d(
            channels + growth_channels * 3, growth_channels, 3, 1, 1
        )
        self.conv5 = nn.Conv2d(channels + growth_channels * 4, channels, 3, 1, 1)
        self.act = nn.GELU()
        self.res_scale = 0.2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.act(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.act(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + x5 * self.res_scale


class RRDB(nn.Module):
    def __init__(self, channels: int, growth_channels: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(channels, growth_channels)
        self.res_scale = 0.2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return x + out * self.res_scale


class RRDBBackbone(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        num_blocks: int = 36,
        growth_channels: int = 32,
    ) -> None:
        super().__init__()
        self.conv_first = nn.Conv2d(channels, channels, 3, 1, 1)
        self.blocks = nn.ModuleList(
            [RRDB(channels, growth_channels) for _ in range(num_blocks)]
        )
        self.conv_body = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(
        self,
        x: torch.Tensor,
        modulation_blocks: dict[int, nn.Module] | None = None,
        modulation_kwargs: dict | None = None,
    ) -> torch.Tensor:
        feat = self.conv_first(x)
        body = feat
        modulation_kwargs = modulation_kwargs or {}
        if modulation_blocks is None:
            modulation_blocks = {}

        for idx, block in enumerate(self.blocks):
            body = block(body)
            mod = modulation_blocks.get(idx)
            if mod is not None:
                body = mod(body, **modulation_kwargs)

        body = self.conv_body(body)
        return body + feat

