"""DBNet inference head used by the bundled default detector."""

import torch
from torch import nn


class DBHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        inner_channels = in_channels // 4
        self.binarize = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, 3, padding=1),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels, inner_channels, 4, 2, 1),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels, 1, 4, 2, 1),
        )
        self.thresh = nn.Sequential(
            nn.Conv2d(
                in_channels,
                inner_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels, inner_channels, 4, 2, 1),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, value):
        return torch.cat(
            (self.binarize(value), self.thresh(value)),
            dim=1,
        )
