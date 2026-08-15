"""Neural-network layers used by the bundled CTD checkpoint."""

import torch
import torch.nn as nn


def autopad(kernel_size, padding=None):
    """Return padding that preserves the spatial size for an odd kernel."""
    if padding is not None:
        return padding
    if not isinstance(kernel_size, int):
        raise TypeError("CTD convolution kernel size must be an integer")
    return kernel_size // 2


class Conv(nn.Module):
    """Convolution, batch normalisation and activation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        groups=1,
        act=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            autopad(kernel_size, padding),
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if act is True:
            self.act = nn.SiLU()
        elif act is False:
            self.act = nn.Identity()
        elif act == "leaky":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        else:
            raise ValueError(f"不支持的 CTD 激活函数: {act!r}")

    def forward(self, value):
        return self.act(self.bn(self.conv(value)))

    def forward_fuse(self, value):
        return self.act(self.conv(value))


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        groups=1,
        expansion=0.5,
        act=True,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1, act=act)
        self.cv2 = Conv(
            hidden_channels,
            out_channels,
            3,
            1,
            groups=groups,
            act=act,
        )
        self.add = shortcut and in_channels == out_channels

    def forward(self, value):
        transformed = self.cv2(self.cv1(value))
        return value + transformed if self.add else transformed


class C3(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        repeats=1,
        shortcut=True,
        groups=1,
        expansion=0.5,
        act=True,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1, act=act)
        self.cv2 = Conv(in_channels, hidden_channels, 1, 1, act=act)
        self.cv3 = Conv(2 * hidden_channels, out_channels, 1, act=act)
        self.m = nn.Sequential(
            *(
                Bottleneck(
                    hidden_channels,
                    hidden_channels,
                    shortcut,
                    groups,
                    expansion=1.0,
                    act=act,
                )
                for _ in range(repeats)
            )
        )

    def forward(self, value):
        return self.cv3(torch.cat((self.m(self.cv1(value)), self.cv2(value)), dim=1))


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.cv2 = Conv(hidden_channels * 4, out_channels, 1, 1)
        self.pool = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

    def forward(self, value):
        value = self.cv1(value)
        pooled_once = self.pool(value)
        pooled_twice = self.pool(pooled_once)
        return self.cv2(
            torch.cat(
                (value, pooled_once, pooled_twice, self.pool(pooled_twice)),
                dim=1,
            )
        )


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.dimension = dimension

    def forward(self, values):
        return torch.cat(values, self.dimension)
