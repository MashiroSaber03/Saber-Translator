"""
TextDetection 模型 (ResNet34 + DBNet)
"""

import torch
import torch.nn as nn
from torchvision.models import resnet34

from . import DBHead


class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, stride=1, planes=256):
        super(double_conv, self).__init__()
        self.planes = planes
        self.down = nn.AvgPool2d(2, stride=2) if stride > 1 else None
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.down is not None:
            x = self.down(x)
        x = self.conv(x)
        return x


class double_conv_up(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, planes=256):
        super(double_conv_up, self).__init__()
        self.planes = planes
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class TextDetection(nn.Module):
    """
    ResNet34 + DBNet 文本检测模型
    
    输入: (N, 3, H, W) 归一化到 [-1, 1]
    输出: (db, mask)
        - db: DBNet 输出，用于文本框提取
        - mask: 文本掩码
    """
    
    def __init__(self, pretrained=False):
        super(TextDetection, self).__init__()
        self.backbone = resnet34(pretrained=pretrained)

        self.conv_db = DBHead.DBHead(64, 0)
        self.conv_mask = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.down_conv1 = double_conv(0, 512, 512, 2)
        self.down_conv2 = double_conv(0, 512, 512, 2)
        self.down_conv3 = double_conv(0, 512, 512, 2)

        self.upconv1 = double_conv_up(0, 512, 256)
        self.upconv2 = double_conv_up(256, 512, 256)
        self.upconv3 = double_conv_up(256, 512, 256)
        self.upconv4 = double_conv_up(256, 512, 256, planes=128)
        self.upconv5 = double_conv_up(256, 256, 128, planes=64)
        self.upconv6 = double_conv_up(128, 128, 64, planes=32)
        self.upconv7 = double_conv_up(64, 64, 64, planes=16)

    def forward(self, x):
        # Backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)  # 64@H/4

        h4 = self.backbone.layer1(x)   # 64@H/4
        h8 = self.backbone.layer2(h4)  # 128@H/8
        h16 = self.backbone.layer3(h8)  # 256@H/16
        h32 = self.backbone.layer4(h16) # 512@H/32
        h64 = self.down_conv1(h32)      # 512@H/64
        h128 = self.down_conv2(h64)     # 512@H/128
        h256 = self.down_conv3(h128)    # 512@H/256

        # Decoder
        up256 = self.upconv1(h256)
        up128 = self.upconv2(torch.cat([up256, h128], dim=1))
        up64 = self.upconv3(torch.cat([up128, h64], dim=1))
        up32 = self.upconv4(torch.cat([up64, h32], dim=1))
        up16 = self.upconv5(torch.cat([up32, h16], dim=1))
        up8 = self.upconv6(torch.cat([up16, h8], dim=1))
        up4 = self.upconv7(torch.cat([up8, h4], dim=1))

        return self.conv_db(up8), self.conv_mask(up4)
