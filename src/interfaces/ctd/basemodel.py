import cv2
import torch
import torch.nn as nn

from .yolov5.common import C3
from .yolov5.yolo import load_yolov5_ckpt


class DoubleConvUpC3(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            C3(in_channels + mid_channels, mid_channels, act="leaky"),
            nn.ConvTranspose2d(
                mid_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, value):
        return self.conv(value)


class DoubleConvC3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.AvgPool2d(2, stride=2)
        self.conv = C3(in_channels, out_channels, act="leaky")

    def forward(self, value):
        return self.conv(self.down(value))


class UnetHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_conv1 = DoubleConvC3(512, 512)
        self.upconv0 = DoubleConvUpC3(0, 512, 256)
        self.upconv2 = DoubleConvUpC3(256, 512, 256)
        self.upconv3 = DoubleConvUpC3(0, 512, 256)
        self.upconv4 = DoubleConvUpC3(128, 256, 128)
        self.upconv5 = DoubleConvUpC3(64, 128, 64)
        self.upconv6 = nn.Sequential(
            nn.ConvTranspose2d(
                64,
                1,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, f160, f80, f40, f20, f3):
        d10 = self.down_conv1(f3)
        u20 = self.upconv0(d10)
        u40 = self.upconv2(torch.cat([f20, u20], dim=1))
        u80 = self.upconv3(torch.cat([f40, u40], dim=1))
        u160 = self.upconv4(torch.cat([f80, u80], dim=1))
        u320 = self.upconv5(torch.cat([f160, u160], dim=1))
        return self.upconv6(u320), (f80, f40, u40)


class DBHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upconv3 = DoubleConvUpC3(0, 512, 256)
        self.upconv4 = DoubleConvUpC3(128, 256, 128)
        self.conv = nn.Sequential(
            nn.Conv2d(128, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.binarize = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, 2),
        )
        self.thresh = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels // 4,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, 2),
            nn.Sigmoid(),
        )

    def forward(self, f80, f40, u40):
        u80 = self.upconv3(torch.cat([f40, u40], dim=1))
        value = self.upconv4(torch.cat([f80, u80], dim=1))
        value = self.conv(value)
        threshold_maps = self.thresh(value)
        shrink_maps = torch.sigmoid(self.binarize(value))
        return torch.cat((shrink_maps, threshold_maps), dim=1)


class TextDetBase(nn.Module):
    def __init__(self, model_path, device="cpu", half=False):
        super().__init__()
        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
        if not isinstance(checkpoint, dict) or set(checkpoint) != {
            "blk_det",
            "text_seg",
            "text_det",
        }:
            raise ValueError("CTD checkpoint fields are invalid")
        self.blk_det = load_yolov5_ckpt(
            checkpoint["blk_det"],
            map_location=device,
        )
        self.text_seg = UnetHead()
        self.text_seg.load_state_dict(checkpoint["text_seg"], strict=True)
        self.text_det = DBHead(64)
        self.text_det.load_state_dict(checkpoint["text_det"], strict=True)
        if half:
            self.blk_det.half()
            self.text_seg.half()
            self.text_det.half()
        else:
            self.blk_det.to(device)
            self.text_seg.to(device)
            self.text_det.to(device)
        self.eval()

    def forward(self, value):
        blocks, features = self.blk_det(value, detect=True)
        mask, text_features = self.text_seg(*features)
        lines = self.text_det(*text_features)
        return blocks[0], mask, lines


class TextDetBaseDNN:
    def __init__(self, input_size, model_path):
        self.input_size = input_size
        self.model = cv2.dnn.readNetFromONNX(model_path)
        self.output_names = self.model.getUnconnectedOutLayersNames()

    def __call__(self, value):
        blob = cv2.dnn.blobFromImage(
            value,
            scalefactor=1 / 255.0,
            size=(self.input_size, self.input_size),
        )
        self.model.setInput(blob)
        return self.model.forward(self.output_names)
