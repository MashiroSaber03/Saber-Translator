"""Minimal YOLOv5 graph loader for the bundled CTD checkpoint."""

import torch
from torch import nn

from .common import C3, Concat, Conv, SPPF
from ..utils.yolov5_utils import (
    check_anchor_order,
    fuse_conv_and_bn,
    initialize_weights,
    make_divisible,
)


class Detect(nn.Module):
    stride = None

    def __init__(self, nc=80, anchors=(), ch=()):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        self.anchor_grid = [torch.zeros(1)] * self.nl
        self.register_buffer(
            "anchors",
            torch.tensor(anchors).float().view(self.nl, -1, 2),
        )
        self.m = nn.ModuleList(nn.Conv2d(value, self.no * self.na, 1) for value in ch)

    def forward(self, values):
        predictions = []
        for index in range(self.nl):
            values[index] = self.m[index](values[index])
            batch_size, _, height, width = values[index].shape
            values[index] = (
                values[index]
                .view(batch_size, self.na, self.no, height, width)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            if self.training:
                continue
            if self.grid[index].shape[2:4] != values[index].shape[2:4]:
                self.grid[index], self.anchor_grid[index] = self._make_grid(
                    width,
                    height,
                    index,
                )

            prediction = values[index].sigmoid()
            prediction[..., 0:2] = (
                prediction[..., 0:2] * 2 - 0.5 + self.grid[index]
            ) * self.stride[index]
            prediction[..., 2:4] = (
                prediction[..., 2:4] * 2
            ) ** 2 * self.anchor_grid[index]
            predictions.append(prediction.view(batch_size, -1, self.no))

        if self.training:
            return values
        return torch.cat(predictions, dim=1), values

    def _make_grid(self, width, height, index):
        device = self.anchors[index].device
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        grid = torch.stack((x_grid, y_grid), dim=2).expand(
            1,
            self.na,
            height,
            width,
            2,
        ).float()
        anchor_grid = (self.anchors[index].clone() * self.stride[index]).view(
            1,
            self.na,
            1,
            1,
            2,
        ).expand(1, self.na, height, width, 2).float()
        return grid, anchor_grid


_MODULE_TYPES = {
    "Conv": Conv,
    "C3": C3,
    "SPPF": SPPF,
    "Concat": Concat,
    "Detect": Detect,
    "nn.Upsample": nn.Upsample,
}


def _resolve_argument(value, *, nc, anchors):
    if not isinstance(value, str):
        return value
    if value == "None":
        return None
    if value == "nc":
        return nc
    if value == "anchors":
        return anchors
    if value == "nearest":
        return value
    raise ValueError(f"CTD checkpoint 使用了不支持的参数: {value!r}")


class Model(nn.Module):
    def __init__(self, config, channels=3):
        super().__init__()
        if not isinstance(config, dict):
            raise TypeError("CTD checkpoint 中的 YOLO 配置必须是对象")
        self.model, self.save = parse_model(config, channels=[channels])
        self.out_indices = None
        detection = self.model[-1]
        if not isinstance(detection, Detect):
            raise ValueError("CTD checkpoint 的最后一层必须是 Detect")
        sample_size = 256
        detection.stride = torch.tensor(
            [
                sample_size / output.shape[-2]
                for output in self.forward(
                    torch.zeros(1, channels, sample_size, sample_size)
                )
            ]
        )
        detection.anchors /= detection.stride.view(-1, 1, 1)
        check_anchor_order(detection)
        self.stride = detection.stride
        initialize_weights(self)

    def forward(self, value, detect=False):
        return self._forward_once(value, detect=detect)

    def _forward_once(self, value, detect=False):
        saved_outputs = []
        selected_outputs = []
        for layer in self.model:
            if layer.f != -1:
                value = (
                    saved_outputs[layer.f]
                    if isinstance(layer.f, int)
                    else [
                        value if source == -1 else saved_outputs[source]
                        for source in layer.f
                    ]
                )
            value = layer(value)
            saved_outputs.append(value if layer.i in self.save else None)
            if self.out_indices is not None and layer.i in self.out_indices:
                selected_outputs.append(value)

        if self.out_indices is None:
            return value
        if detect:
            return value, selected_outputs
        return selected_outputs

    def fuse(self):
        for layer in self.model.modules():
            if isinstance(layer, Conv) and hasattr(layer, "bn"):
                layer.conv = fuse_conv_and_bn(layer.conv, layer.bn)
                delattr(layer, "bn")
                layer.forward = layer.forward_fuse
        return self

    def _apply(self, function):
        super()._apply(function)
        detection = self.model[-1]
        detection.stride = function(detection.stride)
        detection.grid = [function(value) for value in detection.grid]
        detection.anchor_grid = [function(value) for value in detection.anchor_grid]
        return self


def parse_model(config, channels):
    required_keys = {
        "anchors",
        "nc",
        "depth_multiple",
        "width_multiple",
        "backbone",
        "head",
    }
    allowed_keys = required_keys | {"ch"}
    if set(config) != allowed_keys or config["ch"] != 3:
        raise ValueError("CTD checkpoint 的 YOLO 配置字段无效")

    anchors = config["anchors"]
    class_count = config["nc"]
    depth_multiple = config["depth_multiple"]
    width_multiple = config["width_multiple"]
    anchor_count = len(anchors[0]) // 2
    output_channels = anchor_count * (class_count + 5)

    layers = []
    saved_indices = []
    output_channel_count = channels[-1]
    definitions = config["backbone"] + config["head"]
    for index, definition in enumerate(definitions):
        if not isinstance(definition, list) or len(definition) != 4:
            raise ValueError(f"CTD checkpoint 的第 {index} 层配置无效")
        sources, repeats, module_name, arguments = definition
        if module_name not in _MODULE_TYPES:
            raise ValueError(f"CTD checkpoint 使用了不支持的层: {module_name!r}")
        if not isinstance(arguments, list):
            raise ValueError(f"CTD checkpoint 的第 {index} 层参数必须是列表")
        module_type = _MODULE_TYPES[module_name]
        arguments = [
            _resolve_argument(value, nc=class_count, anchors=anchors)
            for value in arguments
        ]
        repeats = max(round(repeats * depth_multiple), 1) if repeats > 1 else repeats

        if module_type in {Conv, C3, SPPF}:
            in_channels = channels[sources]
            output_channel_count = arguments[0]
            if output_channel_count != output_channels:
                output_channel_count = make_divisible(
                    output_channel_count * width_multiple,
                    8,
                )
            arguments = [in_channels, output_channel_count, *arguments[1:]]
            if module_type is C3:
                arguments.insert(2, repeats)
                repeats = 1
        elif module_type is Concat:
            output_channel_count = sum(channels[source] for source in sources)
        elif module_type is Detect:
            arguments.append([channels[source] for source in sources])
        else:
            output_channel_count = channels[sources]

        layer = (
            nn.Sequential(*(module_type(*arguments) for _ in range(repeats)))
            if repeats > 1
            else module_type(*arguments)
        )
        layer.i = index
        layer.f = sources
        saved_indices.extend(
            source % index
            for source in ([sources] if isinstance(sources, int) else sources)
            if source != -1
        )
        layers.append(layer)
        if index == 0:
            channels = []
        channels.append(output_channel_count)

    return nn.Sequential(*layers), sorted(saved_indices)


@torch.no_grad()
def load_yolov5_ckpt(weights, map_location="cpu"):
    if not isinstance(weights, dict):
        raise TypeError("CTD checkpoint 的检测器数据必须是对象")
    if set(weights) != {"cfg", "weights"}:
        raise ValueError("CTD checkpoint 的检测器数据字段无效")
    model = Model(weights["cfg"])
    model.load_state_dict(weights["weights"], strict=True)
    model = model.float()
    model.fuse()
    model.eval()
    model.out_indices = [1, 3, 5, 7, 9]
    return model.to(map_location)
