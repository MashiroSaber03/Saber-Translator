"""Detector configuration shared by plugin validation and core execution."""

from collections.abc import Mapping
import math
from typing import Any


DETECTOR_CONFIG_FIELDS = frozenset({
    "detector_type", "expand_ratio", "expand_top", "expand_bottom",
    "expand_left", "expand_right", "enable_aux_yolo_detection",
    "aux_yolo_conf_threshold", "aux_yolo_overlap_threshold",
    "enable_saber_yolo_refine", "saber_yolo_refine_overlap_threshold",
    "min_text_block_area_percent",
})


def validate_detector_config(config: Mapping[str, Any]) -> None:
    missing = DETECTOR_CONFIG_FIELDS - config.keys()
    unknown = config.keys() - DETECTOR_CONFIG_FIELDS
    if missing or unknown:
        raise ValueError(
            "detectorConfig 字段不符合要求："
            f"缺少字段={sorted(missing)}，不支持的字段={sorted(unknown)}"
        )
    if config["detector_type"] not in {"default", "ctd", "yolo"}:
        raise ValueError("detector type is invalid")
    for field in ("enable_aux_yolo_detection", "enable_saber_yolo_refine"):
        if not isinstance(config[field], bool):
            raise ValueError(f"detector configuration {field} must be boolean")
    for field in DETECTOR_CONFIG_FIELDS - {
        "detector_type", "enable_aux_yolo_detection", "enable_saber_yolo_refine",
    }:
        value = config[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"detector configuration {field} must be finite")
    for field in (
        "aux_yolo_conf_threshold", "aux_yolo_overlap_threshold",
        "saber_yolo_refine_overlap_threshold",
    ):
        if not 0 <= config[field] <= 1:
            raise ValueError(f"detector configuration {field} must be from 0 to 1")
    if config["min_text_block_area_percent"] < 0:
        raise ValueError("minimum text block area cannot be negative")
