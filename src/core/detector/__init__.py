"""
统一文本检测器框架

提供:
- 统一数据结构: TextLine, TextBlock, DetectionResult
- 检测器基类: BaseTextDetector
- 统一调度接口: get_detector, detect
"""

from .data_types import TextLine, TextBlock, DetectionResult
from .base import BaseTextDetector
from .registry import (
    get_detector, detect, detect_to_legacy_format, register_detector, 
    DETECTOR_CTD, DETECTOR_YOLO, DETECTOR_YOLOV5, DETECTOR_DEFAULT
)

__all__ = [
    # 数据类型
    'TextLine',
    'TextBlock', 
    'DetectionResult',
    # 基类
    'BaseTextDetector',
    # 调度接口
    'get_detector',
    'detect',
    'detect_to_legacy_format',
    'register_detector',
    # 常量
    'DETECTOR_CTD',
    'DETECTOR_YOLO',
    'DETECTOR_YOLOV5',
    'DETECTOR_DEFAULT',
]
