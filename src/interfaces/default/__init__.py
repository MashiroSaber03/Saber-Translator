"""
Default (DBNet ResNet34) 检测器模块
"""

from . import DBHead
from .DBNet_resnet34 import TextDetection
from .imgproc import resize_aspect_ratio, adjustResultCoordinates

__all__ = ['DBHead', 'TextDetection', 'resize_aspect_ratio', 'adjustResultCoordinates']
