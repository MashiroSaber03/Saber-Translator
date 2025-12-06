"""
检测器后端

包含各个模型的具体实现
"""

from .ctd_backend import CTDBackend
from .yolo_backend import YoloBackend
from .yolov5_backend import YoloV5Backend
from .default_backend import DefaultBackend

__all__ = ['CTDBackend', 'YoloBackend', 'YoloV5Backend', 'DefaultBackend']
