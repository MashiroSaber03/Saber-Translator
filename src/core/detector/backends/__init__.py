"""
检测器后端

包含各个模型的具体实现
"""

from .ctd_backend import CTDBackend
from .yolo_backend import YoloBackend
from .default_backend import DefaultBackend

__all__ = ['CTDBackend', 'YoloBackend', 'DefaultBackend']
