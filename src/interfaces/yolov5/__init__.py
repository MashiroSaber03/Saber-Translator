"""
YOLOv5 文本检测器 (旧版本)
使用 torch.hub 加载本地 YOLOv5 模型

现已迁移到统一检测器框架: src/core/detector/
此模块保留以便向后兼容
"""

from src.core.detector.backends.yolov5_backend import YoloV5Backend as YoloV5TextDetector
from src.core.detector import get_detector, DETECTOR_YOLOV5


def get_yolov5_detector(**kwargs):
    """获取 YOLOv5 检测器单例（向后兼容）"""
    return get_detector(DETECTOR_YOLOV5, **kwargs)


def reset_yolov5_detector():
    """重置 YOLOv5 检测器（向后兼容）"""
    from src.core.detector.registry import reset_detector
    reset_detector(DETECTOR_YOLOV5)


__all__ = ['YoloV5TextDetector', 'get_yolov5_detector', 'reset_yolov5_detector']
