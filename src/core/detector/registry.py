"""
检测器注册表和统一调度接口

管理所有检测器后端的单例和调用
"""

import logging
from typing import Dict, Type, Optional, Literal

from PIL import Image

from .base import BaseTextDetector
from .data_types import DetectionResult

logger = logging.getLogger("DetectorRegistry")

# 检测器类型常量
DETECTOR_CTD = 'ctd'
DETECTOR_YOLO = 'yolo'
DETECTOR_YOLOV5 = 'yolov5'
DETECTOR_DEFAULT = 'default'  # DBNet ResNet34 (detect-20241225.ckpt)

DetectorType = Literal['default', 'ctd', 'yolo', 'yolov5']

# 支持的检测器描述
SUPPORTED_DETECTORS = {
    DETECTOR_DEFAULT: 'Default (DBNet ResNet34)',
    DETECTOR_CTD: 'CTD (Comic Text Detector)',
    DETECTOR_YOLO: 'YSGYolo',
    DETECTOR_YOLOV5: 'YOLOv5 (旧版)'
}

# 默认检测器
DEFAULT_DETECTOR = DETECTOR_DEFAULT

# 检测器注册表
_detector_registry: Dict[str, Type[BaseTextDetector]] = {}
_detector_instances: Dict[str, BaseTextDetector] = {}


def register_detector(name: str, detector_class: Type[BaseTextDetector]):
    """
    注册检测器类
    
    Args:
        name: 检测器名称
        detector_class: 检测器类（必须继承 BaseTextDetector）
    """
    if not issubclass(detector_class, BaseTextDetector):
        raise TypeError(f"{detector_class} 必须继承 BaseTextDetector")
    _detector_registry[name] = detector_class
    logger.info(f"已注册检测器: {name}")


def _lazy_register_builtin():
    """延迟注册内置检测器"""
    if _detector_registry:
        return
    
    from .backends.ctd_backend import CTDBackend
    from .backends.yolo_backend import YoloBackend
    from .backends.yolov5_backend import YoloV5Backend
    from .backends.default_backend import DefaultBackend
    
    register_detector(DETECTOR_CTD, CTDBackend)
    register_detector(DETECTOR_YOLO, YoloBackend)
    register_detector(DETECTOR_YOLOV5, YoloV5Backend)
    register_detector(DETECTOR_DEFAULT, DefaultBackend)


def get_detector(
    detector_type: DetectorType = None,
    force_reload: bool = False,
    **kwargs
) -> BaseTextDetector:
    """
    获取检测器实例（单例模式）
    
    Args:
        detector_type: 检测器类型，默认使用 CTD
        force_reload: 是否强制重新加载
        **kwargs: 传递给检测器构造函数的参数
        
    Returns:
        BaseTextDetector: 检测器实例
    """
    _lazy_register_builtin()
    
    if detector_type is None:
        detector_type = DEFAULT_DETECTOR
    
    if detector_type not in _detector_registry:
        raise ValueError(
            f"未知的检测器类型: {detector_type}\n"
            f"支持的类型: {list(_detector_registry.keys())}"
        )
    
    # 检查是否需要创建新实例
    if force_reload or detector_type not in _detector_instances:
        logger.info(f"创建检测器实例: {detector_type}")
        detector_class = _detector_registry[detector_type]
        _detector_instances[detector_type] = detector_class(**kwargs)
    
    return _detector_instances[detector_type]


def reset_detector(detector_type: DetectorType = None):
    """
    重置检测器实例
    
    Args:
        detector_type: 要重置的检测器类型，None 表示重置所有
    """
    global _detector_instances
    
    if detector_type is None:
        _detector_instances = {}
        logger.info("所有检测器已重置")
    elif detector_type in _detector_instances:
        del _detector_instances[detector_type]
        logger.info(f"检测器 {detector_type} 已重置")


def detect(
    image: Image.Image,
    detector_type: DetectorType = None,
    merge_lines: bool = None,
    edge_ratio_threshold: float = 0.0,
    expand_ratio: float = 0,
    expand_top: float = 0,
    expand_bottom: float = 0,
    expand_left: float = 0,
    expand_right: float = 0,
    enable_large_image: bool = None,
    **kwargs
) -> DetectionResult:
    """
    统一检测入口函数
    
    Args:
        image: PIL 图像
        detector_type: 检测器类型
        merge_lines: 是否合并文本行（None 时使用检测器默认设置）
        edge_ratio_threshold: 边缘距离比例阈值
        expand_ratio: 整体扩展比例 (%)
        expand_top/bottom/left/right: 各边额外扩展 (%)
        enable_large_image: 是否启用超长图片切割（None 时使用全局配置）
        **kwargs: 其他参数
        
    Returns:
        DetectionResult: 检测结果
    """
    from src.shared import constants
    
    detector = get_detector(detector_type)
    
    # 判断是否启用大图检测
    if enable_large_image is None:
        enable_large_image = getattr(constants, 'LARGE_IMAGE_ENABLED', True)
    
    if enable_large_image:
        # 使用大图检测包装器
        try:
            from src.core.large_image_detection import LargeImageDetectorWrapper
            
            target_size = getattr(constants, 'LARGE_IMAGE_TARGET_SIZE', 1536)
            
            wrapper = LargeImageDetectorWrapper(
                detector=detector,
                target_size=target_size
            )
            
            return wrapper.detect(
                image,
                merge_lines=merge_lines,
                edge_ratio_threshold=edge_ratio_threshold,
                expand_ratio=expand_ratio,
                expand_top=expand_top,
                expand_bottom=expand_bottom,
                expand_left=expand_left,
                expand_right=expand_right,
                **kwargs
            )
        except ImportError as e:
            logger.warning(f"大图检测模块导入失败，回退到普通检测: {e}")
        except Exception as e:
            logger.warning(f"大图检测失败，回退到普通检测: {e}")
    
    # 普通检测
    return detector.detect(
        image,
        merge_lines=merge_lines,
        edge_ratio_threshold=edge_ratio_threshold,
        expand_ratio=expand_ratio,
        expand_top=expand_top,
        expand_bottom=expand_bottom,
        expand_left=expand_left,
        expand_right=expand_right,
        **kwargs
    )


def detect_to_legacy_format(
    image: Image.Image,
    detector_type: DetectorType = None,
    **kwargs
) -> dict:
    """
    检测并返回旧格式结果
    
    向后兼容接口
    
    Returns:
        dict: {'coords': [...], 'polygons': [...], 'angles': [...]}
    """
    result = detect(image, detector_type=detector_type, **kwargs)
    return result.to_legacy_format()
