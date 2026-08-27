"""
YSGYolo 后端

移植自 BallonsTranslator 项目
只保留模型推理核心逻辑
"""

import logging
import math
import os
from typing import List, Tuple, Optional

import cv2
import numpy as np

from ..base import BaseTextDetector
from ..data_types import TextLine
from src.shared.path_helpers import resource_path
from src.shared.user_logging import user_log

logger = logging.getLogger("YoloBackend")

# 默认配置
DEFAULT_MODEL_DIR = 'models/yolo'
DEFAULT_MODEL_NAME = 'ysgyolo_1.2_OS1.0.pt'
DEFAULT_CONF_THRESH = 0.3
DEFAULT_IOU_THRESH = 0.5
# 显式固定历史推理尺寸，避免 Ultralytics 默认值变化影响检测结果。
DEFAULT_DETECT_SIZE = 640
DEFAULT_MASK_DILATE = 2

# 默认标签
DEFAULT_LABELS = {
    'balloon': True,
    'qipao': True,
    'shuqing': True,
    'changfangtiao': True,
    'hengxie': True,
    'other': False
}


class YoloBackend(BaseTextDetector):
    """
    YSGYolo 检测后端
    
    100% 遵循 BallonsTranslator 的实现
    """
    detector_id: str = 'yolo'
    
    def __init__(self,
                 model_dir: str = None,
                 device: str = 'cuda',
                 conf_thresh: float = DEFAULT_CONF_THRESH,
                 iou_thresh: float = DEFAULT_IOU_THRESH,
                 detect_size: int = DEFAULT_DETECT_SIZE,
                 mask_dilate_size: int = DEFAULT_MASK_DILATE,
                 labels: dict = None):
        """
        初始化 YSGYolo 检测器
        
        Args:
            model_dir: 模型目录
            device: 设备
            conf_thresh: 置信度阈值
            iou_thresh: IoU阈值
            detect_size: 检测尺寸
            mask_dilate_size: 掩码膨胀大小
            labels: 标签配置
        """
        self.model_dir = resource_path(DEFAULT_MODEL_DIR) if model_dir is None else model_dir
        for label, value in (("置信度", conf_thresh), ("IoU", iou_thresh)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0 <= float(value) <= 1
            ):
                raise ValueError(f"YSGYolo {label}阈值必须是 0 到 1 之间的数字")
        if isinstance(detect_size, bool) or not isinstance(detect_size, int) or detect_size <= 0:
            raise ValueError("YSGYolo 检测尺寸必须是正整数")
        if (
            isinstance(mask_dilate_size, bool)
            or not isinstance(mask_dilate_size, int)
            or mask_dilate_size < 0
        ):
            raise ValueError("YSGYolo 掩码膨胀尺寸必须是非负整数")
        if labels is not None and (
            not isinstance(labels, dict)
            or any(
                not isinstance(key, str) or not isinstance(value, bool)
                for key, value in labels.items()
            )
        ):
            raise TypeError("YSGYolo 标签配置必须是字符串到布尔值的映射")
        self.conf_thresh = float(conf_thresh)
        self.iou_thresh = float(iou_thresh)
        self.detect_size = detect_size
        self.mask_dilate_size = mask_dilate_size
        self.labels = DEFAULT_LABELS.copy() if labels is None else labels.copy()
        self.model_path = None
        
        super().__init__(device=device)
    
    def _load_model(self):
        """加载 YSGYolo 模型"""
        # 查找模型路径
        model_path = os.path.join(self.model_dir, DEFAULT_MODEL_NAME)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YSGYolo 模型文件未找到: {model_path}\n"
                f"请从 https://huggingface.co/YSGforMTL/YSGYoloDetector 下载模型"
            )
        
        # 根据模型文件名选择加载器
        if 'rtdetr' in os.path.basename(model_path):
            from ultralytics import RTDETR as MODEL
        else:
            from ultralytics import YOLO as MODEL
        
        self.model = MODEL(model_path).to(device=self.device)
        self.model_path = model_path
        
        logger.debug(f"YSGYolo 检测器初始化完成 - 设备: {self.device}, 模型: {model_path}")
        user_log("system", f"YSGYolo 文本检测模型已加载｜设备 {self.device.upper()}")
    
    def get_valid_labels(self) -> List[str]:
        """获取有效标签"""
        return [k for k, v in self.labels.items() if v]
    
    def _detect_raw(
        self,
        image: np.ndarray,
        conf_thresh: float = None,
        iou_thresh: float = None,
    ) -> Tuple[List[TextLine], Optional[np.ndarray]]:
        """
        执行原始检测
        
        Args:
            image: OpenCV BGR 格式图像
            
        Returns:
            Tuple[List[TextLine], Optional[np.ndarray]]
        """
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        im_h, im_w = image.shape[:2]
        conf_thresh = self.conf_thresh if conf_thresh is None else conf_thresh
        iou_thresh = self.iou_thresh if iou_thresh is None else iou_thresh
        for label, value in (("置信度", conf_thresh), ("IoU", iou_thresh)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0 <= value <= 1
            ):
                raise ValueError(f"YSGYolo {label}阈值必须是 0 到 1 之间的数字")
        
        # YOLO 推理
        result = self.model.predict(
            source=image,
            save=False,
            show=False,
            verbose=False,
            conf=conf_thresh,
            iou=iou_thresh,
            imgsz=self.detect_size,
            agnostic_nms=True
        )[0]
        
        valid_labels = set(self.get_valid_labels())
        valid_ids = [idx for idx, name in result.names.items() if name in valid_labels]
        
        mask = np.zeros_like(image[..., 0])
        if not valid_ids:
            return [], mask
        
        textlines = []
        
        # 处理标准框
        dets = result.boxes
        if dets is not None and len(dets.cls) > 0:
            for i in range(len(dets.cls)):
                cls_idx = int(dets.cls[i])
                if cls_idx in valid_ids:
                    xyxy = dets.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy.astype(int)
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                    
                    # 创建四边形
                    pts = np.array([
                        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                    ], dtype=np.int32)
                    
                    conf = float(dets.conf[i].cpu().numpy())
                    textlines.append(TextLine(pts=pts, confidence=conf))
        
        # 处理旋转框
        dets = result.obb
        if dets is not None and len(dets.cls) > 0:
            for i in range(len(dets.cls)):
                cls_idx = int(dets.cls[i])
                if cls_idx in valid_ids:
                    pts = dets.xyxyxyxy[i].cpu().numpy().astype(np.int32)
                    cv2.fillPoly(mask, [pts], 255)
                    
                    conf = float(dets.conf[i].cpu().numpy())
                    textlines.append(TextLine(pts=pts, confidence=conf))
        
        # 掩码膨胀
        if self.mask_dilate_size > 0:
            ksize = self.mask_dilate_size
            element = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (2 * ksize + 1, 2 * ksize + 1), 
                (ksize, ksize)
            )
            mask = cv2.dilate(mask, element)
        
        logger.debug(f"YSGYolo 检测到 {len(textlines)} 个文本区域")
        return textlines, mask
