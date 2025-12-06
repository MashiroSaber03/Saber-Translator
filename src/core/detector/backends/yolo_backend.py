"""
YSGYolo 后端

移植自 BallonsTranslator 项目
只保留模型推理核心逻辑
"""

import os
import os.path as osp
import logging
from typing import List, Tuple, Optional

import cv2
import numpy as np

from ..base import BaseTextDetector
from ..data_types import TextLine
from ..geometry import xywh_to_polygon
from src.shared.path_helpers import resource_path
from src.shared import constants

logger = logging.getLogger("YoloBackend")

# 默认配置
DEFAULT_MODEL_DIR = 'models/yolo'
DEFAULT_MODEL_NAME = 'ysgyolo_1.2_OS1.0.pt'
DEFAULT_CONF_THRESH = 0.3
DEFAULT_IOU_THRESH = 0.5
DEFAULT_DETECT_SIZE = 1024
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


def _update_ckpt_list(model_dir: str) -> List[str]:
    """更新可用模型列表"""
    ckpt_list = []
    if not osp.exists(model_dir):
        return ckpt_list
    for p in os.listdir(model_dir):
        if p.startswith('ysgyolo') or p.startswith('ultralyticsyolo'):
            ckpt_list.append(osp.join(model_dir, p).replace('\\', '/'))
    return ckpt_list


class YoloBackend(BaseTextDetector):
    """
    YSGYolo 检测后端
    
    100% 遵循 BallonsTranslator 的实现
    """
    
    def __init__(self,
                 model_dir: str = None,
                 device: str = 'cuda',
                 conf_thresh: float = DEFAULT_CONF_THRESH,
                 iou_thresh: float = DEFAULT_IOU_THRESH,
                 detect_size: int = DEFAULT_DETECT_SIZE,
                 mask_dilate_size: int = DEFAULT_MASK_DILATE,
                 labels: dict = None,
                 **kwargs):
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
        self.model_dir = model_dir or resource_path(DEFAULT_MODEL_DIR)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.detect_size = detect_size
        self.mask_dilate_size = mask_dilate_size
        self.labels = labels or DEFAULT_LABELS.copy()
        self.model_path = None
        
        super().__init__(device=device, **kwargs)
    
    def _load_model(self, **kwargs):
        """加载 YSGYolo 模型"""
        # 查找模型路径
        ckpt_list = _update_ckpt_list(self.model_dir)
        model_path = osp.join(self.model_dir, DEFAULT_MODEL_NAME)
        
        if not osp.exists(model_path):
            df_model_path = model_path
            for p in ckpt_list:
                if osp.exists(p):
                    df_model_path = p
                    break
            logger.warning(f'{model_path} 不存在，尝试使用 {df_model_path}')
            model_path = df_model_path
        
        if not osp.exists(model_path):
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
        
        logger.info(f"YSGYolo 检测器初始化完成 - 设备: {self.device}, 模型: {model_path}")
    
    def get_valid_labels(self) -> List[str]:
        """获取有效标签"""
        return [k for k, v in self.labels.items() if v]
    
    def _detect_raw(self, image: np.ndarray, **kwargs) -> Tuple[List[TextLine], Optional[np.ndarray]]:
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
        
        # YOLO 推理
        result = self.model.predict(
            source=image,
            save=False,
            show=False,
            verbose=False,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
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
        
        logger.info(f"YSGYolo 检测到 {len(textlines)} 个文本区域")
        return textlines, mask
    
    def set_valid_labels(self, labels: dict):
        """设置有效的标签类别"""
        self.labels = labels
        logger.info(f"有效标签已更新: {self.get_valid_labels()}")
