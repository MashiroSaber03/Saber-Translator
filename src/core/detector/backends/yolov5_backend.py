"""
YOLOv5 后端 (旧版本)

使用 torch.hub 加载本地 YOLOv5 仓库
"""

import os
import logging
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch

from ..base import BaseTextDetector
from ..data_types import TextLine
from src.shared.path_helpers import resource_path

logger = logging.getLogger("YoloV5Backend")

# 模块路径
MODULE_DIR = resource_path('src/interfaces/yolov5')
DEFAULT_REPO_DIR = os.path.join(MODULE_DIR, 'repo')
DEFAULT_WEIGHTS_DIR = os.path.join(MODULE_DIR, 'models')
DEFAULT_WEIGHTS = os.path.join(DEFAULT_WEIGHTS_DIR, 'best.pt')
DEFAULT_CONF_THRESH = 0.6


class YoloV5Backend(BaseTextDetector):
    """
    YOLOv5 检测后端（旧版本）
    
    使用 torch.hub.load 加载本地 YOLOv5 仓库
    """
    
    # YOLOv5 直接输出完整文本框，不需要合并
    requires_merge: bool = False
    
    def __init__(self,
                 weights_path: str = None,
                 repo_dir: str = None,
                 device: str = 'cuda',
                 conf_thresh: float = DEFAULT_CONF_THRESH,
                 **kwargs):
        """
        初始化 YOLOv5 检测器
        
        Args:
            weights_path: 权重文件路径
            repo_dir: YOLOv5 仓库目录
            device: 运行设备
            conf_thresh: 置信度阈值
        """
        self.weights_path = weights_path or DEFAULT_WEIGHTS
        self.repo_dir = repo_dir or DEFAULT_REPO_DIR
        self.conf_thresh = conf_thresh
        
        super().__init__(device=device, **kwargs)
    
    def _load_model(self, **kwargs):
        """加载 YOLOv5 模型"""
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"YOLOv5 权重文件未找到: {self.weights_path}\n"
                f"请将 best.pt 放到 {DEFAULT_WEIGHTS_DIR} 目录"
            )
        if not os.path.exists(self.repo_dir):
            raise FileNotFoundError(
                f"YOLOv5 仓库目录未找到: {self.repo_dir}\n"
                f"请将 ultralytics_yolov5_master 文件夹复制到 {MODULE_DIR} 并重命名为 repo"
            )
        
        logger.info(f"加载 YOLOv5 模型: {self.weights_path}")
        
        try:
            self.model = torch.hub.load(
                repo_or_dir=self.repo_dir,
                model='custom',
                path=self.weights_path,
                source='local',
                force_reload=False,
                trust_repo=True
            )
        except TypeError:
            logger.warning("当前 PyTorch Hub 版本不支持 trust_repo=True，使用旧版 API")
            self.model = torch.hub.load(
                repo_or_dir=self.repo_dir,
                model='custom',
                path=self.weights_path,
                source='local',
                force_reload=False
            )
        
        self.model.conf = self.conf_thresh
        logger.info(f"YOLOv5 检测器初始化完成 - 设备: {self.device}")
    
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
        
        # 执行推理
        results = self.model(image)
        predictions = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, class]
        
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        
        logger.info(f"YOLOv5 检测到 {len(boxes)} 个候选框 (阈值: {self.model.conf})")
        
        textlines = []
        mask = np.zeros((im_h, im_w), dtype=np.uint8)
        
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            
            # 创建四边形
            pts = np.array([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ], dtype=np.int32)
            
            textlines.append(TextLine(pts=pts, confidence=float(score)))
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        return textlines, mask
    
    def set_conf_threshold(self, conf_thresh: float):
        """设置置信度阈值"""
        self.conf_thresh = conf_thresh
        if self.model:
            self.model.conf = conf_thresh
            logger.info(f"YOLOv5 置信度阈值已更新: {conf_thresh}")
