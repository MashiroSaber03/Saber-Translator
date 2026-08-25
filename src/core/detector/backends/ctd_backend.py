"""
CTD (Comic Text Detector) 后端

只保留模型推理核心逻辑
"""

import logging
import math
import os
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch

from ..base import BaseTextDetector
from ..data_types import TextLine
from src.shared.path_helpers import resource_path
from src.shared.user_logging import user_log

logger = logging.getLogger("CTDBackend")

# 默认配置
DEFAULT_MODEL_DIR = 'models/ctd'
DEFAULT_INPUT_SIZE = 1024
DEFAULT_TEXT_THRESHOLD = 0.3
DEFAULT_BOX_THRESHOLD = 0.7


class CTDBackend(BaseTextDetector):
    """
    CTD 检测后端
    """
    detector_id: str = 'ctd'
    
    def __init__(self, 
                 model_dir: str = None,
                 device: str = 'cuda',
                 input_size: int = DEFAULT_INPUT_SIZE,
                 half: bool = False,
                 text_threshold: float = DEFAULT_TEXT_THRESHOLD,
                 box_threshold: float = DEFAULT_BOX_THRESHOLD):
        """
        初始化 CTD 检测器
        
        Args:
            model_dir: 模型文件目录
            device: 设备
            input_size: 输入图像大小
            half: 是否使用半精度
            text_threshold: 文本像素阈值
            box_threshold: 文本框置信度阈值
        """
        self.model_dir = resource_path(DEFAULT_MODEL_DIR) if model_dir is None else model_dir
        if isinstance(input_size, bool) or not isinstance(input_size, int) or input_size <= 0:
            raise ValueError("CTD 输入尺寸必须是正整数")
        if not isinstance(half, bool):
            raise TypeError("CTD 半精度开关必须是布尔值")
        for label, value in (
            ("文本", text_threshold),
            ("文本框", box_threshold),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0 <= float(value) <= 1
            ):
                raise ValueError(f"CTD {label}阈值必须是 0 到 1 之间的数字")
        self.input_size = (input_size, input_size)
        self.half = half
        self.text_threshold = float(text_threshold)
        self.box_threshold = float(box_threshold)
        self.backend = None
        self.seg_rep = None
        
        super().__init__(device=device)
    
    def _load_model(self):
        """加载 CTD 模型"""
        # 延迟导入，避免循环依赖
        from src.interfaces.ctd.utils.db_utils import SegDetectorRepresenter
        from src.interfaces.ctd.basemodel import TextDetBase, TextDetBaseDNN
        
        self.seg_rep = SegDetectorRepresenter(thresh=self.text_threshold)
        
        if self.device == 'cuda' or self.device == 'mps':
            model_path = os.path.join(self.model_dir, 'comictextdetector.pt')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件未找到: {model_path}")
            self.model = TextDetBase(
                model_path,
                device=self.device,
                half=self.half,
            )
            self.model.to(self.device)
            self.backend = 'torch'
            logger.debug(f"加载 PyTorch 模型: {model_path}")
        else:
            model_path = os.path.join(self.model_dir, 'comictextdetector.pt.onnx')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件未找到: {model_path}")
            self.model = TextDetBaseDNN(self.input_size[0], model_path)
            self.backend = 'opencv'
            logger.debug(f"加载 ONNX 模型: {model_path}")
        
        logger.debug(f"CTD 检测器初始化完成 - 设备: {self.device}")
        user_log("system", f"CTD 文本检测模型已加载｜设备 {self.device.upper()}")
    
    
    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """预处理图像"""
        from src.interfaces.ctd.detector import preprocess_img
        return preprocess_img(
            image, 
            input_size=self.input_size, 
            device=self.device,
            half=self.half, 
            to_tensor=self.backend == 'torch'
        )
    
    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """后处理掩码"""
        from src.interfaces.ctd.detector import postprocess_mask
        return postprocess_mask(mask)
    
    @torch.no_grad()
    def _detect_raw(
        self,
        image: np.ndarray,
    ) -> Tuple[List[TextLine], Optional[np.ndarray]]:
        """
        执行原始检测
        
        Args:
            image: OpenCV BGR 格式图像
        Returns:
            Tuple[List[TextLine], Optional[np.ndarray]]
        """
        im_h, im_w = image.shape[:2]
        
        # 预处理
        img_in, ratio, dw, dh = self._preprocess_image(image)
        
        # 推理
        blks, mask, lines_map = self.model(img_in)
        
        if self.backend == 'opencv':
            if mask.shape[1] == 2:
                tmp = mask
                mask = lines_map
                lines_map = tmp
        
        mask = mask.squeeze()
        mask = mask[..., :mask.shape[0]-dh, :mask.shape[1]-dw]
        lines_map = lines_map[..., :lines_map.shape[2]-dh, :lines_map.shape[3]-dw]
        
        # 后处理掩码
        mask = self._postprocess_mask(mask)
        
        # 提取文本行
        lines, scores = self.seg_rep(lines_map, height=im_h, width=im_w)
        idx = np.where(scores[0] > self.box_threshold)
        lines, scores = lines[0][idx], scores[0][idx]
        
        # 调整掩码大小
        mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
        
        # 转换为 TextLine 列表
        textlines = []
        for pts, score in zip(lines, scores):
            pts = pts.astype(np.int32)
            textline = TextLine(pts=pts, confidence=float(score))
            textlines.append(textline)
        
        logger.debug(f"CTD 检测到 {len(textlines)} 个文本行")
        return textlines, mask
