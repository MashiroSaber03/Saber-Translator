"""
检测器基类

定义所有检测器的统一接口
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .data_types import TextLine, TextBlock, DetectionResult
from .textline_merge import merge_textlines
from .postprocess import postprocess_blocks

logger = logging.getLogger("BaseDetector")


class BaseTextDetector(ABC):
    """
    文本检测器基类
    
    所有检测器后端必须继承此类并实现:
    - _load_model(): 加载模型
    - _detect_raw(): 执行原始检测，返回 TextLine 列表
    """
    
    # 是否需要合并文本行（CTD/YSGYolo需要，YOLOv5不需要）
    requires_merge: bool = True
    
    def __init__(self, device: str = 'cuda', **kwargs):
        """
        初始化检测器
        
        Args:
            device: 运行设备 ('cuda', 'cpu', 'mps')
        """
        self.device = self._resolve_device(device)
        self.model = None
        self._load_model(**kwargs)
    
    @staticmethod
    def _resolve_device(device: str) -> str:
        """解析设备，处理 CUDA 不可用的情况"""
        import torch
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，回退到 CPU")
            return 'cpu'
        return device
    
    @abstractmethod
    def _load_model(self, **kwargs):
        """
        加载模型
        
        子类必须实现此方法
        """
        pass
    
    @abstractmethod
    def _detect_raw(self, image: np.ndarray, **kwargs) -> Tuple[List[TextLine], Optional[np.ndarray]]:
        """
        执行原始检测
        
        子类必须实现此方法
        
        Args:
            image: OpenCV BGR 格式图像
            
        Returns:
            Tuple[List[TextLine], Optional[np.ndarray]]: 
                - 文本行列表
                - 可选的掩码图像
        """
        pass
    
    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """
        预处理：PIL 图像转 OpenCV BGR
        """
        img_np = np.array(image.convert('RGB'))
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    def detect(
        self,
        image: Image.Image,
        merge_lines: bool = None,
        edge_ratio_threshold: float = 0.0,
        expand_ratio: float = 0,
        expand_top: float = 0,
        expand_bottom: float = 0,
        expand_left: float = 0,
        expand_right: float = 0,
        **kwargs
    ) -> DetectionResult:
        """
        统一检测入口
        
        流程:
        1. 预处理图像
        2. 调用 _detect_raw() 获取原始文本行
        3. 文本行合并（根据 requires_merge 决定）
        4. 后处理
        
        Args:
            image: PIL 图像
            merge_lines: 是否合并文本行（None 时使用检测器默认值 requires_merge）
            edge_ratio_threshold: 边缘距离比例阈值
            expand_ratio: 整体扩展比例 (%)
            expand_top/bottom/left/right: 各边额外扩展 (%)
            
        Returns:
            DetectionResult: 检测结果
        """
        # 使用检测器默认的合并设置
        if merge_lines is None:
            merge_lines = self.requires_merge
        
        # 1. 预处理
        img_cv = self._preprocess(image)
        im_w, im_h = image.width, image.height
        
        # 2. 原始检测
        textlines, mask = self._detect_raw(img_cv, **kwargs)
        
        if not textlines:
            logger.info("未检测到文本区域")
            return DetectionResult(blocks=[], mask=mask, raw_lines=[])
        
        logger.info(f"检测到 {len(textlines)} 个文本行")
        
        # 3. 文本行合并
        if merge_lines and textlines:
            blocks = merge_textlines(
                textlines, im_w, im_h,
                edge_ratio_threshold=edge_ratio_threshold,
                verbose=True
            )
            logger.info(f"合并后得到 {len(blocks)} 个文本块")
        else:
            # 不合并时，每个 TextLine 创建一个 TextBlock
            blocks = [TextBlock(lines=[line]) for line in textlines]
        
        # 4. 后处理
        blocks = postprocess_blocks(
            blocks, im_w, im_h,
            expand_ratio=expand_ratio,
            expand_top=expand_top,
            expand_bottom=expand_bottom,
            expand_left=expand_left,
            expand_right=expand_right
        )
        
        return DetectionResult(
            blocks=blocks,
            mask=mask,
            raw_lines=textlines
        )
    
    def detect_to_legacy_format(
        self,
        image: Image.Image,
        **kwargs
    ) -> dict:
        """
        检测并返回旧格式结果
        
        向后兼容接口，返回 {'coords': [...], 'polygons': [...], 'angles': [...]}
        """
        result = self.detect(image, **kwargs)
        return result.to_legacy_format()
