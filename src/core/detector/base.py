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

from src.shared.user_logging import user_log

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
    
    # 是否需要合并文本行（CTD/YSGYolo/Default 需要）
    requires_merge: bool = True
    detector_id: str = ''
    
    def __init__(self, device: str = 'cuda'):
        """
        初始化检测器
        
        Args:
            device: 运行设备 ('cuda', 'cpu', 'mps')
        """
        self.device = self._resolve_device(device)
        self.model = None
        self._load_model()
    
    @staticmethod
    def _resolve_device(device: str) -> str:
        """解析设备，处理 CUDA 不可用的情况"""
        import torch
        if device == 'cuda' and not torch.cuda.is_available():
            logger.debug("CUDA 不可用，回退到 CPU")
            user_log("warning", "文本检测未找到可用 GPU，已切换到 CPU")
            return 'cpu'
        return device
    
    @abstractmethod
    def _load_model(self):
        """
        加载模型
        
        子类必须实现此方法
        """
        pass
    
    @abstractmethod
    def _detect_raw(
        self,
        image: np.ndarray,
    ) -> Tuple[List[TextLine], Optional[np.ndarray]]:
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

    @staticmethod
    def _validate_raw_result(
        result,
        image_width: int,
        image_height: int,
    ) -> Tuple[List[TextLine], Optional[np.ndarray]]:
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("检测器必须返回 (文本行列表, 掩码)")
        textlines, mask = result
        if not isinstance(textlines, list) or any(
            not isinstance(line, TextLine) for line in textlines
        ):
            raise TypeError("检测器文本行结果必须是 TextLine 列表")
        if mask is not None:
            if not isinstance(mask, np.ndarray) or mask.ndim != 2:
                raise TypeError("检测器掩码必须是二维 numpy 数组")
            if mask.shape != (image_height, image_width):
                raise ValueError("检测器掩码尺寸必须与输入图像一致")
        return textlines, mask
    
    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """
        预处理：PIL 图像转 OpenCV BGR
        """
        converted = image.convert('RGB')
        try:
            img_np = np.array(converted)
        finally:
            if converted is not image:
                converted.close()
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    def detect(
        self,
        image: Image.Image,
        merge_lines: bool = None,
        edge_ratio_threshold: float = 0.0,
        sort_method: str = 'smart',  # 排序方法
        right_to_left: bool = True,  # 阅读方向
        enable_aux_yolo_detection: bool = None,
        aux_yolo_conf_threshold: float = None,
        aux_yolo_overlap_threshold: float = None,
    ) -> DetectionResult:
        """
        统一检测入口
        
        流程:
        1. 预处理图像
        2. 调用 _detect_raw() 获取原始文本行
        3. 文本行合并（根据 requires_merge 决定）
        4. 后处理（包括智能排序）
        
        Args:
            image: PIL 图像
            merge_lines: 是否合并文本行（None 时使用检测器默认值 requires_merge）
            edge_ratio_threshold: 边缘距离比例阈值
            sort_method: 排序方法 ('smart', 'area', 'reading', 'none')
            right_to_left: 是否从右到左阅读（日漫模式）
            
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
        textlines, mask = self._validate_raw_result(
            self._detect_raw(img_cv),
            im_w,
            im_h,
        )

        if enable_aux_yolo_detection is None:
            from src.shared import constants

            enable_aux_yolo_detection = constants.ENABLE_AUX_YOLO_DETECTION

        if enable_aux_yolo_detection:
            from .aux_yolo import maybe_merge_with_aux_yolo

            textlines = maybe_merge_with_aux_yolo(
                img_cv,
                textlines,
                detector_type=self.detector_id,
                enabled=enable_aux_yolo_detection,
                conf_threshold=aux_yolo_conf_threshold,
                overlap_threshold=aux_yolo_overlap_threshold,
            )
        
        if not textlines:
            logger.debug("未检测到文本区域")
            return DetectionResult(blocks=[], mask=mask, raw_lines=[])
        
        logger.debug(f"检测到 {len(textlines)} 个文本行")
        
        # 3. 文本行合并
        if merge_lines and textlines:
            blocks = merge_textlines(
                textlines, im_w, im_h,
                edge_ratio_threshold=edge_ratio_threshold,
                verbose=True
            )
            logger.debug(f"合并后得到 {len(blocks)} 个文本块")
        else:
            # 不合并时，每个 TextLine 创建一个 TextBlock
            blocks = [TextBlock(lines=[line]) for line in textlines]
        
        # 4. 后处理（包括智能排序）
        blocks = postprocess_blocks(
            blocks,
            sort_method=sort_method,
            img=img_cv,  # 传递图像用于分镜检测
            right_to_left=right_to_left
        )
        
        return DetectionResult(
            blocks=blocks,
            mask=mask,
            raw_lines=textlines
        )
