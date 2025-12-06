"""
CTD (Comic Text Detector) 后端

移植自 manga-image-translator 项目
只保留模型推理核心逻辑
"""

import os
import logging
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import einops

from ..base import BaseTextDetector
from ..data_types import TextLine
from src.shared.path_helpers import resource_path

logger = logging.getLogger("CTDBackend")

# 默认配置
DEFAULT_MODEL_DIR = 'models/ctd'
DEFAULT_INPUT_SIZE = 1024
DEFAULT_NMS_THRESH = 0.35
DEFAULT_CONF_THRESH = 0.4


class CTDBackend(BaseTextDetector):
    """
    CTD 检测后端
    
    100% 遵循 manga-image-translator 的实现
    """
    
    def __init__(self, 
                 model_dir: str = None,
                 device: str = 'cuda',
                 input_size: int = DEFAULT_INPUT_SIZE,
                 half: bool = False,
                 nms_thresh: float = DEFAULT_NMS_THRESH,
                 conf_thresh: float = DEFAULT_CONF_THRESH,
                 **kwargs):
        """
        初始化 CTD 检测器
        
        Args:
            model_dir: 模型文件目录
            device: 设备
            input_size: 输入图像大小
            half: 是否使用半精度
            nms_thresh: NMS阈值
            conf_thresh: 置信度阈值
        """
        self.model_dir = model_dir or resource_path(DEFAULT_MODEL_DIR)
        self.input_size = (input_size, input_size)
        self.half = half
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.backend = None
        self.seg_rep = None
        
        super().__init__(device=device, **kwargs)
    
    def _load_model(self, **kwargs):
        """加载 CTD 模型"""
        # 延迟导入，避免循环依赖
        from src.interfaces.ctd.utils.db_utils import SegDetectorRepresenter
        from src.interfaces.ctd.basemodel import TextDetBase, TextDetBaseDNN
        
        self.seg_rep = SegDetectorRepresenter(thresh=0.3)
        
        if self.device == 'cuda' or self.device == 'mps':
            model_path = os.path.join(self.model_dir, 'comictextdetector.pt')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件未找到: {model_path}")
            self.model = TextDetBase(model_path, device=self.device, act='leaky')
            self.model.to(self.device)
            self.backend = 'torch'
            logger.info(f"加载 PyTorch 模型: {model_path}")
        else:
            model_path = os.path.join(self.model_dir, 'comictextdetector.pt.onnx')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件未找到: {model_path}")
            self.model = TextDetBaseDNN(self.input_size[0], model_path)
            self.backend = 'opencv'
            logger.info(f"加载 ONNX 模型: {model_path}")
        
        logger.info(f"CTD 检测器初始化完成 - 设备: {self.device}")
    
    def _det_batch_forward(self, batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """批量前向推理"""
        from src.interfaces.ctd.basemodel import TextDetBase, TextDetBaseDNN
        
        if isinstance(self.model, TextDetBase):
            batch = einops.rearrange(batch.astype(np.float32) / 255., 'n h w c -> n c h w')
            batch = torch.from_numpy(batch).to(self.device)
            _, mask, lines = self.model(batch)
            mask = mask.detach().cpu().numpy()
            lines = lines.detach().cpu().numpy()
        elif isinstance(self.model, TextDetBaseDNN):
            mask_lst, line_lst = [], []
            for b in batch:
                _, mask, lines = self.model(b)
                if mask.shape[1] == 2:
                    tmp = mask
                    mask = lines
                    lines = tmp
                mask_lst.append(mask)
                line_lst.append(lines)
            lines, mask = np.concatenate(line_lst, 0), np.concatenate(mask_lst, 0)
        else:
            raise NotImplementedError
        return lines, mask
    
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
    def _detect_raw(self, image: np.ndarray, 
                    detect_size: int = None,
                    text_threshold: float = 0.5,
                    box_threshold: float = 0.7,
                    **kwargs) -> Tuple[List[TextLine], Optional[np.ndarray]]:
        """
        执行原始检测
        
        Args:
            image: OpenCV BGR 格式图像
            detect_size: 检测尺寸
            text_threshold: 文本阈值
            box_threshold: 框阈值
            
        Returns:
            Tuple[List[TextLine], Optional[np.ndarray]]
        """
        if detect_size is None:
            detect_size = self.input_size[0]
        
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
        lines, scores = self.seg_rep(None, lines_map, height=im_h, width=im_w)
        idx = np.where(scores[0] > box_threshold)
        lines, scores = lines[0][idx], scores[0][idx]
        
        # 调整掩码大小
        mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
        
        # 转换为 TextLine 列表
        textlines = []
        for pts, score in zip(lines, scores):
            pts = pts.astype(np.int32)
            textline = TextLine(pts=pts, confidence=float(score))
            textlines.append(textline)
        
        logger.info(f"CTD 检测到 {len(textlines)} 个文本行")
        return textlines, mask
