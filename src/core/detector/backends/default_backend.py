"""
Default (DBNet ResNet34) 后端

使用 manga-image-translator 的 detect-20241225.ckpt 模型
只保留模型推理核心逻辑，合并和后处理使用统一架构
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

logger = logging.getLogger("DefaultBackend")

# 默认配置
DEFAULT_MODEL_DIR = 'models/default'
DEFAULT_MODEL_NAME = 'detect-20241225.ckpt'
DEFAULT_DETECT_SIZE = 1536
DEFAULT_TEXT_THRESHOLD = 0.5
DEFAULT_BOX_THRESHOLD = 0.7
DEFAULT_UNCLIP_RATIO = 2.2


class DefaultBackend(BaseTextDetector):
    """
    Default (DBNet ResNet34) 检测后端
    
    使用 manga-image-translator 的 detect-20241225.ckpt 模型
    输出文本行需要合并 (requires_merge = True)
    """
    
    # 需要文本行合并
    requires_merge: bool = True
    
    def __init__(self,
                 model_dir: str = None,
                 device: str = 'cuda',
                 detect_size: int = DEFAULT_DETECT_SIZE,
                 text_threshold: float = DEFAULT_TEXT_THRESHOLD,
                 box_threshold: float = DEFAULT_BOX_THRESHOLD,
                 unclip_ratio: float = DEFAULT_UNCLIP_RATIO,
                 **kwargs):
        """
        初始化 Default 检测器
        
        Args:
            model_dir: 模型目录
            device: 设备 ('cuda', 'cpu', 'mps')
            detect_size: 检测尺寸
            text_threshold: 文本阈值
            box_threshold: 框阈值
            unclip_ratio: 框扩展比例
        """
        self.model_dir = model_dir or resource_path(DEFAULT_MODEL_DIR)
        self.detect_size = detect_size
        self.text_threshold = text_threshold
        self.box_threshold = box_threshold
        self.unclip_ratio = unclip_ratio
        self.seg_rep = None
        
        super().__init__(device=device, **kwargs)
    
    def _load_model(self, **kwargs):
        """加载模型"""
        from src.interfaces.default.DBNet_resnet34 import TextDetection
        from src.interfaces.ctd.utils.db_utils import SegDetectorRepresenter
        
        model_path = os.path.join(self.model_dir, DEFAULT_MODEL_NAME)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"模型文件未找到: {model_path}\n"
                f"请从 https://github.com/zyddnys/manga-image-translator/releases 下载 detect-20241225.ckpt\n"
                f"并放置到 {self.model_dir} 目录"
            )
        
        # 加载模型
        logger.info(f"加载 Default 模型: {model_path}")
        self.model = TextDetection()
        sd = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(sd['model'] if 'model' in sd else sd)
        self.model.eval()
        
        if self.device in ('cuda', 'mps'):
            self.model = self.model.to(self.device)
        
        # 初始化后处理器 (复用 CTD 的 SegDetectorRepresenter)
        self.seg_rep = SegDetectorRepresenter(
            thresh=self.text_threshold,
            box_thresh=self.box_threshold,
            unclip_ratio=self.unclip_ratio
        )
        
        logger.info(f"Default 检测器初始化完成 - 设备: {self.device}")
    
    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """
        预处理图像
        
        Args:
            image: BGR 格式图像
            
        Returns:
            img_resized: 缩放后的图像
            ratio: 缩放比例
            pad_w: 宽度填充
            pad_h: 高度填充
        """
        from src.interfaces.default.imgproc import resize_aspect_ratio
        
        # 双边滤波降噪 (与原实现一致)
        image_filtered = cv2.bilateralFilter(image, 17, 80, 80)
        
        img_resized, ratio, _, pad_w, pad_h = resize_aspect_ratio(
            image_filtered,
            self.detect_size,
            cv2.INTER_LINEAR,
            mag_ratio=1
        )
        return img_resized, ratio, pad_w, pad_h
    
    @torch.no_grad()
    def _detect_raw(self, image: np.ndarray, **kwargs) -> Tuple[List[TextLine], Optional[np.ndarray]]:
        """
        执行原始检测
        
        Args:
            image: OpenCV BGR 格式图像
            
        Returns:
            Tuple[List[TextLine], Optional[np.ndarray]]:
                - 文本行列表 (四边形)
                - 文本掩码
        """
        from src.interfaces.default.imgproc import adjustResultCoordinates
        
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        im_h, im_w = image.shape[:2]
        
        # 预处理
        img_resized, ratio, pad_w, pad_h = self._preprocess_image(image)
        img_resized_h, img_resized_w = img_resized.shape[:2]
        ratio_h = ratio_w = 1 / ratio
        
        # 转换为 tensor: (H, W, C) -> (1, C, H, W), 归一化到 [-1, 1]
        batch = einops.rearrange(
            img_resized.astype(np.float32) / 127.5 - 1.0,
            'h w c -> 1 c h w'
        )
        batch = torch.from_numpy(batch).to(self.device)
        
        # 模型推理
        db, mask = self.model(batch)
        db = db.sigmoid().cpu().numpy()
        mask = mask.cpu().numpy()
        
        # 后处理 - 使用 SegDetectorRepresenter 提取文本框
        # 注意: CTD 版本的 SegDetectorRepresenter 需要显式传入 height/width 参数
        mask_squeezed = mask[0, 0, :, :]
        boxes, scores = self.seg_rep(
            None, db,
            height=img_resized_h,
            width=img_resized_w
        )
        boxes, scores = boxes[0], scores[0]
        
        # 过滤无效框并调整坐标
        textlines = []
        if boxes.size > 0:
            # 过滤全零框
            idx = boxes.reshape(boxes.shape[0], -1).sum(axis=1) > 0
            polys = boxes[idx].astype(np.float64)
            valid_scores = scores[idx]
            
            # 调整坐标到原图尺寸
            polys = adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=1)
            polys = polys.astype(np.int32)
            
            # 转换为 TextLine 对象
            for pts, score in zip(polys, valid_scores):
                if pts.shape[0] == 4:
                    textline = TextLine(pts=pts, confidence=float(score))
                    # 过滤太小的区域
                    if textline.area > 16:
                        textlines.append(textline)
        
        # 处理掩码 - 缩放到原图尺寸
        # mask 输出是 1/2 分辨率（从 up4 输出，经过 upconv7 上采样后是 H/2）
        # 不需要额外放大，直接移除 padding
        
        # 移除填充区域 (pad 值需要除以 2，因为 mask 是在 1/2 分辨率上)
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        
        mask_cropped = mask_squeezed
        if pad_h_half > 0:
            mask_cropped = mask_cropped[:-pad_h_half, :]
        if pad_w_half > 0:
            mask_cropped = mask_cropped[:, :-pad_w_half]
        
        # 缩放到原图尺寸
        raw_mask = cv2.resize(mask_cropped, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
        raw_mask = np.clip(raw_mask * 255, 0, 255).astype(np.uint8)
        
        logger.info(f"Default 检测到 {len(textlines)} 个文本行")
        return textlines, raw_mask
