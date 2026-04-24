"""
SaberYOLO OBB 后端

用于二阶段防误合并纠错，直接检测最终文本块候选。
"""

import os
import logging
from typing import List, Tuple, Optional

import numpy as np

from ..base import BaseTextDetector
from ..data_types import TextLine
from src.shared.path_helpers import resource_path
from src.shared import constants

logger = logging.getLogger("SaberYoloBackend")


class SaberYoloBackend(BaseTextDetector):
    """
    SaberYOLO 检测后端

    直接输出最终文本块候选，不参与文本行合并。
    """

    requires_merge: bool = False

    def __init__(
        self,
        model_dir: str = None,
        device: str = 'cuda',
        conf_thresh: float = constants.SABER_YOLO_CONF_THRESH,
        iou_thresh: float = constants.SABER_YOLO_IOU_THRESH,
        **kwargs
    ):
        self.model_dir = model_dir or resource_path(constants.SABER_YOLO_MODEL_DIR)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.model_path = None

        super().__init__(device=device, **kwargs)

    def _load_model(self, **kwargs):
        from ultralytics import YOLO as MODEL

        model_path = os.path.join(self.model_dir, constants.SABER_YOLO_MODEL_NAME)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SaberYOLO 模型文件未找到: {model_path}")

        self.model = MODEL(model_path).to(device=self.device)
        self.model_path = model_path
        logger.info(f"SaberYOLO 检测器初始化完成 - 设备: {self.device}, 模型: {model_path}")

    def _detect_raw(self, image: np.ndarray, **kwargs) -> Tuple[List[TextLine], Optional[np.ndarray]]:
        if self.model is None:
            raise RuntimeError("模型未加载")

        result = self.model.predict(
            source=image,
            save=False,
            show=False,
            verbose=False,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            agnostic_nms=True
        )[0]

        if result.obb is None or len(result.obb.cls) == 0:
            return [], None

        textlines = []
        for i in range(len(result.obb.cls)):
            pts = result.obb.xyxyxyxy[i].cpu().numpy().astype(np.int32)
            conf = float(result.obb.conf[i].cpu().numpy())
            textlines.append(TextLine(pts=pts, confidence=conf))

        logger.info(f"SaberYOLO 检测到 {len(textlines)} 个文本块候选")
        return textlines, None
