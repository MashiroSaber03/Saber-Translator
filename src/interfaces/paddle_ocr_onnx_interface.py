"""PP-OCRv6 ONNX integration backed by RapidOCR 3.x.

The desktop app ships the official PP-OCRv6 ``medium`` detection and
recognition ONNX models.  PaddlePaddle itself is not required at runtime.
"""

from __future__ import annotations

import logging
import os
import time
from numbers import Real
from typing import List, Tuple

import numpy as np
from PIL import Image

from src.core.ocr_types import OcrResult, create_ocr_result
from src.shared import constants
from src.shared.image_helpers import image_to_rgb_array
from src.shared.memory_errors import is_memory_allocation_error
from src.shared.path_helpers import resource_path

logger = logging.getLogger("PaddleOCR_ONNX")


class PaddleOCRHandlerONNX:
    """Bubble-level PP-OCRv6 Medium handler using ONNX Runtime."""

    MODEL_VERSION = constants.PADDLE_OCR_VERSION
    MODEL_TIER = constants.PADDLE_OCR_MODEL_TIER
    CUDA_EXECUTION_PROVIDER = "CUDAExecutionProvider"

    def __init__(self) -> None:
        self.model_base_dir = resource_path(constants.PADDLE_OCR_MODEL_DIR)
        self.ocr = None
        self.initialized = False

    def _get_model_paths(self) -> Tuple[str, str, str]:
        return (
            os.path.join(self.model_base_dir, "det.onnx"),
            os.path.join(self.model_base_dir, "rec.onnx"),
            os.path.join(self.model_base_dir, "ppocrv6_dict.txt"),
        )

    @staticmethod
    def _check_models_exist(
        det_path: str,
        rec_path: str,
        dict_path: str,
    ) -> Tuple[bool, List[str]]:
        missing: List[str] = []
        if not os.path.isfile(det_path):
            missing.append(f"检测模型: {det_path}")
        if not os.path.isfile(rec_path):
            missing.append(f"识别模型: {rec_path}")
        if not os.path.isfile(dict_path):
            missing.append(f"识别字典: {dict_path}")
        return not missing, missing

    @classmethod
    def _cuda_execution_provider_available(cls) -> bool:
        import onnxruntime as ort

        if cls.CUDA_EXECUTION_PROVIDER not in ort.get_available_providers():
            return False

        import torch

        if not torch.cuda.is_available():
            return False

        preload_dlls = getattr(ort, "preload_dlls", None)
        if callable(preload_dlls):
            try:
                preload_dlls()
            except Exception as error:
                if is_memory_allocation_error(error):
                    raise
                logger.warning("CUDA 运行库预加载失败，PP-OCRv6 将使用 CPU: %s", error)
                return False
        return True

    def initialize(self) -> bool:
        try:
            if self.initialized and self.ocr is not None:
                return True

            det_path, rec_path, dict_path = self._get_model_paths()

            exists, missing = self._check_models_exist(
                det_path,
                rec_path,
                dict_path,
            )
            if not exists:
                for item in missing:
                    logger.error("缺少 %s", item)
                logger.error(
                    "请将单独发布的 %s %s 模型包完整解压到 %s",
                    self.MODEL_VERSION,
                    self.MODEL_TIER,
                    self.model_base_dir,
                )
                return False

            from rapidocr import (
                EngineType,
                LangDet,
                LangRec,
                ModelType,
                OCRVersion,
                RapidOCR,
            )
            use_cuda = self._cuda_execution_provider_available()

            logger.info(
                "初始化 %s %s ONNX 模型 (%s)",
                self.MODEL_VERSION,
                self.MODEL_TIER,
                "CUDA" if use_cuda else "CPU",
            )
            self.ocr = RapidOCR(
                params={
                    "Global.use_cls": False,
                    "EngineConfig.onnxruntime.intra_op_num_threads": 1,
                    "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                    "EngineConfig.onnxruntime.use_cuda": use_cuda,
                    "Det.engine_type": EngineType.ONNXRUNTIME,
                    "Det.lang_type": LangDet.CH,
                    "Det.model_type": ModelType.MEDIUM,
                    "Det.ocr_version": OCRVersion.PPOCRV6,
                    "Det.model_path": det_path,
                    "Rec.engine_type": EngineType.ONNXRUNTIME,
                    "Rec.lang_type": LangRec.CH,
                    "Rec.model_type": ModelType.MEDIUM,
                    "Rec.ocr_version": OCRVersion.PPOCRV6,
                    "Rec.model_path": rec_path,
                    "Rec.rec_keys_path": dict_path,
                }
            )
            self.initialized = True
            logger.info("%s %s 已初始化", self.MODEL_VERSION, self.MODEL_TIER)
            return True
        except ImportError as error:
            logger.error("rapidocr 3.x 或 onnxruntime 未安装: %s", error)
            logger.error("请安装 requirements-cpu.txt 或 requirements-gpu.txt 中的依赖")
            return False
        except Exception as error:
            logger.error("Paddle OCR 初始化失败: %s", error, exc_info=True)
            self.ocr = None
            self.initialized = False
            if is_memory_allocation_error(error):
                raise
            return False

    def recognize_text(
        self,
        image: Image.Image,
        bubble_coords: List[Tuple[int, int, int, int]],
        textlines_per_bubble: List[List[dict]],
    ) -> List[str]:
        return [
            result.text
            for result in self.recognize_text_with_details(
                image,
                bubble_coords,
                textlines_per_bubble,
            )
        ]

    @staticmethod
    def _extract_output_lines(output: object) -> Tuple[List[str], List[float]]:
        """Normalize RapidOCR 3.x output and reject malformed partial data."""

        texts_raw = getattr(output, "txts", None)
        scores_raw = getattr(output, "scores", None)
        if texts_raw is None:
            return [], []
        if scores_raw is None:
            raise RuntimeError("RapidOCR 返回文本但缺少置信度")

        texts = list(texts_raw)
        scores = list(scores_raw)
        if len(texts) != len(scores):
            raise RuntimeError("RapidOCR 文本行与置信度数量不一致")

        normalized_texts: List[str] = []
        normalized_scores: List[float] = []
        for text, score in zip(texts, scores):
            if not isinstance(text, str):
                raise RuntimeError("RapidOCR 文本必须是字符串")
            if isinstance(score, bool) or not isinstance(score, Real):
                raise RuntimeError("RapidOCR 置信度必须是数字")
            normalized_texts.append(text)
            normalized_scores.append(float(score))
        return normalized_texts, normalized_scores

    @staticmethod
    def _expand_textline_polygon(
        polygon: object,
        *,
        image_width: int,
        image_height: int,
    ) -> np.ndarray:
        """Add one source pixel around an ordered text-line quadrilateral."""

        try:
            points = np.asarray(polygon, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError("文本行多边形必须由数字坐标组成") from error
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("文本行多边形必须包含四个有效坐标点")

        horizontal = (points[1] - points[0] + points[2] - points[3]) / 2
        vertical = (points[3] - points[0] + points[2] - points[1]) / 2
        horizontal_length = float(np.linalg.norm(horizontal))
        vertical_length = float(np.linalg.norm(vertical))
        if horizontal_length <= 0 or vertical_length <= 0:
            raise ValueError("文本行多边形面积无效")

        horizontal /= horizontal_length
        vertical /= vertical_length
        points = points + np.stack(
            (
                -horizontal - vertical,
                horizontal - vertical,
                horizontal + vertical,
                -horizontal + vertical,
            )
        )
        points[:, 0] = np.clip(points[:, 0], 0, image_width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, image_height - 1)
        return np.ascontiguousarray(points, dtype=np.float32)

    def recognize_text_with_details(
        self,
        image: Image.Image,
        bubble_coords: List[Tuple[int, int, int, int]],
        textlines_per_bubble: List[List[dict]],
        primary_engine: str = "paddle_ocr",
        fallback_used: bool = False,
    ) -> List[OcrResult]:
        if not self.initialized or self.ocr is None:
            raise RuntimeError("Paddle OCR 未初始化")
        if not bubble_coords:
            return []
        if not isinstance(textlines_per_bubble, list):
            raise ValueError("Paddle OCR 文本行必须是数组")
        if len(textlines_per_bubble) != len(bubble_coords):
            raise ValueError("Paddle OCR 文本行分组数量与气泡数量不匹配")

        try:
            from rapidocr.utils.process_img import get_rotate_crop_image

            rgb = image_to_rgb_array(image)
            # RapidOCR treats a three-channel numpy array as BGR.  Convert
            # explicitly so coloured pages are not inferred with swapped
            # red/blue channels.
            image_bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        except Exception as error:
            logger.error("图像转换失败: %s", error, exc_info=True)
            raise

        recognized_results: List[OcrResult] = []
        image_height, image_width = image_bgr.shape[:2]
        for index, textlines in enumerate(textlines_per_bubble):
            try:
                if not isinstance(textlines, list) or not textlines:
                    raise ValueError(f"气泡 {index} 缺少当前文本行")

                started = time.perf_counter()
                texts: List[str] = []
                scores: List[float] = []
                for line_index, textline in enumerate(textlines):
                    if not isinstance(textline, dict):
                        raise ValueError(
                            f"气泡 {index} 的文本行 {line_index} 必须是对象"
                        )
                    points = self._expand_textline_polygon(
                        textline.get("polygon"),
                        image_width=image_width,
                        image_height=image_height,
                    )
                    line_image = np.ascontiguousarray(
                        get_rotate_crop_image(image_bgr, points)
                    )
                    if line_image.size == 0:
                        raise ValueError(
                            f"气泡 {index} 的文本行 {line_index} 图像区域无效"
                        )
                    output = self.ocr(
                        line_image,
                        use_det=False,
                        use_cls=False,
                        use_rec=True,
                    )
                    line_texts, line_scores = self._extract_output_lines(output)
                    texts.extend(line_texts)
                    scores.extend(line_scores)
                elapsed = time.perf_counter() - started

                text = " ".join(texts)
                confidence = float(np.mean(scores)) if scores else 0.0
                recognized_results.append(
                    create_ocr_result(
                        text,
                        "paddle_ocr",
                        confidence=confidence,
                        confidence_supported=True,
                        primary_engine=primary_engine,
                        fallback_used=fallback_used,
                    )
                )
                logger.info(
                    "气泡 %d/%d OCR 完成，文本行=%d，耗时=%.2fs",
                    index + 1,
                    len(bubble_coords),
                    len(textlines),
                    elapsed,
                )
            except Exception as error:
                logger.error("气泡 %d 识别失败: %s", index, error, exc_info=True)
                raise

        if len(recognized_results) != len(bubble_coords):
            raise RuntimeError("Paddle OCR 结果数量与气泡数量不一致")
        return recognized_results


_paddle_ocr_onnx_handler: PaddleOCRHandlerONNX | None = None


def get_paddle_ocr_handler() -> PaddleOCRHandlerONNX:
    """Return the process-local lazy PP-OCRv6 handler."""

    global _paddle_ocr_onnx_handler
    if _paddle_ocr_onnx_handler is None:
        _paddle_ocr_onnx_handler = PaddleOCRHandlerONNX()
    return _paddle_ocr_onnx_handler


def reset_paddle_ocr_handler() -> None:
    """Release RapidOCR/ONNX sessions and reset the lazy singleton."""

    global _paddle_ocr_onnx_handler
    handler = _paddle_ocr_onnx_handler
    _paddle_ocr_onnx_handler = None
    if handler is None:
        return
    handler.ocr = None
    handler.initialized = False
