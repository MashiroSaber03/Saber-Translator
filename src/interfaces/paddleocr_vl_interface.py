"""PaddleOCR-VL-1.6 bubble OCR using the native Transformers 5 model."""

from __future__ import annotations

import logging
import os
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from src.shared import constants
from src.shared.memory_errors import is_memory_allocation_error
from src.shared.paddleocr_vl import (
    PADDLEOCR_VL_LANGUAGE_NAMES,
    build_paddleocr_vl_prompt,
)
from src.shared.path_helpers import resource_path
from src.shared.user_logging import inline_log_text, user_log

logger = logging.getLogger("PaddleOCR_VL")

PADDLEOCR_VL_MAX_NEW_TOKENS = 512

_REQUIRED_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "processor_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "chat_template.jinja",
)


class PaddleOCRVLHandler:
    """Process-local native PaddleOCR-VL-1.6 handler."""

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.device: str | None = None
        self.torch_dtype: torch.dtype | None = None
        self.initialized = False

    def _get_model_path(self) -> str:
        model_path = resource_path(constants.PADDLEOCR_VL_MODEL_DIR)
        missing = [
            filename
            for filename in _REQUIRED_MODEL_FILES
            if not os.path.isfile(os.path.join(model_path, filename))
        ]
        if missing:
            raise FileNotFoundError(
                f"{constants.PADDLEOCR_VL_VERSION} 模型包不完整，缺少: "
                f"{', '.join(missing)}。请将单独发布的模型包完整解压到 "
                f"{model_path}"
            )
        return model_path

    @staticmethod
    def _resolve_device_and_dtype(requested_device: str) -> Tuple[str, torch.dtype]:
        if requested_device == "cuda" and torch.cuda.is_available():
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
            return "cuda", dtype
        if requested_device == "mps" and torch.backends.mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32

    def _release_loaded_model(self) -> None:
        loaded_device = self.device
        self.model = None
        self.processor = None
        self.device = None
        self.torch_dtype = None
        self.initialized = False
        if loaded_device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif loaded_device == "mps":
            torch.mps.empty_cache()

    def initialize(self, device: str = "cpu") -> bool:
        if self.initialized:
            return True

        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            model_path = self._get_model_path()
            self.device, self.torch_dtype = self._resolve_device_and_dtype(device)

            self.processor = AutoProcessor.from_pretrained(
                model_path,
                local_files_only=True,
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                local_files_only=True,
                dtype=self.torch_dtype,
            )
            self.model = self.model.to(self.device).eval()
            self.initialized = True
            logger.debug(
                "%s 原生模型已加载到 %s，精度=%s",
                constants.PADDLEOCR_VL_VERSION,
                self.device,
                self.torch_dtype,
            )
            user_log(
                "system",
                f"{constants.PADDLEOCR_VL_VERSION} 模型已加载｜"
                f"设备 {self.device.upper()}｜精度 {self.torch_dtype}",
            )
            return True
        except ImportError as error:
            logger.error("PaddleOCR-VL 原生 Transformers 运行时不可用: %s", error)
            user_log(
                "error",
                f"PaddleOCR-VL 运行库不可用｜{inline_log_text(error)}",
            )
            self._release_loaded_model()
            return False
        except Exception as error:
            logger.error("PaddleOCR-VL 初始化失败: %s", error, exc_info=True)
            self._release_loaded_model()
            if is_memory_allocation_error(error):
                raise
            return False

    def _recognize_single(
        self,
        image: np.ndarray | Image.Image,
        source_language: str,
    ) -> str:
        if not self.initialized or self.model is None or self.processor is None:
            raise RuntimeError("PaddleOCR-VL 未初始化")

        prompt = build_paddleocr_vl_prompt(source_language)

        owns_image = False
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
            owns_image = True
        else:
            pil_image = image

        if pil_image.mode != "RGB":
            converted = pil_image.convert("RGB")
            if owns_image:
                pil_image.close()
            pil_image = converted
            owns_image = True

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=PADDLEOCR_VL_MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                )

            input_length = inputs["input_ids"].shape[-1]
            output_text = self.processor.decode(
                generated_ids[0][input_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return output_text.strip()
        except Exception as error:
            logger.error(
                "PaddleOCR-VL 识别失败 (%s, image=%s): %s",
                type(error).__name__,
                pil_image.size,
                error,
            )
            raise
        finally:
            if owns_image:
                pil_image.close()

    def recognize_text(
        self,
        image: Image.Image,
        bubble_coords: List[Tuple[int, int, int, int]],
        source_language: str,
    ) -> List[str]:
        if not self.initialized or self.model is None:
            raise RuntimeError("PaddleOCR-VL 未初始化")
        if not bubble_coords:
            return []

        build_paddleocr_vl_prompt(source_language)
        logger.debug(
            "使用 %s 识别 %d 个气泡，源语言=%s",
            constants.PADDLEOCR_VL_VERSION,
            len(bubble_coords),
            PADDLEOCR_VL_LANGUAGE_NAMES[source_language],
        )

        converted = image.convert("RGB")
        try:
            image_array = np.array(converted)
        finally:
            converted.close()

        results: List[str] = []
        for index, (x1, y1, x2, y2) in enumerate(bubble_coords):
            bubble = image_array[y1:y2, x1:x2]
            if bubble.size == 0:
                raise ValueError(f"气泡 {index} 图像区域无效")
            text = self._recognize_single(bubble, source_language)
            results.append(text)
            logger.debug(
                "气泡 %d/%d %s 识别完成",
                index + 1,
                len(bubble_coords),
                constants.PADDLEOCR_VL_VERSION,
            )
        return results


_paddleocr_vl_handler: PaddleOCRVLHandler | None = None


def get_paddleocr_vl_handler() -> PaddleOCRVLHandler:
    global _paddleocr_vl_handler
    if _paddleocr_vl_handler is None:
        _paddleocr_vl_handler = PaddleOCRVLHandler()
    return _paddleocr_vl_handler


def reset_paddleocr_vl_handler() -> None:
    global _paddleocr_vl_handler
    handler = _paddleocr_vl_handler
    _paddleocr_vl_handler = None
    if handler is not None:
        handler._release_loaded_model()
    logger.debug("PaddleOCR-VL 处理器已重置")
