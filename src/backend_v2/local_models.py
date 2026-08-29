"""Built-in local models that can be kept resident by the Worker."""

from __future__ import annotations

from dataclasses import dataclass
import sys


@dataclass(frozen=True, slots=True)
class LocalModelOption:
    model_id: str
    label: str
    description: str


LOCAL_MODEL_OPTIONS = (
    LocalModelOption(
        "detector_default",
        "Default 文本检测",
        "默认 DBNet 检测模型",
    ),
    LocalModelOption(
        "detector_ctd",
        "CTD 文本检测",
        "Comic Text Detector 检测模型",
    ),
    LocalModelOption(
        "detector_yolo",
        "YSGYolo 文本检测",
        "YSGYolo 检测模型，也用于辅助检测",
    ),
    LocalModelOption(
        "saber_yolo",
        "SaberYOLO 文本检测",
        "SaberYOLO 旋转框检测模型",
    ),
    LocalModelOption(
        "manga_ocr",
        "MangaOCR",
        "日文漫画 OCR 模型",
    ),
    LocalModelOption(
        "paddle_ocr",
        "PP-OCRv6",
        "Paddle 文本检测与识别模型",
    ),
    LocalModelOption(
        "ocr_48px",
        "48px OCR",
        "48px 漫画文本识别模型",
    ),
    LocalModelOption(
        "paddleocr_vl",
        "PaddleOCR-VL",
        "PaddleOCR-VL 视觉语言识别模型",
    ),
    LocalModelOption(
        "lama_mpe",
        "LaMA MPE",
        "LaMA MPE 文字背景修复模型",
    ),
    LocalModelOption(
        "litelama",
        "LiteLaMA",
        "LiteLaMA 文字背景修复模型",
    ),
)
LOCAL_MODEL_IDS = tuple(option.model_id for option in LOCAL_MODEL_OPTIONS)
LOCAL_MODEL_LABELS = {
    option.model_id: option.label for option in LOCAL_MODEL_OPTIONS
}

_DETECTOR_TYPES = {
    "detector_default": "default",
    "detector_ctd": "ctd",
    "detector_yolo": "yolo",
    "saber_yolo": "saber_yolo",
}
_SINGLETON_RESETTERS = {
    "manga_ocr": (
        "src.interfaces.manga_ocr_interface",
        "_manga_ocr_instance",
        "reset_manga_ocr_instance",
    ),
    "ocr_48px": (
        "src.interfaces.ocr_48px.interface",
        "_model_48px_instance",
        "reset_48px_ocr_handler",
    ),
    "paddleocr_vl": (
        "src.interfaces.paddleocr_vl_interface",
        "_paddleocr_vl_handler",
        "reset_paddleocr_vl_handler",
    ),
    "paddle_ocr": (
        "src.interfaces.paddle_ocr_onnx_interface",
        "_paddle_ocr_onnx_handler",
        "reset_paddle_ocr_handler",
    ),
    "litelama": (
        "src.interfaces.lama_interface",
        "_litelama_inpainter",
        "reset_litelama_inpainter",
    ),
    "lama_mpe": (
        "src.interfaces.lama_mpe_interface",
        "_inpainter",
        "reset_lama_mpe_inpainter",
    ),
}


def normalize_resident_models(model_ids: object) -> tuple[str, ...]:
    """Validate, deduplicate, and return model ids in catalog order."""

    if model_ids is None:
        return ()
    if isinstance(model_ids, (str, bytes)) or not isinstance(
        model_ids,
        (list, tuple, set, frozenset),
    ):
        raise ValueError("resident models must be a collection of model ids")
    requested: set[str] = set()
    for model_id in model_ids:
        if not isinstance(model_id, str) or model_id not in LOCAL_MODEL_LABELS:
            raise ValueError(f"unsupported resident model: {model_id!r}")
        requested.add(model_id)
    return tuple(model_id for model_id in LOCAL_MODEL_IDS if model_id in requested)


def preload_local_models(model_ids: object) -> tuple[str, ...]:
    """Load configured resident models using their normal lazy-load entrypoints."""

    resident_models = normalize_resident_models(model_ids)
    if not resident_models:
        return resident_models

    from src.shared.user_logging import inline_log_text, user_log

    labels = [LOCAL_MODEL_LABELS[model_id] for model_id in resident_models]
    user_log("system", f"正在加载常驻模型｜{', '.join(labels)}")
    for model_id in resident_models:
        try:
            _load_local_model(model_id)
        except Exception as error:
            label = LOCAL_MODEL_LABELS[model_id]
            user_log(
                "error",
                f"常驻模型加载失败｜{label}｜{inline_log_text(error)}",
            )
            raise RuntimeError(f"常驻模型 {label} 加载失败：{error}") from error
    user_log("system", f"常驻模型已就绪｜{', '.join(labels)}")
    return resident_models


def _load_local_model(model_id: str) -> None:
    detector_type = _DETECTOR_TYPES.get(model_id)
    if detector_type is not None:
        from src.core.detector.registry import get_detector

        get_detector(detector_type)
        return
    if model_id == "manga_ocr":
        from src.interfaces.manga_ocr_interface import get_manga_ocr_instance

        get_manga_ocr_instance()
        return
    if model_id == "paddle_ocr":
        from src.interfaces.paddle_ocr_onnx_interface import get_paddle_ocr_handler

        if not get_paddle_ocr_handler().initialize():
            raise RuntimeError("PP-OCRv6 初始化失败")
        return
    if model_id == "ocr_48px":
        import torch

        from src.interfaces.ocr_48px import get_48px_ocr_handler

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not get_48px_ocr_handler().initialize(device):
            raise RuntimeError("48px OCR 初始化失败")
        return
    if model_id == "paddleocr_vl":
        import torch

        from src.interfaces.paddleocr_vl_interface import get_paddleocr_vl_handler

        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        if not get_paddleocr_vl_handler().initialize(device):
            raise RuntimeError("PaddleOCR-VL 初始化失败")
        return
    if model_id == "lama_mpe":
        from src.interfaces.lama_mpe_interface import get_lama_mpe_inpainter

        get_lama_mpe_inpainter().load()
        return
    if model_id == "litelama":
        from src.interfaces.lama_interface import get_litelama_inpainter

        get_litelama_inpainter().load()
        return
    raise ValueError(f"unsupported local model: {model_id}")


def release_loaded_local_model(model_id: str) -> bool:
    """Release one model only when its process-local singleton is loaded."""

    detector_type = _DETECTOR_TYPES.get(model_id)
    if detector_type is not None:
        module = sys.modules.get("src.core.detector.registry")
        if module is None:
            return False
        instances = getattr(module, "_detector_instances", None)
        if not isinstance(instances, dict) or detector_type not in instances:
            return False
        resetter = getattr(module, "reset_detector", None)
        if not callable(resetter):
            return False
        resetter(detector_type)
        return True

    resetter_spec = _SINGLETON_RESETTERS.get(model_id)
    if resetter_spec is None:
        raise ValueError(f"unsupported local model: {model_id}")
    module_name, singleton_name, resetter_name = resetter_spec
    module = sys.modules.get(module_name)
    if module is None or getattr(module, singleton_name, None) is None:
        return False
    resetter = getattr(module, resetter_name, None)
    if not callable(resetter):
        return False
    resetter()
    return True
