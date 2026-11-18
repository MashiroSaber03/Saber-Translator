"""Built-in local models that can be kept resident by the Worker."""

from __future__ import annotations

import sys

from src.shared.model_catalog import LOCAL_MODELS, model_available

LOCAL_MODEL_OPTIONS = LOCAL_MODELS
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
    "lama_manga": (
        "src.interfaces.lama_interface",
        "_lama_manga_inpainter",
        "reset_lama_manga_inpainter",
    ),
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

    available = tuple(model_id for model_id in resident_models if model_available(model_id))
    missing = set(resident_models) - set(available)
    if missing:
        user_log("system", "常驻模型尚未安装，跳过预加载；可在设置 → 更多 → 模型管理下载｜" +
                 "、".join(LOCAL_MODEL_LABELS[key] for key in resident_models if key in missing))
    if not available:
        return ()
    labels = [LOCAL_MODEL_LABELS[model_id] for model_id in available]
    user_log("system", f"正在加载常驻模型｜{', '.join(labels)}")
    loaded: list[str] = []
    for model_id in available:
        try:
            _load_local_model(model_id)
        except Exception as error:
            label = LOCAL_MODEL_LABELS[model_id]
            user_log(
                "error",
                f"常驻模型加载失败，已跳过预加载；请在设置 → 更多 → 模型管理校验或修复｜{label}｜{inline_log_text(error)}",
            )
            release_loaded_local_model(model_id)
            continue
        loaded.append(model_id)
    if loaded:
        user_log("system", "常驻模型已就绪｜" + '、'.join(LOCAL_MODEL_LABELS[key] for key in loaded))
    return tuple(loaded)


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
    if model_id == "lama_manga":
        from src.interfaces.lama_interface import get_lama_manga_inpainter

        get_lama_manga_inpainter().load()
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
