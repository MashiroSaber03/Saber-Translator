"""Administrator-owned limits for ordinary users in the public profile."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from typing import Any

from sqlalchemy import Engine, select, update

from src.backend_v2.auth.context import current_user_role
from src.backend_v2.content.page_style import validate_page_style
from src.backend_v2.runtime_profile import RuntimeProfile
from src.backend_v2.serialization import canonical_json
from src.backend_v2.storage.schema import pages, platform_config
from src.backend_v2.timestamps import utcnow


FEATURE_LABELS = {
    "translation": "翻译",
    "insight": "漫画分析",
    "characterStudio": "角色工坊",
    "editMode": "编辑模式",
}
MODEL_LABELS = {
    "detector_default": "Default（DBNet）文字检测",
    "detector_ctd": "CTD 文字检测",
    "detector_yolo": "YOLO 文字检测",
    "aux_ysg_yolo": "辅助 YSGYolo 检测",
    "saber_yolo": "SaberYOLO 二阶段纠错",
    "manga_ocr": "MangaOCR",
    "ocr_48px": "48px OCR",
    "paddle_ocr": "PaddleOCR",
    "paddleocr_vl": "PaddleOCR-VL",
    "lama_mpe": "LAMA 修复（速度优化）",
    "litelama": "LAMA 修复（通用）",
}

DEFAULT_PUBLIC_USER_POLICY: dict[str, Any] = {
    "features": {key: True for key in FEATURE_LABELS},
    "models": {key: True for key in MODEL_LABELS},
    "settings": {
        "lamaDisableResize": {"editable": False, "value": False},
        "parallel": {"allowed": False},
    },
}

_DETECTOR_MODELS = {
    "default": "detector_default",
    "ctd": "detector_ctd",
    "yolo": "detector_yolo",
}
_OCR_MODELS = {
    "manga_ocr": "manga_ocr",
    "48px_ocr": "ocr_48px",
    "paddle_ocr": "paddle_ocr",
    "paddleocr_vl": "paddleocr_vl",
}
_INPAINT_MODELS = {
    "lama_mpe": "lama_mpe",
    "litelama": "litelama",
}


class PublicPolicyDenied(PermissionError):
    """A public ordinary-user request violates the administrator policy."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _exact_object(
    value: object,
    *,
    label: str,
    keys: set[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} 字段无效")
    return dict(value)


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} 必须是布尔值")
    return value


def validate_public_user_policy(value: object) -> dict[str, Any]:
    root = _exact_object(
        value,
        label="公网用户策略",
        keys={"features", "models", "settings"},
    )
    features = _exact_object(
        root["features"],
        label="公网功能策略",
        keys=set(FEATURE_LABELS),
    )
    models = _exact_object(
        root["models"],
        label="本地模型策略",
        keys=set(MODEL_LABELS),
    )
    settings = _exact_object(
        root["settings"],
        label="设置权限策略",
        keys={"lamaDisableResize", "parallel"},
    )
    lama = _exact_object(
        settings["lamaDisableResize"],
        label="LAMA 自动缩放策略",
        keys={"editable", "value"},
    )
    parallel = _exact_object(
        settings["parallel"],
        label="并行模式策略",
        keys={"allowed"},
    )
    return {
        "features": {
            key: _boolean(features[key], f"features.{key}")
            for key in FEATURE_LABELS
        },
        "models": {
            key: _boolean(models[key], f"models.{key}")
            for key in MODEL_LABELS
        },
        "settings": {
            "lamaDisableResize": {
                "editable": _boolean(
                    lama["editable"],
                    "settings.lamaDisableResize.editable",
                ),
                "value": _boolean(
                    lama["value"],
                    "settings.lamaDisableResize.value",
                ),
            },
            "parallel": {
                "allowed": _boolean(
                    parallel["allowed"],
                    "settings.parallel.allowed",
                ),
            },
        },
    }


class PublicUserPolicyRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def load(self) -> dict[str, Any]:
        with self.engine.connect() as connection:
            payload = connection.execute(
                select(platform_config.c.public_user_policy_json).where(
                    platform_config.c.singleton_id == 1
                )
            ).scalar_one()
        try:
            value = json.loads(str(payload))
        except json.JSONDecodeError as exc:
            raise RuntimeError("公网用户策略不是有效 JSON") from exc
        return validate_public_user_policy(value)

    def save(self, value: object) -> dict[str, Any]:
        policy = validate_public_user_policy(value)
        with self.engine.begin() as connection:
            connection.execute(
                update(platform_config)
                .where(platform_config.c.singleton_id == 1)
                .values(
                    public_user_policy_json=canonical_json(policy),
                    updated_at=utcnow(),
                )
            )
        return policy


class PublicUserPolicyAccess:
    """Apply the stored policy only to ordinary public-profile requests."""

    def __init__(self, engine: Engine, profile: RuntimeProfile) -> None:
        self.profile = profile
        self.repository = (
            PublicUserPolicyRepository(engine)
            if profile.name == "public"
            else None
        )

    def restricted(self) -> bool:
        return self.profile.name == "public" and current_user_role() != "admin"

    def policy(self) -> dict[str, Any]:
        if self.repository is None:
            raise RuntimeError("public policy is unavailable in the local profile")
        return self.repository.load()

    def require_feature(self, feature: str) -> None:
        if feature not in FEATURE_LABELS:
            raise ValueError(f"unknown public feature: {feature}")
        if self.restricted() and not self.policy()["features"][feature]:
            raise PublicPolicyDenied(
                "feature_disabled",
                f"管理员已关闭{FEATURE_LABELS[feature]}",
            )

    def require_model(self, model: str) -> None:
        if model not in MODEL_LABELS:
            raise ValueError(f"unknown local model policy: {model}")
        if self.restricted() and not self.policy()["models"][model]:
            raise PublicPolicyDenied(
                "model_disabled",
                f"管理员已关闭{MODEL_LABELS[model]}",
            )

    def apply_translation_setting(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        result = deepcopy(dict(payload))
        if not self.restricted():
            return result
        settings = self.policy()["settings"]
        lama = settings["lamaDisableResize"]
        if not lama["editable"]:
            result["lamaDisableResize"] = lama["value"]
        parallel = dict(result.get("parallel", {}))
        if not settings["parallel"]["allowed"]:
            parallel["enabled"] = False
        result["parallel"] = parallel
        return result

    def apply_settings_document(self, document: dict[str, Any]) -> dict[str, Any]:
        if not self.restricted():
            return document
        for row in document.get("settings", []):
            if isinstance(row, dict) and row.get("domain") == "translation":
                payload = row.get("payload")
                if isinstance(payload, Mapping):
                    row["payload"] = self.apply_translation_setting(payload)
        return document

    def enforce_translation_command(self, command: Mapping[str, Any]) -> None:
        if not self.restricted():
            return
        if (
            command.get("executionMode") == "parallel"
            and not self.policy()["settings"]["parallel"]["allowed"]
        ):
            raise PublicPolicyDenied(
                "setting_locked",
                "管理员已关闭普通用户的并行模式",
            )

    def apply_resolved_translation(
        self,
        config: Mapping[str, Any],
        *,
        page_ids: Sequence[str] = (),
    ) -> dict[str, Any]:
        result = deepcopy(dict(config))
        if not self.restricted():
            return result
        policy = self.policy()
        parallel = policy["settings"]["parallel"]
        if not parallel["allowed"]:
            result["executionMode"] = "sequential"
        inpainting = result.get("inpainting")
        if isinstance(inpainting, Mapping):
            normalized_inpainting = dict(inpainting)
            lama = policy["settings"]["lamaDisableResize"]
            if not lama["editable"]:
                normalized_inpainting["disable_resize"] = lama["value"]
            result["inpainting"] = normalized_inpainting
        self._require_resolved_models(result)
        self._require_page_inpaint_models(page_ids, result.get("textStyleSnapshot"))
        return result

    def apply_page_repair_settings(self, value: Mapping[str, Any]) -> dict[str, Any]:
        result = deepcopy(dict(value))
        if not self.restricted():
            return result
        lama = self.policy()["settings"]["lamaDisableResize"]
        if not lama["editable"]:
            result["disableResize"] = lama["value"]
        return result

    def require_inpaint_method(self, method: object) -> None:
        model = _INPAINT_MODELS.get(str(method))
        if model is not None:
            self.require_model(model)

    def require_page_operation(self, kind: str, payload: Mapping[str, Any]) -> None:
        self.require_feature("editMode")
        if not self.restricted():
            return
        if kind == "page_detect":
            detector = payload.get("detector")
            self._require_detector(
                detector if isinstance(detector, Mapping) else payload
            )
        elif kind == "bubble_ocr":
            ocr = payload.get("ocr")
            self._require_ocr(ocr if isinstance(ocr, Mapping) else payload)

    def _require_resolved_models(self, config: Mapping[str, Any]) -> None:
        detector = config.get("detector")
        if isinstance(detector, Mapping):
            self._require_detector(detector)
        ocr = config.get("ocr")
        if isinstance(ocr, Mapping):
            self._require_ocr(ocr)

    def _require_detector(self, detector: Mapping[str, Any]) -> None:
        selected = _DETECTOR_MODELS.get(str(detector.get("detector_type")))
        if selected is not None:
            self.require_model(selected)
            # CTD/Yolo 只负责文本框，精确文字掩膜仍由 Default 生成。
            if selected != "detector_default":
                self.require_model("detector_default")
        if detector.get("enable_aux_yolo_detection") is True:
            self.require_model("aux_ysg_yolo")
        if detector.get("enable_saber_yolo_refine") is True:
            self.require_model("saber_yolo")

    def _require_ocr(self, ocr: Mapping[str, Any]) -> None:
        selected = _OCR_MODELS.get(str(ocr.get("ocr_engine")))
        if selected is not None:
            self.require_model(selected)
        if ocr.get("enable_hybrid_ocr") is True:
            secondary = _OCR_MODELS.get(str(ocr.get("secondary_ocr_engine")))
            if secondary is not None:
                self.require_model(secondary)

    def _require_page_inpaint_models(
        self,
        page_ids: Sequence[str],
        style_snapshot: object,
    ) -> None:
        if not self.restricted():
            return
        if isinstance(style_snapshot, Mapping):
            defaults = validate_page_style(
                style_snapshot.get("pageStyleDefaults"),
                partial=False,
            )
            self.require_inpaint_method(defaults["inpaintMethod"])
            return
        if not page_ids:
            return
        with self.engine.connect() as connection:
            values = list(
                connection.execute(
                    select(pages.c.page_style_defaults_json).where(
                        pages.c.id.in_(tuple(page_ids))
                    )
                ).scalars()
            )
        for value in values:
            defaults = validate_page_style(json.loads(str(value)), partial=False)
            self.require_inpaint_method(defaults["inpaintMethod"])
