"""Immutable factory text-style defaults.

Runtime settings are backend-v2 database facts. This module only reads the
bundled factory resource used when creating current settings and page state.
"""

from __future__ import annotations

import copy
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any


TEXT_STYLE_FACTORY_DEFAULTS_PATH = Path(__file__).with_name(
    "text_style_defaults_factory.json"
)

_REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "fontSize": int,
    "autoFontSize": bool,
    "fontFamily": str,
    "layoutDirection": str,
    "textColor": str,
    "fillColor": str,
    "inpaintMethod": str,
    "useAutoTextColor": bool,
    "strokeEnabled": bool,
    "strokeColor": str,
    "strokeWidth": (int, float),
    "lineSpacing": (int, float),
    "inlineAlign": str,
    "blockAlign": str,
}


def _validate_text_style_defaults(data: dict[str, Any]) -> dict[str, Any]:
    missing = set(_REQUIRED_FIELDS) - set(data)
    unknown = set(data) - set(_REQUIRED_FIELDS)
    if missing or unknown:
        raise RuntimeError(
            "text_style_defaults_factory.json 字段不匹配: "
            f"缺少 {sorted(missing)}，多余 {sorted(unknown)}"
        )

    for key, expected_type in _REQUIRED_FIELDS.items():
        value = data[key]
        if (
            isinstance(value, bool)
            and expected_type in {int, (int, float)}
        ) or not isinstance(value, expected_type):
            expected_name = (
                ", ".join(t.__name__ for t in expected_type)
                if isinstance(expected_type, tuple)
                else expected_type.__name__
            )
            raise RuntimeError(
                f"text_style_defaults_factory.json 字段 {key} 类型错误: "
                f"期望 {expected_name}, 实际 {type(value).__name__}"
            )

    if data["layoutDirection"] not in {"auto", "vertical", "horizontal"}:
        raise RuntimeError(
            "text_style_defaults_factory.json 的 layoutDirection 必须是 "
            "auto/vertical/horizontal"
        )
    if data["inpaintMethod"] not in {"solid", "lama_mpe", "litelama"}:
        raise RuntimeError(
            "text_style_defaults_factory.json 的 inpaintMethod 必须是 "
            "solid/lama_mpe/litelama"
        )
    for field in ("inlineAlign", "blockAlign"):
        if data[field] not in {"start", "center", "end"}:
            raise RuntimeError(
                f"text_style_defaults_factory.json 的 {field} 必须是 "
                "start/center/end"
            )
    if data["fontSize"] <= 0:
        raise RuntimeError("text_style_defaults_factory.json 的 fontSize 必须大于 0")
    if not math.isfinite(data["strokeWidth"]) or data["strokeWidth"] < 0:
        raise RuntimeError("text_style_defaults_factory.json 的 strokeWidth 必须为不小于 0 的有限数字")
    if float(data["lineSpacing"]) <= 0:
        raise RuntimeError("text_style_defaults_factory.json 的 lineSpacing 必须大于 0")

    normalized = dict(data)
    normalized["lineSpacing"] = float(normalized["lineSpacing"])
    return normalized


@lru_cache(maxsize=1)
def load_text_style_factory_defaults() -> dict[str, Any]:
    if not TEXT_STYLE_FACTORY_DEFAULTS_PATH.is_file():
        raise RuntimeError("text_style_defaults_factory.json 不存在，请检查安装文件是否完整")

    with TEXT_STYLE_FACTORY_DEFAULTS_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise RuntimeError("text_style_defaults_factory.json 必须是对象")

    return _validate_text_style_defaults(data)


def get_text_style_factory_defaults() -> dict[str, Any]:
    return copy.deepcopy(load_text_style_factory_defaults())
