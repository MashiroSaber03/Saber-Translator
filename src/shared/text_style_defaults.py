"""
文字样式默认值加载器

从 config/text_style_defaults.json 读取唯一真源，并提供前后端共用的
默认值访问与最小校验能力。
"""

from __future__ import annotations

import copy
import json
import os
from functools import lru_cache
from typing import Any, Dict


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEXT_STYLE_DEFAULTS_PATH = os.path.join(PROJECT_ROOT, "config", "text_style_defaults.json")

_BOOTSTRAP_TEXT_STYLE_DEFAULTS: Dict[str, Any] = {
    "fontSize": 25,
    "autoFontSize": False,
    "fontFamily": "fonts/思源黑体SourceHanSansK-Bold.TTF",
    "layoutDirection": "auto",
    "textColor": "#000000",
    "fillColor": "#FFFFFF",
    "inpaintMethod": "solid",
    "useAutoTextColor": False,
    "strokeEnabled": True,
    "strokeColor": "#FFFFFF",
    "strokeWidth": 3,
    "lineSpacing": 1.0,
    "textAlign": "start",
}

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
    "strokeWidth": int,
    "lineSpacing": (int, float),
    "textAlign": str,
}


def _validate_text_style_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
    missing = [key for key in _REQUIRED_FIELDS if key not in data]
    if missing:
        raise RuntimeError(f"text_style_defaults.json 缺少字段: {', '.join(missing)}")

    for key, expected_type in _REQUIRED_FIELDS.items():
        value = data[key]
        if not isinstance(value, expected_type):
            expected_name = (
                ", ".join(t.__name__ for t in expected_type)
                if isinstance(expected_type, tuple)
                else expected_type.__name__
            )
            raise RuntimeError(
                f"text_style_defaults.json 字段 {key} 类型错误: 期望 {expected_name}, 实际 {type(value).__name__}"
            )

    if data["layoutDirection"] not in {"auto", "vertical", "horizontal"}:
        raise RuntimeError("text_style_defaults.json 的 layoutDirection 必须是 auto/vertical/horizontal")
    if data["inpaintMethod"] not in {"solid", "lama_mpe", "litelama"}:
        raise RuntimeError("text_style_defaults.json 的 inpaintMethod 必须是 solid/lama_mpe/litelama")
    if data["textAlign"] not in {"start", "center", "end"}:
        raise RuntimeError("text_style_defaults.json 的 textAlign 必须是 start/center/end")
    if data["fontSize"] <= 0:
        raise RuntimeError("text_style_defaults.json 的 fontSize 必须大于 0")
    if data["strokeWidth"] < 0:
        raise RuntimeError("text_style_defaults.json 的 strokeWidth 不能小于 0")
    if float(data["lineSpacing"]) <= 0:
        raise RuntimeError("text_style_defaults.json 的 lineSpacing 必须大于 0")

    normalized = dict(data)
    normalized["lineSpacing"] = float(normalized["lineSpacing"])
    return normalized


def ensure_text_style_defaults_file() -> None:
    if os.path.exists(TEXT_STYLE_DEFAULTS_PATH):
        return

    os.makedirs(os.path.dirname(TEXT_STYLE_DEFAULTS_PATH), exist_ok=True)
    with open(TEXT_STYLE_DEFAULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(_BOOTSTRAP_TEXT_STYLE_DEFAULTS, file, indent=2, ensure_ascii=False)


@lru_cache(maxsize=1)
def load_text_style_defaults() -> Dict[str, Any]:
    ensure_text_style_defaults_file()

    with open(TEXT_STYLE_DEFAULTS_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise RuntimeError("text_style_defaults.json 必须是对象")

    return _validate_text_style_defaults(data)


def get_text_style_defaults() -> Dict[str, Any]:
    return copy.deepcopy(load_text_style_defaults())


def reload_text_style_defaults() -> Dict[str, Any]:
    load_text_style_defaults.cache_clear()
    return get_text_style_defaults()


def get_backend_default_text_direction() -> str:
    layout_direction = load_text_style_defaults()["layoutDirection"]
    return layout_direction if layout_direction in {"vertical", "horizontal"} else "vertical"


def get_backend_default_font_relative_path() -> str:
    font_family = load_text_style_defaults()["fontFamily"]
    normalized_font_path = font_family.replace("/", os.sep)
    return os.path.join("src", "app", "static", normalized_font_path)
