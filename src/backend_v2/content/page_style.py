"""Canonical page text-style validation and new-page default resolution."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping

from sqlalchemy import select
from sqlalchemy.engine import Connection

from src.backend_v2.storage.defaults import TEXT_STYLE_DEFAULTS_SCHEMA_VERSION
from src.backend_v2.storage.schema import app_settings, fonts


PAGE_STYLE_FIELDS = frozenset(
    {
        "fontSize",
        "autoFontSize",
        "layoutDirection",
        "textColor",
        "fillColor",
        "inpaintMethod",
        "useAutoTextColor",
        "strokeEnabled",
        "strokeColor",
        "strokeWidth",
        "lineSpacing",
        "inlineAlign",
        "blockAlign",
    }
)
PAGE_STYLE_SCHEMA_VERSION = 2
TEXT_STYLE_DEFAULT_FIELDS = PAGE_STYLE_FIELDS | {"fontFamily"}
_COLOR_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _integer(
    value: object,
    *,
    field: str,
    minimum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ValueError(f"{field} must be an integer of at least {minimum}")
    return value


def _number(
    value: object,
    *,
    field: str,
    exclusive_minimum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be greater than {exclusive_minimum}")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= exclusive_minimum:
        raise ValueError(f"{field} must be greater than {exclusive_minimum}")
    return normalized


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _choice(value: object, *, field: str, choices: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{field} must be one of {', '.join(sorted(choices))}")
    return value


def _color(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _COLOR_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a #RRGGBB color")
    return value


def rgb_to_hex(value: object) -> str:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("RGB color must contain exactly three channels")
    channels: list[int] = []
    for part in value:
        if isinstance(part, bool) or not isinstance(part, int) or not 0 <= part <= 255:
            raise ValueError("RGB color channels must be integers from 0 to 255")
        channels.append(part)
    red, green, blue = channels
    return f"#{red:02X}{green:02X}{blue:02X}"


def validate_page_style(
    value: object,
    *,
    partial: bool,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("page text style must be an object")
    result = dict(value)
    unknown = set(result) - PAGE_STYLE_FIELDS
    if unknown:
        raise ValueError(
            "unknown page style fields: " + ", ".join(sorted(unknown))
        )
    if not partial and set(result) != PAGE_STYLE_FIELDS:
        missing = PAGE_STYLE_FIELDS - set(result)
        raise ValueError(
            "page text style is missing fields: " + ", ".join(sorted(missing))
        )
    if "fontSize" in result:
        result["fontSize"] = _integer(
            result["fontSize"],
            field="fontSize",
            minimum=1,
        )
    if "autoFontSize" in result:
        result["autoFontSize"] = _boolean(
            result["autoFontSize"],
            field="autoFontSize",
        )
    if "layoutDirection" in result:
        result["layoutDirection"] = _choice(
            result["layoutDirection"],
            field="layoutDirection",
            choices=frozenset({"auto", "vertical", "horizontal"}),
        )
    for field in ("textColor", "fillColor", "strokeColor"):
        if field in result:
            result[field] = _color(result[field], field=field)
    if "inpaintMethod" in result:
        result["inpaintMethod"] = _choice(
            result["inpaintMethod"],
            field="inpaintMethod",
            choices=frozenset({"solid", "lama_mpe", "litelama"}),
        )
    for field in ("useAutoTextColor", "strokeEnabled"):
        if field in result:
            result[field] = _boolean(result[field], field=field)
    if "strokeWidth" in result:
        result["strokeWidth"] = _integer(
            result["strokeWidth"],
            field="strokeWidth",
            minimum=0,
        )
    if "lineSpacing" in result:
        result["lineSpacing"] = _number(
            result["lineSpacing"],
            field="lineSpacing",
            exclusive_minimum=0,
        )
    for field in ("inlineAlign", "blockAlign"):
        if field in result:
            result[field] = _choice(
                result[field],
                field=field,
                choices=frozenset({"start", "center", "end"}),
            )
    return result


def validate_text_style_defaults(
    connection: Connection,
    value: object,
) -> tuple[str, dict[str, object]]:
    if not isinstance(value, Mapping):
        raise ValueError("text_style_defaults must be an object")
    payload = dict(value)
    unknown = set(payload) - TEXT_STYLE_DEFAULT_FIELDS
    if unknown:
        raise ValueError(
            "unknown text_style_defaults fields: " + ", ".join(sorted(unknown))
        )
    missing = TEXT_STYLE_DEFAULT_FIELDS - set(payload)
    if missing:
        raise ValueError(
            "text_style_defaults is missing fields: " + ", ".join(sorted(missing))
        )
    font_id = payload.pop("fontFamily")
    if not isinstance(font_id, str) or not font_id:
        raise ValueError("fontFamily must be a font ID")
    if connection.execute(
        select(fonts.c.id).where(fonts.c.id == font_id)
    ).scalar_one_or_none() is None:
        raise ValueError("fontFamily does not reference an existing font")
    return font_id, validate_page_style(payload, partial=False)


def resolve_new_page_style(
    connection: Connection,
) -> tuple[str, dict[str, object]]:
    setting = connection.execute(
        select(
            app_settings.c.payload_json,
            app_settings.c.schema_version,
        ).where(
            app_settings.c.domain == "text_style_defaults"
        )
    ).mappings().one_or_none()
    if setting is None:
        raise ValueError("text_style_defaults setting is missing")
    if setting["schema_version"] != TEXT_STYLE_DEFAULTS_SCHEMA_VERSION:
        raise ValueError("text_style_defaults schema version is not current")
    payload: Any = json.loads(setting["payload_json"])
    return validate_text_style_defaults(connection, payload)
