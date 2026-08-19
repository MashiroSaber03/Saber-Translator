"""Strict current bubble models shared by persistence and rendering."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import re
from typing import Any

from src.core.ocr_types import OcrResult
from src.shared import constants


BUBBLE_PAYLOAD_SCHEMA_VERSION = 2
STORED_BUBBLE_FIELDS = frozenset(
    {
        "originalText",
        "translatedText",
        "textboxText",
        "coords",
        "polygon",
        "fontSize",
        "textDirection",
        "autoTextDirection",
        "textColor",
        "fillColor",
        "rotationAngle",
        "position",
        "strokeEnabled",
        "strokeColor",
        "strokeWidth",
        "lineSpacing",
        "inlineAlign",
        "blockAlign",
        "inpaintMethod",
        "autoFgColor",
        "autoBgColor",
        "colorConfidence",
        "textlines",
        "ocrResult",
    }
)
RENDER_BUBBLE_FIELDS = STORED_BUBBLE_FIELDS | {"fontFamily"}
BUBBLE_TEXTLINE_FIELDS = frozenset({"polygon", "direction", "confidence"})
_COLOR_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _finite_number(value: object, *, field_name: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field_name} must be a finite number")
    return value


def _integer(
    value: object,
    *,
    field_name: str,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")
    return value


def _confidence(value: object, *, field_name: str) -> float:
    number = _finite_number(value, field_name=field_name)
    if not 0 <= number <= 1:
        raise ValueError(f"{field_name} must be between zero and one")
    return float(number)


def _string(value: object, *, field_name: str, non_empty: bool = False) -> str:
    if not isinstance(value, str) or (non_empty and not value):
        requirement = "a non-empty string" if non_empty else "a string"
        raise ValueError(f"{field_name} must be {requirement}")
    return value


def _choice(
    value: object,
    *,
    field_name: str,
    choices: frozenset[str],
) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(
            f"{field_name} must be one of {', '.join(sorted(choices))}"
        )
    return value


def _boolean(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be boolean")
    return value


def _color(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _COLOR_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a #RRGGBB color")
    return value


def _integer_polygon(
    value: object,
    *,
    field_name: str,
    allow_empty: bool,
) -> list[list[int]]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    if not value and allow_empty:
        return []
    if len(value) != 4:
        raise ValueError(f"{field_name} must contain four points")
    polygon: list[list[int]] = []
    for point in value:
        if (
            not isinstance(point, list)
            or len(point) != 2
            or any(isinstance(part, bool) or not isinstance(part, int) for part in point)
        ):
            raise ValueError(f"{field_name} points must be integer pairs")
        polygon.append(list(point))
    return polygon


def _coords(value: object) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(part, bool) or not isinstance(part, int) for part in value)
    ):
        raise ValueError("coords must contain four integers")
    x1, y1, x2, y2 = value
    if x1 >= x2 or y1 >= y2:
        raise ValueError("coords must describe a positive-area box")
    return list(value)


def _position(value: object) -> dict[str, int | float]:
    if not isinstance(value, Mapping) or set(value) != {"x", "y"}:
        raise ValueError("position must contain exactly x and y")
    return {
        "x": _finite_number(value["x"], field_name="position.x"),
        "y": _finite_number(value["y"], field_name="position.y"),
    }


def _rgb(value: object, *, field_name: str) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field_name} must be null or an RGB array")
    if any(
        isinstance(channel, bool)
        or not isinstance(channel, int)
        or not 0 <= channel <= 255
        for channel in value
    ):
        raise ValueError(f"{field_name} channels must be integers from 0 to 255")
    return list(value)


@dataclass
class BubbleTextline:
    polygon: list[list[int]] = field(default_factory=list)
    direction: str = "h"
    confidence: float = 0.0

    def __post_init__(self) -> None:
        self.polygon = _integer_polygon(
            self.polygon,
            field_name="textline.polygon",
            allow_empty=False,
        )
        self.direction = _choice(
            self.direction,
            field_name="textline.direction",
            choices=frozenset({"h", "v"}),
        )
        self.confidence = _confidence(
            self.confidence,
            field_name="textline.confidence",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "polygon": [list(point) for point in self.polygon],
            "direction": self.direction,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BubbleTextline":
        if not isinstance(data, Mapping) or set(data) != BUBBLE_TEXTLINE_FIELDS:
            raise ValueError("bubble textline does not match the current schema")
        return cls(
            polygon=data["polygon"],
            direction=data["direction"],
            confidence=data["confidence"],
        )


def _textlines(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("textlines must be an array")
    return [BubbleTextline.from_dict(item).to_dict() for item in value]


def _ocr_result(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("ocrResult must be null or an object")
    return OcrResult.from_dict(value).to_dict()


def validate_bubble_payload(
    value: object,
    *,
    render: bool,
    partial: bool = False,
) -> dict[str, Any]:
    """Validate a current stored or materialized-render bubble payload."""

    if not isinstance(value, Mapping):
        raise ValueError("bubble payload must be an object")
    fields = RENDER_BUBBLE_FIELDS if render else STORED_BUBBLE_FIELDS
    keys = set(value)
    unknown = keys - fields
    if unknown:
        raise ValueError(
            "unknown bubble payload fields: " + ", ".join(sorted(unknown))
        )
    if not partial and keys != fields:
        missing = fields - keys
        raise ValueError(
            "bubble payload is missing fields: " + ", ".join(sorted(missing))
        )

    result = dict(value)
    for name in ("originalText", "translatedText", "textboxText"):
        if name in result:
            result[name] = _string(result[name], field_name=name)
    if "coords" in result:
        result["coords"] = _coords(result["coords"])
    if "polygon" in result:
        result["polygon"] = _integer_polygon(
            result["polygon"],
            field_name="polygon",
            allow_empty=True,
        )
    if "fontSize" in result:
        result["fontSize"] = _integer(
            result["fontSize"],
            field_name="fontSize",
            minimum=1,
        )
    if "fontFamily" in result:
        result["fontFamily"] = _string(
            result["fontFamily"],
            field_name="fontFamily",
            non_empty=True,
        )
    for name in ("textDirection", "autoTextDirection"):
        if name in result:
            result[name] = _choice(
                result[name],
                field_name=name,
                choices=frozenset({"horizontal", "vertical"}),
            )
    for name in ("textColor", "fillColor", "strokeColor"):
        if name in result:
            result[name] = _color(result[name], field_name=name)
    if "rotationAngle" in result:
        result["rotationAngle"] = _finite_number(
            result["rotationAngle"],
            field_name="rotationAngle",
        )
    if "position" in result:
        result["position"] = _position(result["position"])
    if "strokeEnabled" in result:
        result["strokeEnabled"] = _boolean(
            result["strokeEnabled"],
            field_name="strokeEnabled",
        )
    if "strokeWidth" in result:
        result["strokeWidth"] = _integer(
            result["strokeWidth"],
            field_name="strokeWidth",
            minimum=0,
        )
    if "lineSpacing" in result:
        line_spacing = _finite_number(
            result["lineSpacing"],
            field_name="lineSpacing",
        )
        if line_spacing <= 0:
            raise ValueError("lineSpacing must be greater than zero")
        result["lineSpacing"] = float(line_spacing)
    for name in ("inlineAlign", "blockAlign"):
        if name in result:
            result[name] = _choice(
                result[name],
                field_name=name,
                choices=frozenset({"center", "end", "start"}),
            )
    if "inpaintMethod" in result:
        result["inpaintMethod"] = _choice(
            result["inpaintMethod"],
            field_name="inpaintMethod",
            choices=frozenset({"lama_mpe", "litelama", "solid"}),
        )
    for name in ("autoFgColor", "autoBgColor"):
        if name in result:
            result[name] = _rgb(result[name], field_name=name)
    if "colorConfidence" in result:
        result["colorConfidence"] = _confidence(
            result["colorConfidence"],
            field_name="colorConfidence",
        )
    if "textlines" in result:
        result["textlines"] = _textlines(result["textlines"])
    if "ocrResult" in result:
        result["ocrResult"] = _ocr_result(result["ocrResult"])
    return result


@dataclass
class BubbleState:
    original_text: str = ""
    translated_text: str = ""
    textbox_text: str = ""
    coords: tuple[int, int, int, int] = (0, 0, 100, 100)
    polygon: list[list[int]] = field(default_factory=list)
    font_size: int = constants.DEFAULT_FONT_SIZE
    font_family: str = constants.DEFAULT_FONT_RELATIVE_PATH
    text_direction: str = constants.DEFAULT_TEXT_DIRECTION
    auto_text_direction: str = constants.DEFAULT_TEXT_DIRECTION
    text_color: str = constants.DEFAULT_TEXT_COLOR
    fill_color: str = constants.DEFAULT_FILL_COLOR
    rotation_angle: int | float = constants.DEFAULT_ROTATION_ANGLE
    position_offset: dict[str, int | float] = field(
        default_factory=lambda: {"x": 0, "y": 0}
    )
    stroke_enabled: bool = constants.DEFAULT_STROKE_ENABLED
    stroke_color: str = constants.DEFAULT_STROKE_COLOR
    stroke_width: int = constants.DEFAULT_STROKE_WIDTH
    line_spacing: float = constants.DEFAULT_LINE_SPACING
    inline_align: str = constants.DEFAULT_INLINE_ALIGN
    block_align: str = constants.DEFAULT_BLOCK_ALIGN
    inpaint_method: str = constants.DEFAULT_INPAINT_METHOD
    auto_fg_color: tuple[int, int, int] | None = None
    auto_bg_color: tuple[int, int, int] | None = None
    color_confidence: float = 0.0
    textlines: list[BubbleTextline] = field(default_factory=list)
    ocr_result: OcrResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "originalText": self.original_text,
            "translatedText": self.translated_text,
            "textboxText": self.textbox_text,
            "coords": list(self.coords),
            "polygon": [list(point) for point in self.polygon],
            "fontSize": self.font_size,
            "fontFamily": self.font_family,
            "textDirection": self.text_direction,
            "autoTextDirection": self.auto_text_direction,
            "textColor": self.text_color,
            "fillColor": self.fill_color,
            "rotationAngle": self.rotation_angle,
            "position": dict(self.position_offset),
            "strokeEnabled": self.stroke_enabled,
            "strokeColor": self.stroke_color,
            "strokeWidth": self.stroke_width,
            "lineSpacing": self.line_spacing,
            "inlineAlign": self.inline_align,
            "blockAlign": self.block_align,
            "inpaintMethod": self.inpaint_method,
            "autoFgColor": list(self.auto_fg_color) if self.auto_fg_color else None,
            "autoBgColor": list(self.auto_bg_color) if self.auto_bg_color else None,
            "colorConfidence": self.color_confidence,
            "textlines": [textline.to_dict() for textline in self.textlines],
            "ocrResult": self.ocr_result.to_dict() if self.ocr_result else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BubbleState":
        payload = validate_bubble_payload(data, render=True)
        return cls(
            original_text=payload["originalText"],
            translated_text=payload["translatedText"],
            textbox_text=payload["textboxText"],
            coords=tuple(payload["coords"]),
            polygon=payload["polygon"],
            font_size=payload["fontSize"],
            font_family=payload["fontFamily"],
            text_direction=payload["textDirection"],
            auto_text_direction=payload["autoTextDirection"],
            text_color=payload["textColor"],
            fill_color=payload["fillColor"],
            rotation_angle=payload["rotationAngle"],
            position_offset=payload["position"],
            stroke_enabled=payload["strokeEnabled"],
            stroke_color=payload["strokeColor"],
            stroke_width=payload["strokeWidth"],
            line_spacing=payload["lineSpacing"],
            inline_align=payload["inlineAlign"],
            block_align=payload["blockAlign"],
            inpaint_method=payload["inpaintMethod"],
            auto_fg_color=(
                tuple(payload["autoFgColor"])
                if payload["autoFgColor"] is not None
                else None
            ),
            auto_bg_color=(
                tuple(payload["autoBgColor"])
                if payload["autoBgColor"] is not None
                else None
            ),
            color_confidence=payload["colorConfidence"],
            textlines=[
                BubbleTextline.from_dict(item) for item in payload["textlines"]
            ],
            ocr_result=(
                OcrResult.from_dict(payload["ocrResult"])
                if payload["ocrResult"] is not None
                else None
            ),
        )
