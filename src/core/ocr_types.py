"""Strict current OCR result models shared by Worker algorithms and rendering."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


OCR_RESULT_FIELDS = frozenset(
    {
        "text",
        "confidence",
        "confidenceSupported",
        "engine",
        "primaryEngine",
        "fallbackUsed",
    }
)


def _confidence(value: object, *, supported: bool) -> Optional[float]:
    if value is None:
        if supported:
            raise ValueError("supported OCR confidence cannot be null")
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
    ):
        raise ValueError("OCR confidence must be between zero and one")
    if not supported:
        raise ValueError("unsupported OCR confidence must be null")
    return float(value)


def _non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _polygon(value: object) -> list[list[int]]:
    if not isinstance(value, list):
        raise ValueError("OCR textline polygon must be an array")
    if not value:
        return []
    if len(value) != 4:
        raise ValueError("OCR textline polygon must contain four points")
    result: list[list[int]] = []
    for point in value:
        if (
            not isinstance(point, list)
            or len(point) != 2
            or any(isinstance(part, bool) or not isinstance(part, int) for part in point)
        ):
            raise ValueError("OCR textline polygon points must be integer pairs")
        result.append(list(point))
    return result


def _rgb(value: object, *, field_name: str) -> Optional[tuple[int, int, int]]:
    if value is None:
        return None
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{field_name} must be an RGB tuple")
    if any(
        isinstance(channel, bool)
        or not isinstance(channel, int)
        or not 0 <= channel <= 255
        for channel in value
    ):
        raise ValueError(f"{field_name} channels must be integers from 0 to 255")
    return value


@dataclass
class OcrResult:
    text: str
    confidence: Optional[float]
    confidence_supported: bool
    engine: str
    primary_engine: str
    fallback_used: bool

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("OCR text must be a string")
        if not isinstance(self.confidence_supported, bool):
            raise TypeError("OCR confidenceSupported must be boolean")
        if not isinstance(self.fallback_used, bool):
            raise TypeError("OCR fallbackUsed must be boolean")
        self.confidence = _confidence(
            self.confidence,
            supported=self.confidence_supported,
        )
        self.engine = _non_empty_string(self.engine, field_name="OCR engine")
        self.primary_engine = _non_empty_string(
            self.primary_engine,
            field_name="OCR primaryEngine",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "confidenceSupported": self.confidence_supported,
            "engine": self.engine,
            "primaryEngine": self.primary_engine,
            "fallbackUsed": self.fallback_used,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OcrResult":
        if not isinstance(data, Mapping) or set(data) != OCR_RESULT_FIELDS:
            raise ValueError("OCR result does not match the current schema")
        return cls(
            text=data["text"],
            confidence=data["confidence"],
            confidence_supported=data["confidenceSupported"],
            engine=data["engine"],
            primary_engine=data["primaryEngine"],
            fallback_used=data["fallbackUsed"],
        )


@dataclass
class OcrTextlineResult(OcrResult):
    polygon: list[list[int]] = field(default_factory=list)
    direction: str = "h"
    fg_color: Optional[tuple[int, int, int]] = None
    bg_color: Optional[tuple[int, int, int]] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.polygon = _polygon(self.polygon)
        if self.direction not in {"h", "v"}:
            raise ValueError("OCR textline direction must be h or v")
        self.fg_color = _rgb(self.fg_color, field_name="OCR foreground color")
        self.bg_color = _rgb(self.bg_color, field_name="OCR background color")

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "polygon": [list(point) for point in self.polygon],
            "direction": self.direction,
            "fgColor": list(self.fg_color) if self.fg_color is not None else None,
            "bgColor": list(self.bg_color) if self.bg_color is not None else None,
        }


def create_ocr_result(
    text: str,
    engine: str,
    *,
    confidence: Optional[float] = None,
    confidence_supported: bool = False,
    primary_engine: Optional[str] = None,
    fallback_used: bool = False,
) -> OcrResult:
    return OcrResult(
        text=text,
        confidence=confidence,
        confidence_supported=confidence_supported,
        engine=engine,
        primary_engine=engine if primary_engine is None else primary_engine,
        fallback_used=fallback_used,
    )


def create_ocr_textline_result(
    text: str,
    engine: str,
    *,
    confidence: Optional[float] = None,
    confidence_supported: bool = False,
    primary_engine: Optional[str] = None,
    fallback_used: bool = False,
    polygon: Optional[list[list[int]]] = None,
    direction: str = "h",
    fg_color: Optional[tuple[int, int, int]] = None,
    bg_color: Optional[tuple[int, int, int]] = None,
) -> OcrTextlineResult:
    return OcrTextlineResult(
        text=text,
        confidence=confidence,
        confidence_supported=confidence_supported,
        engine=engine,
        primary_engine=engine if primary_engine is None else primary_engine,
        fallback_used=fallback_used,
        polygon=[] if polygon is None else polygon,
        direction=direction,
        fg_color=fg_color,
        bg_color=bg_color,
    )


def ocr_results_to_dicts(results: list[OcrResult]) -> list[dict[str, Any]]:
    if not isinstance(results, list) or any(
        not isinstance(result, OcrResult) for result in results
    ):
        raise TypeError("OCR results must be an OcrResult list")
    return [result.to_dict() for result in results]


def extract_texts_from_ocr_results(results: list[OcrResult]) -> list[str]:
    if not isinstance(results, list) or any(
        not isinstance(result, OcrResult) for result in results
    ):
        raise TypeError("OCR results must be an OcrResult list")
    return [result.text for result in results]
