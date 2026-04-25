"""
OCR 结果类型定义与辅助函数。

提供统一的 OCR 结果结构，供后端核心流程、API 路由和状态序列化复用。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _normalize_confidence(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if confidence < 0:
        return 0.0
    if confidence > 1:
        return 1.0
    return confidence


@dataclass
class OcrResult:
    text: str = ""
    confidence: Optional[float] = None
    confidence_supported: bool = False
    engine: str = ""
    primary_engine: str = ""
    fallback_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": _normalize_confidence(self.confidence),
            "confidenceSupported": bool(self.confidence_supported),
            "engine": self.engine,
            "primaryEngine": self.primary_engine or self.engine,
            "fallbackUsed": bool(self.fallback_used),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OcrResult":
        if not isinstance(data, dict):
            return cls()

        return cls(
            text=str(data.get("text", "") or ""),
            confidence=_normalize_confidence(data.get("confidence")),
            confidence_supported=bool(
                data.get("confidenceSupported", data.get("confidence_supported", False))
            ),
            engine=str(data.get("engine", "") or ""),
            primary_engine=str(
                data.get("primaryEngine", data.get("primary_engine", data.get("engine", ""))) or ""
            ),
            fallback_used=bool(
                data.get("fallbackUsed", data.get("fallback_used", False))
            ),
        )


@dataclass
class OcrTextlineResult:
    text: str = ""
    confidence: Optional[float] = None
    confidence_supported: bool = False
    engine: str = ""
    primary_engine: str = ""
    fallback_used: bool = False
    polygon: Optional[List[List[int]]] = None
    direction: str = "h"
    fg_color: Optional[Tuple[int, int, int]] = None
    bg_color: Optional[Tuple[int, int, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": _normalize_confidence(self.confidence),
            "confidenceSupported": bool(self.confidence_supported),
            "engine": self.engine,
            "primaryEngine": self.primary_engine or self.engine,
            "fallbackUsed": bool(self.fallback_used),
            "polygon": self.polygon or [],
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
        text=str(text or ""),
        confidence=_normalize_confidence(confidence),
        confidence_supported=bool(confidence_supported),
        engine=engine,
        primary_engine=primary_engine or engine,
        fallback_used=bool(fallback_used),
    )


def create_ocr_textline_result(
    text: str,
    engine: str,
    *,
    confidence: Optional[float] = None,
    confidence_supported: bool = False,
    primary_engine: Optional[str] = None,
    fallback_used: bool = False,
    polygon: Optional[List[List[int]]] = None,
    direction: str = "h",
    fg_color: Optional[Tuple[int, int, int]] = None,
    bg_color: Optional[Tuple[int, int, int]] = None,
) -> OcrTextlineResult:
    return OcrTextlineResult(
        text=str(text or ""),
        confidence=_normalize_confidence(confidence),
        confidence_supported=bool(confidence_supported),
        engine=engine,
        primary_engine=primary_engine or engine,
        fallback_used=bool(fallback_used),
        polygon=polygon or [],
        direction=direction,
        fg_color=fg_color,
        bg_color=bg_color,
    )


def ocr_results_to_dicts(results: List[OcrResult]) -> List[Dict[str, Any]]:
    return [result.to_dict() for result in results]


def extract_texts_from_ocr_results(results: List[OcrResult]) -> List[str]:
    return [str(result.text or "") for result in results]
