"""
MangaOCR / 48px 专用复合 OCR 适配器。

首批重构仅为这组组合提供上游对齐型 textline 级混合 OCR。
"""

import math
import logging
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from PIL import Image

from src.core.ocr_types import OcrResult, OcrTextlineResult, create_ocr_result, create_ocr_textline_result
from src.interfaces.manga_ocr_interface import recognize_japanese_text
from src.interfaces.ocr_48px import get_48px_ocr_handler
from src.interfaces.ocr_48px.interface import get_transformed_region
from src.shared import constants


logger = logging.getLogger("HybridOcrManga48")

SUPPORTED_HYBRID_ENGINES = frozenset({constants.OCR_ENGINE_48PX, 'manga_ocr'})
DEFAULT_HYBRID_THRESHOLD = 0.2


def is_supported_manga_48_hybrid(primary_engine: str, secondary_engine: str) -> bool:
    return {primary_engine, secondary_engine} == SUPPORTED_HYBRID_ENGINES


def validate_manga_48_hybrid_combo(primary_engine: str, secondary_engine: str) -> None:
    if primary_engine == secondary_engine:
        raise ValueError("混合OCR要求主OCR与备用OCR不同")
    if not is_supported_manga_48_hybrid(primary_engine, secondary_engine):
        raise ValueError("首批混合OCR仅支持 MangaOCR / 48px OCR 组合")


def _polygon_bounds(polygon: Sequence[Sequence[int]]) -> tuple[int, int, int, int]:
    pts = np.array(polygon, dtype=np.int32)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return int(x1), int(y1), int(x2), int(y2)


def _polygon_area(polygon: Sequence[Sequence[int]]) -> float:
    pts = np.array(polygon, dtype=np.float32)
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _mangaocr_textline_height(line_info: Dict[str, Any]) -> int:
    x1, y1, x2, y2 = _polygon_bounds(line_info.get('polygon', []))
    direction = line_info.get('direction', 'h')
    if direction == 'h':
        return max(x2 - x1, 2)
    return max(y2 - y1, 2)


def _get_mangaocr_region(image_np: np.ndarray, line_info: Dict[str, Any]) -> np.ndarray:
    polygon = line_info.get('polygon', [])
    pts = np.array(polygon, dtype=np.float32)
    target_height = _mangaocr_textline_height(line_info)
    # 对齐上游 mocr：无论原方向如何，都使用 horizontal 输出给 MangaOCR
    return get_transformed_region(image_np, pts, 'h', target_height=target_height)


def _recognize_manga_textlines(
    image: Image.Image,
    textlines: List[Dict[str, Any]],
    *,
    primary_engine: str,
    fallback_used: bool,
) -> List[OcrTextlineResult]:
    if not textlines:
        return []

    img_np = np.array(image.convert('RGB'))
    results: List[OcrTextlineResult] = []

    for line_info in textlines:
        polygon = line_info.get('polygon', [])
        direction = line_info.get('direction', 'h')
        if not polygon or len(polygon) != 4:
            results.append(
                create_ocr_textline_result(
                    "",
                    'manga_ocr',
                    confidence=None,
                    confidence_supported=False,
                    primary_engine=primary_engine,
                    fallback_used=fallback_used,
                    polygon=polygon,
                    direction=direction,
                )
            )
            continue

        try:
            region = _get_mangaocr_region(img_np, line_info)
            text = recognize_japanese_text(Image.fromarray(region))
        except Exception:
            text = ""

        results.append(
            create_ocr_textline_result(
                text,
                'manga_ocr',
                confidence=None,
                confidence_supported=False,
                primary_engine=primary_engine,
                fallback_used=fallback_used,
                polygon=[list(map(int, point)) for point in polygon],
                direction=direction,
            )
        )

    return results


def _aggregate_bubble_result(
    line_results: List[OcrTextlineResult],
    primary_engine: str,
    secondary_engine: str,
) -> OcrResult:
    non_empty_lines = [line for line in line_results if str(line.text or "").strip()]
    if not non_empty_lines:
        return create_ocr_result(
            "",
            primary_engine,
            confidence=0.0,
            confidence_supported=True,
            primary_engine=primary_engine,
            fallback_used=any(line.fallback_used for line in line_results),
        )

    text = " ".join(line.text for line in non_empty_lines)
    total_area = 0.0
    total_logprob = 0.0
    valid_probability_found = False

    for line in non_empty_lines:
        polygon = line.polygon or []
        area = _polygon_area(polygon)
        if area <= 0:
            area = 1.0

        if line.confidence_supported and line.confidence is not None and line.confidence > 0:
            total_logprob += math.log(max(float(line.confidence), 1e-6)) * area
            total_area += area
            valid_probability_found = True

    confidence = math.exp(total_logprob / total_area) if valid_probability_found and total_area > 0 else 0.0
    fallback_used = any(line.fallback_used for line in non_empty_lines)
    final_engine = secondary_engine if fallback_used else primary_engine

    return create_ocr_result(
        text,
        final_engine,
        confidence=confidence,
        confidence_supported=True,
        primary_engine=primary_engine,
        fallback_used=fallback_used,
    )


def recognize_manga_48_hybrid(
    image: Image.Image,
    bubble_coords: List[tuple[int, int, int, int]],
    textlines_per_bubble: List[List[Dict[str, Any]]],
    *,
    primary_engine: str,
    secondary_engine: str,
    threshold: float = DEFAULT_HYBRID_THRESHOLD,
) -> List[OcrResult]:
    validate_manga_48_hybrid_combo(primary_engine, secondary_engine)
    logger.info(
        "混合OCR专用链路启动: %s -> %s, 气泡=%d",
        primary_engine,
        secondary_engine,
        len(bubble_coords),
    )

    ocr48_handler = get_48px_ocr_handler()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not ocr48_handler.initialize(device):
        raise RuntimeError("48px OCR 初始化失败")

    hybrid_results: List[OcrResult] = []

    for bubble_index, bubble_coords_item in enumerate(bubble_coords):
        textlines = textlines_per_bubble[bubble_index] if bubble_index < len(textlines_per_bubble) else []

        if not textlines:
            # 无文本行时退化为整块 48px or Manga 识别，但仍保持当前组合语义。
            x1, y1, x2, y2 = bubble_coords_item
            bubble_image = image.crop((x1, y1, x2, y2))
            if primary_engine == constants.OCR_ENGINE_48PX:
                bubble_results = ocr48_handler.recognize_text_with_details(
                    bubble_image,
                    [(0, 0, bubble_image.width, bubble_image.height)],
                    None,
                    primary_engine=primary_engine,
                    fallback_used=False,
                )
                hybrid_results.append(bubble_results[0] if bubble_results else create_ocr_result("", primary_engine, confidence=0.0, confidence_supported=True, primary_engine=primary_engine))
            else:
                text = recognize_japanese_text(bubble_image)
                hybrid_results.append(
                    create_ocr_result(
                        text,
                        primary_engine,
                        primary_engine=primary_engine,
                    )
                )
            continue

        ocr48_lines = ocr48_handler.recognize_textlines_with_details(
            image,
            textlines,
            primary_engine=primary_engine,
            fallback_used=False,
        )

        if primary_engine == constants.OCR_ENGINE_48PX:
            manga_retry_lines = [
                textlines[index]
                for index, line in enumerate(ocr48_lines)
                if not str(line.text or "").strip() or float(line.confidence or 0.0) < threshold
            ]
            manga_retry_results = _recognize_manga_textlines(
                image,
                manga_retry_lines,
                primary_engine=primary_engine,
                fallback_used=True,
            )

            retry_cursor = 0
            final_lines: List[OcrTextlineResult] = []
            for line in ocr48_lines:
                needs_retry = (not str(line.text or "").strip()) or float(line.confidence or 0.0) < threshold
                if needs_retry:
                    fallback_line = manga_retry_results[retry_cursor]
                    retry_cursor += 1
                    if str(fallback_line.text or "").strip():
                        line.text = fallback_line.text
                        line.engine = secondary_engine
                        line.fallback_used = True
                final_lines.append(line)
        else:
            manga_primary_lines = _recognize_manga_textlines(
                image,
                textlines,
                primary_engine=primary_engine,
                fallback_used=False,
            )
            final_lines = []
            for index, manga_line in enumerate(manga_primary_lines):
                line_48 = ocr48_lines[index]
                combined_line = create_ocr_textline_result(
                    manga_line.text,
                    primary_engine,
                    confidence=line_48.confidence,
                    confidence_supported=True,
                    primary_engine=primary_engine,
                    fallback_used=False,
                    polygon=manga_line.polygon,
                    direction=manga_line.direction,
                    fg_color=line_48.fg_color,
                    bg_color=line_48.bg_color,
                )
                needs_retry = not str(combined_line.text or "").strip()
                if needs_retry and str(line_48.text or "").strip():
                    line_48.primary_engine = primary_engine
                    line_48.fallback_used = True
                    final_lines.append(line_48)
                else:
                    final_lines.append(combined_line)

        hybrid_results.append(
            _aggregate_bubble_result(
                final_lines,
                primary_engine=primary_engine,
                secondary_engine=secondary_engine,
            )
        )

    return hybrid_results
