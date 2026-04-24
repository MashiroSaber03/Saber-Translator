"""
SaberYOLO 二阶段防误合并纠错
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon

from src.shared import constants
from src.shared.config_loader import load_json_config

from .data_types import DetectionResult, TextBlock, TextLine
from .registry import detect
from .smart_sort import sort_blocks_by_reading_order
from .textline_merge import build_text_block_from_lines

logger = logging.getLogger("SaberYoloRefinement")


def _load_saber_yolo_refine_settings() -> dict:
    settings = load_json_config(constants.USER_SETTINGS_FILE, default_value={})
    return settings if isinstance(settings, dict) else {}


def _load_saber_yolo_refine_enabled(settings: Optional[dict] = None) -> bool:
    settings = settings or _load_saber_yolo_refine_settings()
    value = settings.get('enableSaberYoloRefine')
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {'true', '1', 'yes', 'on'}:
            return True
        if lowered in {'false', '0', 'no', 'off'}:
            return False
    return constants.ENABLE_SABER_YOLO_REFINE


def _normalize_overlap_threshold(value) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return constants.SABER_YOLO_REFINE_OVERLAP_THRESHOLD

    if threshold > 1:
        threshold = threshold / 100.0
    threshold = max(0.0, min(threshold, 1.0))
    return threshold


def _load_saber_yolo_refine_overlap_threshold(settings: Optional[dict] = None) -> float:
    settings = settings or _load_saber_yolo_refine_settings()
    return _normalize_overlap_threshold(settings.get('saberYoloRefineOverlapThreshold'))


def _block_polygon(block: TextBlock) -> Polygon:
    polygon = Polygon(block.polygon)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon


def _point_in_block(block: TextBlock, point: Sequence[float]) -> bool:
    polygon = _block_polygon(block)
    if polygon.is_empty:
        return False
    return polygon.buffer(1e-6).covers(Point(float(point[0]), float(point[1])))


def _reference_overlaps_block(block: TextBlock, reference_block: TextBlock, overlap_threshold: float) -> bool:
    block_polygon = _block_polygon(block)
    reference_polygon = _block_polygon(reference_block)
    if block_polygon.is_empty or reference_polygon.is_empty or reference_polygon.area <= 0:
        return False
    overlap_area = block_polygon.intersection(reference_polygon).area
    return overlap_area / reference_polygon.area >= overlap_threshold


def _get_candidate_reference_blocks(
    block: TextBlock,
    reference_blocks: Sequence[TextBlock],
    overlap_threshold: float,
) -> List[TextBlock]:
    candidates = []
    for reference_block in reference_blocks:
        if _point_in_block(block, reference_block.center) or _reference_overlaps_block(block, reference_block, overlap_threshold):
            candidates.append(reference_block)
    return candidates


def _choose_reference_index(line: TextLine, reference_blocks: Sequence[TextBlock]) -> Optional[int]:
    if not reference_blocks:
        return None

    reference_polygons = [_block_polygon(block) for block in reference_blocks]
    line_center = Point(float(line.center[0]), float(line.center[1]))

    best_overlap_index = None
    best_overlap_area = 0.0
    for idx, polygon in enumerate(reference_polygons):
        if polygon.is_empty:
            continue
        overlap_area = line.polygon.intersection(polygon).area
        if overlap_area > best_overlap_area:
            best_overlap_area = overlap_area
            best_overlap_index = idx

    if best_overlap_index is not None and best_overlap_area > 0:
        return best_overlap_index

    containing_indices = [
        idx for idx, polygon in enumerate(reference_polygons)
        if not polygon.is_empty and polygon.buffer(1e-6).covers(line_center)
    ]
    if containing_indices:
        return min(
            containing_indices,
            key=lambda idx: float(np.linalg.norm(line.center - reference_blocks[idx].center))
        )

    nearest_index = min(
        range(len(reference_blocks)),
        key=lambda idx: float(np.linalg.norm(line.center - reference_blocks[idx].center))
    )
    nearest_distance = float(np.linalg.norm(line.center - reference_blocks[nearest_index].center))
    if nearest_distance <= line.font_size * 1.5:
        return nearest_index

    return None


def _split_block_by_reference_blocks(
    block: TextBlock,
    reference_blocks: Sequence[TextBlock],
    overlap_threshold: float,
) -> Optional[List[TextBlock]]:
    candidate_reference_blocks = _get_candidate_reference_blocks(block, reference_blocks, overlap_threshold)
    if len(candidate_reference_blocks) <= 1:
        return None

    grouped_lines: List[List[TextLine]] = [[] for _ in candidate_reference_blocks]
    for line in block.lines:
        ref_index = _choose_reference_index(line, candidate_reference_blocks)
        if ref_index is None:
            return None
        grouped_lines[ref_index].append(line)

    non_empty_groups = [group for group in grouped_lines if group]
    if len(non_empty_groups) < 2:
        return None

    rebuilt_blocks = []
    for group in non_empty_groups:
        rebuilt_block = build_text_block_from_lines(group)
        if rebuilt_block is None:
            return None
        rebuilt_blocks.append(rebuilt_block)
    return rebuilt_blocks


def refine_detection_result_with_reference_blocks(
    detection_result: DetectionResult,
    reference_result: DetectionResult,
    image: Image.Image,
    right_to_left: bool = True,
    reference_overlap_threshold: float = constants.SABER_YOLO_REFINE_OVERLAP_THRESHOLD,
) -> DetectionResult:
    if not detection_result.blocks or not reference_result.blocks:
        return detection_result

    normalized_overlap_threshold = _normalize_overlap_threshold(reference_overlap_threshold)
    refined_blocks: List[TextBlock] = []
    changed = False
    for block in detection_result.blocks:
        replacement_blocks = _split_block_by_reference_blocks(
            block,
            reference_result.blocks,
            normalized_overlap_threshold,
        )
        if replacement_blocks is None:
            refined_blocks.append(block)
            continue

        refined_blocks.extend(replacement_blocks)
        changed = True

    if not changed:
        return detection_result

    img_cv = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
    sorted_blocks = sort_blocks_by_reading_order(refined_blocks, right_to_left=right_to_left, img=img_cv)
    return DetectionResult(
        blocks=sorted_blocks,
        mask=detection_result.mask,
        raw_lines=detection_result.raw_lines,
    )


def apply_saber_yolo_refinement(
    image: Image.Image,
    detection_result: DetectionResult,
    detector_type: Optional[str] = None,
    enabled: Optional[bool] = None,
    right_to_left: bool = True,
    reference_overlap_threshold: Optional[float] = None,
) -> DetectionResult:
    if detector_type == constants.DETECTOR_SABER_YOLO:
        return detection_result

    settings = None
    if enabled is None or reference_overlap_threshold is None:
        settings = _load_saber_yolo_refine_settings()

    if enabled is None:
        enabled = _load_saber_yolo_refine_enabled(settings)
    if reference_overlap_threshold is None:
        reference_overlap_threshold = _load_saber_yolo_refine_overlap_threshold(settings)
    else:
        reference_overlap_threshold = _normalize_overlap_threshold(reference_overlap_threshold)

    if not enabled or len(detection_result.blocks) == 0 or len(detection_result.raw_lines) < 2:
        return detection_result

    try:
        reference_result = detect(
            image,
            detector_type=constants.DETECTOR_SABER_YOLO,
            merge_lines=False,
            expand_ratio=0,
            expand_top=0,
            expand_bottom=0,
            expand_left=0,
            expand_right=0,
            sort_method='none',
        )
    except Exception as error:
        logger.warning(f"SaberYOLO 二阶段纠错失败，回退原检测结果: {error}")
        return detection_result

    if len(reference_result.blocks) < 2:
        return detection_result

    return refine_detection_result_with_reference_blocks(
        detection_result,
        reference_result,
        image,
        right_to_left=right_to_left,
        reference_overlap_threshold=reference_overlap_threshold,
    )
