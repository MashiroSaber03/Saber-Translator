"""
YSGYolo 辅助一阶段检测融合逻辑

职责：
- 调用 YSGYolo 作为辅助检测器输出原始 TextLine
- 在 merge 之前对主检测器的原始文本行进行保守几何融合
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from src.shared import constants

from .data_types import TextLine

logger = logging.getLogger("AuxYoloDetection")


def normalize_aux_overlap_threshold(value: Optional[float]) -> float:
    """归一化辅助检测重叠阈值，兼容 0-1 / 0-100 输入。"""
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return constants.AUX_YOLO_OVERLAP_THRESHOLD

    if threshold > 1:
        threshold = threshold / 100.0
    return max(0.0, min(threshold, 1.0))


def _line_coords_key(line: TextLine) -> tuple:
    return tuple(int(v) for v in line.pts.reshape(-1))


def _box_area(line: TextLine) -> float:
    x1, y1, x2, y2 = line.xyxy
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def _intersection_area(a: TextLine, b: TextLine) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    return float((ix2 - ix1) * (iy2 - iy1))


def _overlap_ratio(a: TextLine, b: TextLine) -> float:
    smaller_area = min(_box_area(a), _box_area(b))
    if smaller_area <= 0:
        return 0.0
    return _intersection_area(a, b) / smaller_area


def _contains(outer: TextLine, inner: TextLine, eps: float = 2.0) -> bool:
    ox1, oy1, ox2, oy2 = outer.xyxy
    ix1, iy1, ix2, iy2 = inner.xyxy
    return ox1 <= ix1 + eps and oy1 <= iy1 + eps and ox2 >= ix2 - eps and oy2 >= iy2 - eps


def merge_aux_yolo_lines(
    main_lines: Sequence[TextLine],
    aux_lines: Sequence[TextLine],
    overlap_threshold: Optional[float] = None,
) -> List[TextLine]:
    """
    使用保守规则融合主检测器与辅助 YSGYolo 的原始文本行。

    规则：
    - 无显著重叠：添加辅助框
    - 完整包含且面积 >= 被包含主框总面积 * 2 且与其他主框重叠低于阈值：辅助框替换被包含主框
    - 其余显著重叠：丢弃辅助框
    """
    normalized_overlap_threshold = normalize_aux_overlap_threshold(overlap_threshold)
    main_lines = list(main_lines)
    aux_lines = list(aux_lines)

    if not main_lines:
        return list(aux_lines)
    if not aux_lines:
        return main_lines

    main_indices_to_remove = set()
    aux_lines_to_add: List[TextLine] = []

    for aux_line in aux_lines:
        aux_area = _box_area(aux_line)
        if aux_area <= 0:
            continue

        contained_main_indices = set()
        max_overlap_ratio_with_others = 0.0

        for main_idx, main_line in enumerate(main_lines):
            overlap_ratio = _overlap_ratio(aux_line, main_line)
            if overlap_ratio <= 0:
                continue

            if _contains(aux_line, main_line):
                contained_main_indices.add(main_idx)
            else:
                max_overlap_ratio_with_others = max(max_overlap_ratio_with_others, overlap_ratio)

        if contained_main_indices:
            contained_main_area = sum(_box_area(main_lines[idx]) for idx in contained_main_indices)
            if (
                contained_main_area > 0
                and aux_area >= contained_main_area * 2
                and max_overlap_ratio_with_others < normalized_overlap_threshold
            ):
                main_indices_to_remove.update(contained_main_indices)
                aux_lines_to_add.append(aux_line)
            continue

        if max_overlap_ratio_with_others < normalized_overlap_threshold:
            aux_lines_to_add.append(aux_line)

    merged_lines = [line for idx, line in enumerate(main_lines) if idx not in main_indices_to_remove]
    merged_lines.extend(aux_lines_to_add)

    unique_lines: List[TextLine] = []
    seen = set()
    for line in merged_lines:
        key = _line_coords_key(line)
        if key in seen:
            continue
        seen.add(key)
        unique_lines.append(line)
    return unique_lines


def detect_aux_yolo_lines(
    image_cv,
    conf_threshold: Optional[float] = None,
    aux_detector=None,
) -> List[TextLine]:
    """在 OpenCV BGR 图像上运行辅助 YSGYolo，返回原始 TextLine 列表。"""
    from .registry import get_detector

    detector = aux_detector or get_detector(constants.DETECTOR_YOLO)
    conf = constants.AUX_YOLO_CONF_THRESHOLD if conf_threshold is None else float(conf_threshold)
    textlines, _ = detector._detect_raw(image_cv, conf_thresh=conf)
    return textlines or []


def maybe_merge_with_aux_yolo(
    image_cv,
    main_lines: Sequence[TextLine],
    detector_type: Optional[str],
    enabled: Optional[bool] = None,
    conf_threshold: Optional[float] = None,
    overlap_threshold: Optional[float] = None,
    aux_detector=None,
) -> List[TextLine]:
    """按配置决定是否执行辅助检测并融合主检测器文本行。"""
    if enabled is None:
        enabled = constants.ENABLE_AUX_YOLO_DETECTION
    if not enabled or detector_type == constants.DETECTOR_YOLO:
        return list(main_lines)

    try:
        aux_lines = detect_aux_yolo_lines(
            image_cv,
            conf_threshold=conf_threshold,
            aux_detector=aux_detector,
        )
    except Exception as error:
        logger.warning(f"辅助 YSGYolo 检测失败，回退主检测结果: {error}")
        return list(main_lines)

    return merge_aux_yolo_lines(main_lines, aux_lines, overlap_threshold=overlap_threshold)
