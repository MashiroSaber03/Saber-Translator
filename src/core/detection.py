"""
文本检测核心模块 (重构版)

使用统一检测器框架。
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Any
from PIL import Image
from collections import Counter

from src.shared import constants
from src.core.detector import DETECTOR_DEFAULT, detect
from src.core.detector.refinement import apply_saber_yolo_refinement

logger = logging.getLogger("CoreDetection")




# ========== 主要检测接口 ==========

def _filter_small_text_blocks(
    blocks,
    image_width: int,
    image_height: int,
    min_text_block_area_percent: float = 0,
):
    """按最终文本框面积占原图面积的百分比过滤极小文本框。"""
    try:
        threshold = float(min_text_block_area_percent or 0)
    except (TypeError, ValueError):
        threshold = 0.0

    if threshold <= 0 or image_width <= 0 or image_height <= 0:
        return list(blocks)

    image_area = float(image_width * image_height)
    min_area = image_area * threshold / 100.0
    filtered_blocks = []

    for block in blocks:
        x1, y1, x2, y2 = block.xyxy
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area >= min_area:
            filtered_blocks.append(block)

    removed_count = len(blocks) - len(filtered_blocks)
    if removed_count > 0:
        logger.info(
            "已按面积阈值过滤 %s 个极小文本框 (阈值: %s%%, 保留: %s)",
            removed_count,
            threshold,
            len(filtered_blocks),
        )

    return filtered_blocks

def _detect_with_optional_saber_refinement(
    image_pil: Image.Image,
    detector_type: str,
    edge_ratio_threshold: float,
    merge_lines: bool = None,
    enable_aux_yolo_detection: bool = None,
    aux_yolo_conf_threshold: float = None,
    aux_yolo_overlap_threshold: float = None,
    enable_saber_yolo_refine: bool = None,
    saber_yolo_refine_overlap_threshold: float = None,
):
    detection_result = detect(
        image_pil,
        detector_type=detector_type,
        merge_lines=merge_lines,
        edge_ratio_threshold=edge_ratio_threshold,
        expand_ratio=0,
        expand_top=0,
        expand_bottom=0,
        expand_left=0,
        expand_right=0,
        enable_aux_yolo_detection=enable_aux_yolo_detection,
        aux_yolo_conf_threshold=aux_yolo_conf_threshold,
        aux_yolo_overlap_threshold=aux_yolo_overlap_threshold,
    )
    return apply_saber_yolo_refinement(
        image_pil,
        detection_result,
        detector_type=detector_type,
        enabled=enable_saber_yolo_refine,
        reference_overlap_threshold=saber_yolo_refine_overlap_threshold,
    )





# ========== 坐标处理函数 ==========

def expand_coordinates(
    coords: List[Tuple[int, int, int, int]],
    image_width: int,
    image_height: int,
    expand_ratio: float = 0,
    expand_top: float = 0,
    expand_bottom: float = 0,
    expand_left: float = 0,
    expand_right: float = 0
) -> List[Tuple[int, int, int, int]]:
    """
    扩展文本框坐标，用于解决检测框偏小导致文字漏出的问题
    """
    if not coords:
        return coords
    
    if expand_ratio == 0 and expand_top == 0 and expand_bottom == 0 and expand_left == 0 and expand_right == 0:
        return coords
    
    expanded = []
    for x1, y1, x2, y2 in coords:
        width = x2 - x1
        height = y2 - y1
        
        if width <= 0 or height <= 0:
            expanded.append((x1, y1, x2, y2))
            continue
        
        base_expand_w = int(width * expand_ratio / 100)
        base_expand_h = int(height * expand_ratio / 100)
        extra_top = int(height * expand_top / 100)
        extra_bottom = int(height * expand_bottom / 100)
        extra_left = int(width * expand_left / 100)
        extra_right = int(width * expand_right / 100)
        
        new_x1 = max(0, x1 - base_expand_w - extra_left)
        new_y1 = max(0, y1 - base_expand_h - extra_top)
        new_x2 = min(image_width, x2 + base_expand_w + extra_right)
        new_y2 = min(image_height, y2 + base_expand_h + extra_bottom)
        
        expanded.append((new_x1, new_y1, new_x2, new_y2))
    
    return expanded


# ========== 自动排版相关函数 ==========

def angle_to_direction(angle_degrees: float) -> str:
    """根据文本行的角度判断排版方向"""
    angle = angle_degrees % 180
    if angle > 90:
        angle = angle - 180
    
    if abs(angle) <= 45:
        return 'h'
    else:
        return 'v'


def calculate_polygon_angle(polygon: List[List[int]]) -> float:
    """计算四边形文本框的旋转角度（基于长边方向）"""
    if len(polygon) != 4:
        return 0.0
    
    pts = np.array(polygon, dtype=np.float32)
    
    edge_lengths = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        length = np.linalg.norm(p2 - p1)
        edge_lengths.append((length, p1, p2))
    
    edge_lengths.sort(key=lambda x: x[0], reverse=True)
    _, p1, p2 = edge_lengths[0]
    
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.rad2deg(angle_rad)
    
    return angle_deg


def analyze_direction_from_textlines(textlines: List[Dict[str, Any]]) -> str:
    """分析一组文本行，通过多数投票判断排版方向"""
    if not textlines:
        return 'h'
    
    directions = []
    for line in textlines:
        if 'direction' in line and line['direction'] in ('h', 'v'):
            directions.append(line['direction'])
        elif 'angle' in line:
            directions.append(angle_to_direction(line['angle']))
        elif 'polygon' in line:
            angle = calculate_polygon_angle(line['polygon'])
            directions.append(angle_to_direction(angle))
        else:
            directions.append('h')
    
    if not directions:
        return 'h'
    
    counter = Counter(directions)
    most_common = counter.most_common(1)[0]
    return most_common[0]


def get_bubble_detection_result_with_auto_directions(
    image_pil: Image.Image,
    detector_type: str = None,
    expand_ratio: float = 0,
    expand_top: float = 0,
    expand_bottom: float = 0,
    expand_left: float = 0,
    expand_right: float = 0,
    edge_ratio_threshold: float = None,
    enable_aux_yolo_detection: bool = None,
    aux_yolo_conf_threshold: float = None,
    aux_yolo_overlap_threshold: float = None,
    enable_saber_yolo_refine: bool = None,
    saber_yolo_refine_overlap_threshold: float = None,
    min_text_block_area_percent: float = 0,
) -> Dict[str, Any]:
    """
    获取气泡检测结果，并返回每个气泡的自动排版方向
    """
    if detector_type is None:
        detector_type = DETECTOR_DEFAULT
    
    if edge_ratio_threshold is None:
        edge_ratio_threshold = constants.CTD_EDGE_RATIO_THRESHOLD
    
    result = {
        'coords': [],
        'polygons': [],
        'angles': [],
        'auto_directions': [],
        'textlines_per_bubble': [],
        'raw_mask': None,  # 模型生成的精确文字掩膜
        'raw_lines': []  # 原始文本行（合并前的单行框，用于 debug 显示）
    }
    
    try:
        logger.debug(f"使用 {detector_type} 检测器（自动排版）")
        
        detection_result = _detect_with_optional_saber_refinement(
            image_pil,
            detector_type=detector_type,
            edge_ratio_threshold=edge_ratio_threshold,
            merge_lines=None,
            enable_aux_yolo_detection=enable_aux_yolo_detection,
            aux_yolo_conf_threshold=aux_yolo_conf_threshold,
            aux_yolo_overlap_threshold=aux_yolo_overlap_threshold,
            enable_saber_yolo_refine=enable_saber_yolo_refine,
            saber_yolo_refine_overlap_threshold=saber_yolo_refine_overlap_threshold,
        )
        detection_result.blocks = _filter_small_text_blocks(
            detection_result.blocks,
            image_pil.width,
            image_pil.height,
            min_text_block_area_percent=min_text_block_area_percent,
        )
        
        # 保存模型生成的精确文字掩膜
        result['raw_mask'] = detection_result.mask
        # 保存原始文本行（合并前的单行框，用于 debug 显示）
        result['raw_lines'] = detection_result.raw_lines
        
        im_w, im_h = image_pil.width, image_pil.height
        
        for block in detection_result.blocks:
            x1, y1, x2, y2 = block.xyxy
            if x1 >= x2 or y1 >= y2:
                continue
            
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(im_w, int(x2))
            y2 = min(im_h, int(y2))
            
            result['coords'].append((x1, y1, x2, y2))
            result['angles'].append(float(block.angle))
            result['polygons'].append(block.polygon)
            
            # 获取文本行信息
            textlines_info = []
            if block.lines:
                for line in block.lines:
                    line_pts = line.pts.tolist()
                    direction = line.direction
                    textlines_info.append({
                        'polygon': line_pts,
                        'direction': direction
                    })
            
            result['textlines_per_bubble'].append(textlines_info)
            
            # 判断方向
            if textlines_info:
                auto_dir = analyze_direction_from_textlines(textlines_info)
            else:
                auto_dir = 'v' if block.vertical else 'h'
            
            result['auto_directions'].append(auto_dir)
        
        # 应用坐标扩展
        if result['coords']:
            result['coords'] = expand_coordinates(
                result['coords'],
                image_pil.width,
                image_pil.height,
                expand_ratio,
                expand_top,
                expand_bottom,
                expand_left,
                expand_right
            )
        
        # 智能排序已在检测器的后处理中完成，这里不再排序
        
        logger.debug(f"检测完成: {len(result['coords'])} 个气泡")
        return result

    
    except Exception as e:
        logger.error(f"自动排版检测出错: {e}", exc_info=True)
        return result
