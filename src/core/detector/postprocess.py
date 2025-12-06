"""
统一后处理模块

整合了 CTD 和 YOLO 中的后处理逻辑
"""

import logging
from typing import List, Tuple
import numpy as np

from .data_types import TextBlock, DetectionResult
from .geometry import box_area, box_intersection_area, is_box_contained, merge_boxes

logger = logging.getLogger("DetectorPostprocess")


def remove_contained_blocks(blocks: List[TextBlock]) -> List[TextBlock]:
    """删除被其他块完全包围的块"""
    if len(blocks) <= 1:
        return blocks
    
    to_remove = set()
    for i in range(len(blocks)):
        if i in to_remove:
            continue
        for j in range(len(blocks)):
            if i == j or j in to_remove:
                continue
            
            box_i = blocks[i].xyxy
            box_j = blocks[j].xyxy
            
            # 如果 i 被 j 完全包围，删除 i
            if is_box_contained(box_i, box_j):
                to_remove.add(i)
                break
    
    return [b for i, b in enumerate(blocks) if i not in to_remove]


def merge_overlapping_blocks(blocks: List[TextBlock], 
                            overlap_threshold: float = 0.7) -> List[TextBlock]:
    """合并重叠度高的块"""
    if len(blocks) <= 1:
        return blocks
    
    changed = True
    while changed:
        changed = False
        to_remove = set()
        merge_pairs = []
        
        for i in range(len(blocks)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(blocks)):
                if j in to_remove:
                    continue
                
                box_i = blocks[i].xyxy
                box_j = blocks[j].xyxy
                area_i = box_area(box_i)
                area_j = box_area(box_j)
                
                intersection = box_intersection_area(box_i, box_j)
                smaller_area = min(area_i, area_j)
                
                if smaller_area > 0 and intersection / smaller_area > overlap_threshold:
                    merge_pairs.append((i, j))
                    changed = True
        
        if merge_pairs:
            i, j = merge_pairs[0]
            merged_xyxy = merge_boxes(blocks[i].xyxy, blocks[j].xyxy)
            
            # 创建合并后的块
            merged_lines = blocks[i].lines + blocks[j].lines
            merged_block = TextBlock(
                lines=merged_lines,
                font_size=min(blocks[i].font_size, blocks[j].font_size),
                _angle=(blocks[i].angle + blocks[j].angle) / 2,
                fg_color=blocks[i].fg_color,
                bg_color=blocks[i].bg_color
            )
            
            blocks[i] = merged_block
            to_remove.add(j)
        
        if to_remove:
            blocks = [b for idx, b in enumerate(blocks) if idx not in to_remove]
    
    return blocks


def expand_blocks(blocks: List[TextBlock],
                  image_width: int,
                  image_height: int,
                  expand_ratio: float = 0,
                  expand_top: float = 0,
                  expand_bottom: float = 0,
                  expand_left: float = 0,
                  expand_right: float = 0) -> List[TextBlock]:
    """扩展文本块边界"""
    if not blocks:
        return blocks
    
    # 如果所有扩展参数都是0，直接返回
    if expand_ratio == 0 and expand_top == 0 and expand_bottom == 0 and expand_left == 0 and expand_right == 0:
        return blocks
    
    for block in blocks:
        x1, y1, x2, y2 = block.xyxy
        width = x2 - x1
        height = y2 - y1
        
        if width <= 0 or height <= 0:
            continue
        
        # 计算扩展量
        base_expand_w = int(width * expand_ratio / 100)
        base_expand_h = int(height * expand_ratio / 100)
        extra_top = int(height * expand_top / 100)
        extra_bottom = int(height * expand_bottom / 100)
        extra_left = int(width * expand_left / 100)
        extra_right = int(width * expand_right / 100)
        
        # 应用扩展
        new_x1 = max(0, x1 - base_expand_w - extra_left)
        new_y1 = max(0, y1 - base_expand_h - extra_top)
        new_x2 = min(image_width, x2 + base_expand_w + extra_right)
        new_y2 = min(image_height, y2 + base_expand_h + extra_bottom)
        
        # 更新 lines 中的第一个点（用于重新计算 xyxy）
        if block.lines:
            # 创建一个包含扩展边界的新 TextLine
            from .data_types import TextLine
            expanded_pts = np.array([
                [new_x1, new_y1],
                [new_x2, new_y1],
                [new_x2, new_y2],
                [new_x1, new_y2]
            ], dtype=np.int32)
            # 添加边界框作为一个虚拟 line
            block.lines = [TextLine(pts=expanded_pts, confidence=1.0)]
            # 清除所有相关缓存（cached_property 存储在 __dict__ 中）
            for cache_key in ['xyxy', 'xywh', 'center', 'min_rect', 'polygon', 'area']:
                if cache_key in block.__dict__:
                    del block.__dict__[cache_key]
    
    return blocks


def sort_blocks_by_reading_order(blocks: List[TextBlock], 
                                  right_to_left: bool = True) -> List[TextBlock]:
    """按阅读顺序排序（日漫从右到左，从上到下）"""
    if not blocks:
        return blocks
    
    if right_to_left:
        # 日漫阅读顺序：从右到左，从上到下
        blocks = sorted(blocks, key=lambda b: (-b.xyxy[0], b.xyxy[1]))
    else:
        # 普通阅读顺序：从左到右，从上到下
        blocks = sorted(blocks, key=lambda b: (b.xyxy[1], b.xyxy[0]))
    
    return blocks


def sort_blocks_by_area(blocks: List[TextBlock], descending: bool = True) -> List[TextBlock]:
    """按面积排序"""
    return sorted(blocks, key=lambda b: b.area, reverse=descending)


def postprocess_blocks(blocks: List[TextBlock],
                       image_width: int,
                       image_height: int,
                       expand_ratio: float = 0,
                       expand_top: float = 0,
                       expand_bottom: float = 0,
                       expand_left: float = 0,
                       expand_right: float = 0,
                       overlap_threshold: float = 0.7,
                       sort_by_area: bool = True,
                       **kwargs) -> List[TextBlock]:
    """
    完整的后处理流程
    
    1. 删除被包围的小块
    2. 合并重叠度高的块
    3. 扩展边界
    4. 排序
    """
    if not blocks:
        return blocks
    
    # 1. 删除被包围的小块
    blocks = remove_contained_blocks(blocks)
    
    # 2. 合并重叠的块
    blocks = merge_overlapping_blocks(blocks, overlap_threshold)
    
    # 3. 扩展边界
    blocks = expand_blocks(
        blocks, image_width, image_height,
        expand_ratio, expand_top, expand_bottom, expand_left, expand_right
    )
    
    # 4. 排序
    if sort_by_area:
        blocks = sort_blocks_by_area(blocks)
    
    logger.info(f"后处理完成: {len(blocks)} 个文本块")
    return blocks


def postprocess_result(result: DetectionResult,
                      image_width: int,
                      image_height: int,
                      **kwargs) -> DetectionResult:
    """后处理检测结果"""
    result.blocks = postprocess_blocks(
        result.blocks, image_width, image_height, **kwargs
    )
    return result
