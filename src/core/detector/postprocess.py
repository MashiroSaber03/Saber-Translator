"""
统一后处理模块

整合了 CTD 和 YOLO 中的后处理逻辑
"""

import logging
import math
from typing import List
import numpy as np

from .data_types import TextBlock
from .geometry import box_area, box_intersection_area, is_box_contained
from .smart_sort import sort_blocks_by_reading_order

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


def _simple_reading_order_sort(blocks: List[TextBlock], 
                                  right_to_left: bool = True) -> List[TextBlock]:
    """简单阅读顺序排序（日漫从右到左，从上到下）"""
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
                       overlap_threshold: float = 0.7,
                       sort_method: str = 'smart',  # 'smart', 'area', 'reading', 'none'
                       img: np.ndarray = None,  # 用于分镜检测
                       right_to_left: bool = True) -> List[TextBlock]:  # 阅读方向
    """
    完整的后处理流程
    
    1. 删除被包围的小块
    2. 合并重叠度高的块
    3. 排序
    
    Args:
        sort_method: 排序方法
            - 'smart': 智能排序（分镜检测 + 标准差分析）
            - 'area': 按面积排序（默认行为）
            - 'reading': 按阅读顺序排序
            - 'none': 不排序
        img: 原始图像（用于分镜检测，仅 smart 模式需要）
        right_to_left: 是否从右到左阅读（日漫模式）
    """
    if (
        isinstance(overlap_threshold, bool)
        or not isinstance(overlap_threshold, (int, float))
        or not math.isfinite(float(overlap_threshold))
        or not 0 <= float(overlap_threshold) <= 1
    ):
        raise ValueError("文本块重叠阈值必须是 0 到 1 之间的数字")
    if sort_method not in {'smart', 'area', 'reading', 'none'}:
        raise ValueError(f"未知的文本块排序方法: {sort_method}")
    if not isinstance(right_to_left, bool):
        raise TypeError("阅读方向必须是布尔值")
    if not blocks:
        return blocks
    
    # 1. 删除被包围的小块
    blocks = remove_contained_blocks(blocks)
    
    # 2. 合并重叠的块
    blocks = merge_overlapping_blocks(blocks, overlap_threshold)
    
    # 3. 排序
    if sort_method == 'smart':
        blocks = sort_blocks_by_reading_order(blocks, right_to_left=right_to_left, img=img)
    elif sort_method == 'reading':
        blocks = _simple_reading_order_sort(blocks, right_to_left=right_to_left)
    elif sort_method == 'area':
        blocks = sort_blocks_by_area(blocks)
    # 'none' 则不排序
    
    logger.debug(f"后处理完成: {len(blocks)} 个文本块（排序: {sort_method}）")
    return blocks
