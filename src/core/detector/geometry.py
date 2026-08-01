"""
几何计算工具函数

整合了 CTD 和 YOLO 中的几何相关函数
"""

import numpy as np
from typing import Tuple




def box_area(box: Tuple[int, int, int, int]) -> int:
    """计算框的面积"""
    return (box[2] - box[0]) * (box[3] - box[1])


def box_intersection_area(box1: Tuple, box2: Tuple) -> int:
    """计算两个框的交集面积"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x1 < x2 and y1 < y2:
        return (x2 - x1) * (y2 - y1)
    return 0




def is_box_contained(inner: Tuple, outer: Tuple) -> bool:
    """检查 inner 是否被 outer 完全包围"""
    return (outer[0] <= inner[0] and outer[1] <= inner[1] and 
            outer[2] >= inner[2] and outer[3] >= inner[3])




def can_merge_textlines(a, b, 
                        aspect_ratio_tol: float = 1.3,
                        font_size_ratio_tol: float = 2,
                        char_gap_tolerance: float = 1,
                        char_gap_tolerance2: float = 3,
                        discard_connection_gap: float = 2) -> bool:
    """
    判断两个文本行是否可以合并
    
    整合自 CTD 的 quadrilateral_can_merge_region
    """
    fs_a = a.font_size
    fs_b = b.font_size
    char_size = min(fs_a, fs_b)
    
    # 字号差异过大
    if max(fs_a, fs_b) / char_size > font_size_ratio_tol:
        return False
    
    # 宽高比差异过大
    if a.aspect_ratio > aspect_ratio_tol and b.aspect_ratio < 1.0 / aspect_ratio_tol:
        return False
    if b.aspect_ratio > aspect_ratio_tol and a.aspect_ratio < 1.0 / aspect_ratio_tol:
        return False
    
    # 距离检查
    dist = a.polygon.distance(b.polygon)
    if dist > discard_connection_gap * char_size:
        return False
    
    # 角度检查
    angle_diff = abs(a.angle - b.angle)
    if angle_diff > 15 * np.pi / 180:
        return False
    
    # 距离在可接受范围内
    if dist < char_size * char_gap_tolerance:
        return True
    
    # 中点距离检查
    poly_dist = a.poly_distance(b)
    if poly_dist <= char_size * char_gap_tolerance2:
        return True
    
    return False
