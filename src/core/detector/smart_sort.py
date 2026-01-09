"""
智能排序模块 - 完全复刻 manga-image-translator 的排序逻辑

排序策略：
1. 分镜检测 + 分镜内排序（最优）
2. 智能分布分析（标准差法）
3. 简单排序（降级方案）
"""

import logging
import numpy as np
from typing import List, Tuple
from .data_types import TextBlock
from .panel_detector import get_panels_from_array

logger = logging.getLogger("SmartSort")


def sort_regions(
    regions: List[TextBlock],
    right_to_left: bool = True,
    img: np.ndarray = None,
    force_simple_sort: bool = False
) -> List[TextBlock]:
    """
    智能排序文本区域（完全复刻 manga-image-translator）
    
    Args:
        regions: 文本块列表
        right_to_left: 是否从右到左阅读（日漫模式）
        img: 原始图像（用于分镜检测）
        force_simple_sort: 强制使用简单排序
        
    Returns:
        排序后的文本块列表
    """
    if not regions:
        return []
    
    # 如果强制简单排序，直接返回
    if force_simple_sort:
        return _simple_sort(regions, right_to_left)
    
    # 1. 分镜检测 + 分镜内排序
    if img is not None:
        try:
            panels_raw = get_panels_from_array(img, rtl=right_to_left)
            # 转换为 [x1, y1, x2, y2]
            panels = [(x, y, x + w, y + h) for x, y, w, h in panels_raw]
            # 使用自定义排序保持垂直堆叠的分镜在一起
            panels = _sort_panels_fill(panels, right_to_left)
            
            # 为每个文本区域分配所属分镜
            for r in regions:
                cx, cy = r.center
                r.panel_index = -1
                for idx, (x1, y1, x2, y2) in enumerate(panels):
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        r.panel_index = idx
                        break
                
                if r.panel_index < 0:
                    # 如果不在任何分镜内，找最近的一个
                    dists = [
                        ((max(x1 - cx, 0, cx - x2)) ** 2 + (max(y1 - cy, 0, cy - y2)) ** 2, i)
                        for i, (x1, y1, x2, y2) in enumerate(panels)
                    ]
                    if dists:
                        r.panel_index = min(dists)[1]
            
            # 按分镜分组
            grouped = {}
            for r in regions:
                grouped.setdefault(r.panel_index, []).append(r)
            
            sorted_all = []
            # 按分镜索引排序，递归对每个分镜内的文本排序
            for pi in sorted(grouped.keys()):
                panel_sorted = sort_regions(grouped[pi], right_to_left, img=None, force_simple_sort=False)
                sorted_all += panel_sorted
            
            logger.debug(f"使用分镜检测排序，检测到 {len(panels)} 个分镜")
            return sorted_all
        
        except Exception as e:
            logger.debug(f"分镜检测失败 ({e.__class__.__name__}: {str(e)[:100]})，降级到简单排序")
            return _simple_sort(regions, right_to_left)
    
    # 2. 智能排序（无图像时或分镜检测失败）
    xs = [r.center[0] for r in regions]
    ys = [r.center[1] for r in regions]
    
    # 改进的分散度计算：使用标准差
    if len(regions) > 1:
        x_std = np.std(xs) if len(xs) > 1 else 0
        y_std = np.std(ys) if len(ys) > 1 else 0
        
        # 使用标准差比值来判断排列方向
        is_horizontal = x_std > y_std
    else:
        # 只有一个文本块时，默认为纵向
        is_horizontal = False
    
    sorted_regions = []
    if is_horizontal:
        # 横向更分散：先 x 再 y
        primary = sorted(regions, key=lambda r: -r.center[0] if right_to_left else r.center[0])
        group, prev = [], None
        for r in primary:
            cx = r.center[0]
            if prev is not None and abs(cx - prev) > 20:
                group.sort(key=lambda r: r.center[1])
                sorted_regions += group
                group = []
            group.append(r)
            prev = cx
        if group:
            group.sort(key=lambda r: r.center[1])
            sorted_regions += group
    else:
        # 纵向更分散：先 y 再 x
        primary = sorted(regions, key=lambda r: r.center[1])
        group, prev = [], None
        for r in primary:
            cy = r.center[1]
            if prev is not None and abs(cy - prev) > 15:
                group.sort(key=lambda r: -r.center[0] if right_to_left else r.center[0])
                sorted_regions += group
                group = []
            group.append(r)
            prev = cy
        if group:
            group.sort(key=lambda r: -r.center[0] if right_to_left else r.center[0])
            sorted_regions += group
    
    logger.debug(f"使用智能排序（{'横向' if is_horizontal else '纵向'}分散）")
    return sorted_regions


def _simple_sort(regions: List[TextBlock], right_to_left: bool = True) -> List[TextBlock]:
    """
    简单排序（降级方案）
    
    日漫模式：从右到左，从上到下
    普通模式：从左到右，从上到下
    """
    if right_to_left:
        # 日漫：右上角优先
        return sorted(regions, key=lambda r: (-r.center[0], r.center[1]))
    else:
        # 普通：左上角优先
        return sorted(regions, key=lambda r: (r.center[1], r.center[0]))


def _sort_panels_fill(panels: List[Tuple[int, int, int, int]], right_to_left: bool) -> List[Tuple[int, int, int, int]]:
    """
    对分镜进行排序（保持垂直堆叠的分镜在一起）
    
    策略：
    1. 按 Y 坐标分组（垂直相近的为一组）
    2. 组内按 X 坐标排序（右到左或左到右）
    3. 组间保持原始的 Y 顺序
    """
    if not panels:
        return []
    
    if len(panels) == 1:
        return panels
    
    # 按 Y 坐标排序
    panels_sorted_by_y = sorted(panels, key=lambda p: p[1])
    
    # 分组：Y 坐标差小于平均高度的 50% 视为同一行
    avg_height = sum(p[3] - p[1] for p in panels) / len(panels)
    threshold = avg_height * 0.5
    
    groups = []
    current_group = [panels_sorted_by_y[0]]
    
    for i in range(1, len(panels_sorted_by_y)):
        prev_y = panels_sorted_by_y[i-1][1]
        curr_y = panels_sorted_by_y[i][1]
        
        if abs(curr_y - prev_y) < threshold:
            current_group.append(panels_sorted_by_y[i])
        else:
            groups.append(current_group)
            current_group = [panels_sorted_by_y[i]]
    
    groups.append(current_group)
    
    # 组内按 X 坐标排序
    sorted_panels = []
    for group in groups:
        if right_to_left:
            group_sorted = sorted(group, key=lambda p: -p[0])  # 从右到左
        else:
            group_sorted = sorted(group, key=lambda p: p[0])   # 从左到右
        sorted_panels.extend(group_sorted)
    
    return sorted_panels


def sort_blocks_by_reading_order(
    blocks: List[TextBlock],
    right_to_left: bool = True,
    img: np.ndarray = None
) -> List[TextBlock]:
    """
    按阅读顺序排序文本块（统一接口）
    
    这是对外暴露的主接口，内部调用 sort_regions
    
    Args:
        blocks: 文本块列表
        right_to_left: 是否从右到左
        img: 原始图像（可选，用于分镜检测）
        
    Returns:
        排序后的文本块列表
    """
    return sort_regions(blocks, right_to_left=right_to_left, img=img, force_simple_sort=False)
