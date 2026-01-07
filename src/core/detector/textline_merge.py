"""
统一的文本行合并模块

基于 CTD 的合并算法，为所有文本检测器提供统一的合并接口
"""

import logging
import itertools
from typing import List, Set
from collections import Counter

import numpy as np
import networkx as nx
from shapely.geometry import Polygon

from .data_types import TextLine, TextBlock
from .geometry import can_merge_textlines

logger = logging.getLogger("TextlineMerge")


def _split_text_region(
    textlines: List[TextLine],
    connected_region_indices: Set[int],
    width: int,
    height: int,
    gamma: float = 0.5,
    sigma: float = 2
) -> List[Set[int]]:
    """分割文本区域"""
    connected_region_indices = list(connected_region_indices)
    
    # case 1: 只有一个
    if len(connected_region_indices) == 1:
        return [set(connected_region_indices)]
    
    # case 2: 只有两个
    if len(connected_region_indices) == 2:
        fs1 = textlines[connected_region_indices[0]].font_size
        fs2 = textlines[connected_region_indices[1]].font_size
        fs = max(fs1, fs2)
        
        line1 = textlines[connected_region_indices[0]]
        line2 = textlines[connected_region_indices[1]]
        
        if line1.distance_to(line2) < (1 + gamma) * fs \
                and abs(line1.angle - line2.angle) < 0.2 * np.pi:
            return [set(connected_region_indices)]
        else:
            return [set([connected_region_indices[0]]), set([connected_region_indices[1]])]
    
    # case 3: 多个
    G = nx.Graph()
    for idx in connected_region_indices:
        G.add_node(idx)
    for (u, v) in itertools.combinations(connected_region_indices, 2):
        G.add_edge(u, v, weight=textlines[u].distance_to(textlines[v]))
    
    edges = nx.algorithms.tree.minimum_spanning_edges(G, algorithm='kruskal', data=True)
    edges = sorted(edges, key=lambda a: a[2]['weight'], reverse=True)
    distances_sorted = [a[2]['weight'] for a in edges]
    fontsize = np.mean([textlines[idx].font_size for idx in connected_region_indices])
    distances_std = np.std(distances_sorted)
    distances_mean = np.mean(distances_sorted)
    std_threshold = max(0.3 * fontsize + 5, 5)
    
    b1, b2 = textlines[edges[0][0]], textlines[edges[0][1]]
    max_poly_distance = b1.polygon.distance(b2.polygon)
    max_centroid_alignment = min(abs(b1.centroid[0] - b2.centroid[0]), 
                                  abs(b1.centroid[1] - b2.centroid[1]))
    
    if (distances_sorted[0] <= distances_mean + distances_std * sigma
            or distances_sorted[0] <= fontsize * (1 + gamma)) \
            and (distances_std < std_threshold
            or max_poly_distance == 0 and max_centroid_alignment < 5):
        return [set(connected_region_indices)]
    else:
        G = nx.Graph()
        for idx in connected_region_indices:
            G.add_node(idx)
        for edge in edges[1:]:
            G.add_edge(edge[0], edge[1])
        ans = []
        for node_set in nx.algorithms.components.connected_components(G):
            ans.extend(_split_text_region(textlines, node_set, width, height))
        return ans


def _merge_textlines_to_regions(
    textlines: List[TextLine],
    width: int,
    height: int,
    edge_ratio_threshold: float = 0.0
) -> List[TextBlock]:
    """
    合并文本行到文本区域
    
    Args:
        textlines: 文本行列表
        width: 图像宽度
        height: 图像高度
        edge_ratio_threshold: 边缘距离比例阈值，用于断开距离差异过大的连接
    """
    if not textlines:
        return []
    
    # step 1: 构建连接图
    G = nx.Graph()
    for i, line in enumerate(textlines):
        G.add_node(i, line=line)
    
    edge_distances = {}
    for (u, v) in itertools.combinations(range(len(textlines)), 2):
        line_u = textlines[u]
        line_v = textlines[v]
        if can_merge_textlines(line_u, line_v):
            poly_dist = line_u.poly_distance(line_v)
            G.add_edge(u, v, distance=poly_dist)
            edge_distances[(u, v)] = poly_dist
    
    # step 1.5: 边缘距离比例检测
    if edge_ratio_threshold > 0 and len(textlines) > 2:
        edges_to_remove = []
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if len(neighbors) >= 2:
                neighbor_distances = []
                for neighbor in neighbors:
                    edge = (min(node, neighbor), max(node, neighbor))
                    dist = edge_distances.get(edge, 0)
                    neighbor_distances.append((neighbor, dist))
                
                neighbor_distances.sort(key=lambda x: x[1])
                min_dist = neighbor_distances[0][1]
                
                if min_dist > 0:
                    for neighbor, dist in neighbor_distances[1:]:
                        ratio = dist / min_dist
                        if ratio > edge_ratio_threshold:
                            edge_to_remove = (min(node, neighbor), max(node, neighbor))
                            if edge_to_remove not in edges_to_remove:
                                edges_to_remove.append(edge_to_remove)
        
        for edge in edges_to_remove:
            if G.has_edge(edge[0], edge[1]):
                G.remove_edge(edge[0], edge[1])
    
    # step 2: 分割区域
    region_indices: List[Set[int]] = []
    for node_set in nx.algorithms.components.connected_components(G):
        region_indices.extend(_split_text_region(textlines, node_set, width, height))
    
    # step 3: 创建 TextBlock
    blocks = []
    for node_set in region_indices:
        nodes = list(node_set)
        lines = [textlines[i] for i in nodes]
        
        # 去重
        unique_lines = []
        seen_coords = set()
        for line in lines:
            coords_tuple = tuple(line.pts.reshape(-1))
            if coords_tuple not in seen_coords:
                seen_coords.add(coords_tuple)
                unique_lines.append(line)
        
        if not unique_lines:
            continue
        
        # 计算平均颜色
        fg_r = round(np.mean([line.fg_color[0] for line in unique_lines]))
        fg_g = round(np.mean([line.fg_color[1] for line in unique_lines]))
        fg_b = round(np.mean([line.fg_color[2] for line in unique_lines]))
        bg_r = round(np.mean([line.bg_color[0] for line in unique_lines]))
        bg_g = round(np.mean([line.bg_color[1] for line in unique_lines]))
        bg_b = round(np.mean([line.bg_color[2] for line in unique_lines]))
        
        # 投票决定方向
        dirs = [line.direction for line in unique_lines]
        majority_dir_top_2 = Counter(dirs).most_common(2)
        if len(majority_dir_top_2) == 1:
            majority_dir = majority_dir_top_2[0][0]
        elif majority_dir_top_2[0][1] == majority_dir_top_2[1][1]:
            max_aspect_ratio = -100
            majority_dir = 'h'
            for line in unique_lines:
                if line.aspect_ratio > max_aspect_ratio:
                    max_aspect_ratio = line.aspect_ratio
                    majority_dir = line.direction
        else:
            majority_dir = majority_dir_top_2[0][0]
        
        # 按方向排序
        if majority_dir == 'h':
            sorted_indices = sorted(range(len(unique_lines)), 
                                   key=lambda x: unique_lines[x].centroid[1])
        else:
            sorted_indices = sorted(range(len(unique_lines)), 
                                   key=lambda x: -unique_lines[x].centroid[0])
        unique_lines = [unique_lines[i] for i in sorted_indices]
        
        # 计算置信度
        total_logprobs = 0
        total_area = sum([line.area for line in unique_lines])
        for line in unique_lines:
            if line.confidence > 0 and line.area > 0:
                total_logprobs += np.log(line.confidence) * line.area
        if total_area > 0:
            total_logprobs /= total_area
        
        # 计算字号和角度
        font_size = int(min([line.font_size for line in unique_lines]))
        angle = np.rad2deg(np.mean([line.angle for line in unique_lines])) - 90
        
        # 角度归零
        original_angles_deg = [np.rad2deg(line.angle) for line in unique_lines]
        has_near_90_degree = any(abs(orig_angle - 90.0) <= 1.0 for orig_angle in original_angles_deg)
        if has_near_90_degree or abs(angle) < 3:
            angle = 0
        
        block = TextBlock(
            lines=unique_lines,
            texts=[line.text for line in unique_lines],
            font_size=font_size,
            _angle=angle,
            fg_color=(fg_r, fg_g, fg_b),
            bg_color=(bg_r, bg_g, bg_b),
            _direction=majority_dir,
            prob=np.exp(total_logprobs)
        )
        blocks.append(block)
    
    return blocks


def merge_textlines(
    textlines: List[TextLine],
    image_width: int,
    image_height: int,
    edge_ratio_threshold: float = 0.0,
    verbose: bool = False
) -> List[TextBlock]:
    """
    统一的文本行合并接口
    
    Args:
        textlines: TextLine 列表
        image_width: 图像宽度
        image_height: 图像高度
        edge_ratio_threshold: 边缘距离比例阈值
        verbose: 是否输出详细日志
    
    Returns:
        List[TextBlock]: 合并后的文本块列表
    """
    if not textlines:
        return []
    
    blocks = _merge_textlines_to_regions(
        textlines, image_width, image_height, edge_ratio_threshold
    )
    
    if verbose:
        logger.info(f"文本行合并: {len(textlines)} 行 -> {len(blocks)} 块")
    
    return blocks
