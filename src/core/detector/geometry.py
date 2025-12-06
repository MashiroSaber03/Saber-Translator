"""
几何计算工具函数

整合了 CTD 和 YOLO 中的几何相关函数
"""

import numpy as np
from typing import List, Tuple
from shapely.geometry import Polygon


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    """将 xywh 格式转换为 xyxy 格式"""
    xyxy = np.zeros((xywh.shape[0], 4), dtype=xywh.dtype)
    xyxy[:, 0] = xywh[:, 0]
    xyxy[:, 1] = xywh[:, 1]
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2]
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3]
    return xyxy


def xywh_to_polygon(xywh: np.ndarray, to_int: bool = True) -> np.ndarray:
    """将 xywh 格式转换为 8 点多边形坐标"""
    xyxypoly = np.tile(xywh[:, [0, 1]], 4)
    xyxypoly[:, [2, 4]] += xywh[:, [2]]
    xyxypoly[:, [5, 7]] += xywh[:, [3]]
    if to_int:
        xyxypoly = xyxypoly.astype(np.int64)
    return xyxypoly


def xyxy_to_polygon(xyxy: Tuple[int, int, int, int]) -> List[List[int]]:
    """将 xyxy 转换为四边形顶点"""
    x1, y1, x2, y2 = xyxy
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def polygon_to_xyxy(pts: np.ndarray) -> Tuple[int, int, int, int]:
    """将四边形顶点转换为 xyxy"""
    pts = np.array(pts).reshape(-1, 2)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return int(x1), int(y1), int(x2), int(y2)


def rotate_polygons(center: np.ndarray, polygons: np.ndarray, 
                    rotation: float, new_center: np.ndarray = None,
                    to_int: bool = True) -> np.ndarray:
    """旋转多边形"""
    if rotation == 0:
        return polygons
    if new_center is None:
        new_center = center
    
    rotation = np.deg2rad(rotation)
    s, c = np.sin(rotation), np.cos(rotation)
    polygons = polygons.astype(np.float32)
    
    polygons[:, 1::2] -= center[1]
    polygons[:, ::2] -= center[0]
    rotated = np.copy(polygons)
    rotated[:, 1::2] = polygons[:, 1::2] * c - polygons[:, ::2] * s
    rotated[:, ::2] = polygons[:, 1::2] * s + polygons[:, ::2] * c
    rotated[:, 1::2] += new_center[1]
    rotated[:, ::2] += new_center[0]
    
    if to_int:
        return rotated.astype(np.int64)
    return rotated


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


def box_iou(box1: Tuple, box2: Tuple) -> float:
    """计算两个框的 IoU"""
    inter = box_intersection_area(box1, box2)
    area1 = box_area(box1)
    area2 = box_area(box2)
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def is_box_contained(inner: Tuple, outer: Tuple) -> bool:
    """检查 inner 是否被 outer 完全包围"""
    return (outer[0] <= inner[0] and outer[1] <= inner[1] and 
            outer[2] >= inner[2] and outer[3] >= inner[3])


def merge_boxes(box1: Tuple, box2: Tuple) -> Tuple[int, int, int, int]:
    """合并两个框"""
    return (min(box1[0], box2[0]), min(box1[1], box2[1]),
            max(box1[2], box2[2]), max(box1[3], box2[3]))


def distance_point_to_point(a: np.ndarray, b: np.ndarray) -> float:
    """计算两点距离"""
    return np.linalg.norm(a - b)


def distance_point_to_lineseg(p: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """计算点到线段的距离"""
    x, y = p[0], p[1]
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    
    A = x - x1
    B = y - y1
    C = x2 - x1
    D = y2 - y1

    dot = A * C + B * D
    len_sq = C * C + D * D
    param = -1
    if len_sq != 0:
        param = dot / len_sq

    if param < 0:
        xx, yy = x1, y1
    elif param > 1:
        xx, yy = x2, y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D

    dx = x - xx
    dy = y - yy
    return np.sqrt(dx * dx + dy * dy)


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
