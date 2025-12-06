"""
统一数据结构

整合了 CTD 的 Quadrilateral/TextBlock 和 YOLO 的 TextBlock
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
from functools import cached_property
import numpy as np
import cv2
from shapely.geometry import Polygon, MultiPoint


@dataclass
class TextLine:
    """
    单个文本行（四边形）
    
    整合自:
    - CTD: Quadrilateral
    - YOLO: 检测框
    """
    pts: np.ndarray  # shape: (4, 2), 四个角点坐标
    confidence: float = 1.0
    text: str = ""
    
    # 颜色信息（可选）
    fg_color: Tuple[int, int, int] = (0, 0, 0)
    bg_color: Tuple[int, int, int] = (255, 255, 255)
    
    def __post_init__(self):
        # 确保 pts 是正确的格式
        if isinstance(self.pts, list):
            self.pts = np.array(self.pts, dtype=np.int32)
        if self.pts.shape != (4, 2):
            self.pts = self.pts.reshape(4, 2)
        # 排序点
        self.pts, self._is_vertical = self._sort_points(self.pts)
    
    @staticmethod
    def _sort_points(pts: np.ndarray) -> Tuple[np.ndarray, bool]:
        """对四边形的四个点进行排序，并判断是否为竖排文本"""
        pts = pts.astype(np.float32)
        
        # 使用结构向量判断方向
        pairwise_vec = (pts[:, None] - pts[None]).reshape((16, -1))
        pairwise_vec_norm = np.linalg.norm(pairwise_vec, axis=1)
        long_side_ids = np.argsort(pairwise_vec_norm)[[8, 10]]
        long_side_vecs = pairwise_vec[long_side_ids]
        inner_prod = (long_side_vecs[0] * long_side_vecs[1]).sum()
        if inner_prod < 0:
            long_side_vecs[0] = -long_side_vecs[0]
        struc_vec = np.abs(long_side_vecs.mean(axis=0))
        is_vertical = struc_vec[0] <= struc_vec[1]
        
        if is_vertical:
            pts = pts[np.argsort(pts[:, 1])]
            pts = pts[[*np.argsort(pts[:2, 0]), *np.argsort(pts[2:, 0])[::-1] + 2]]
        else:
            pts = pts[np.argsort(pts[:, 0])]
            pts_sorted = np.zeros_like(pts)
            pts_sorted[[0, 3]] = sorted(pts[[0, 1]], key=lambda x: x[1])
            pts_sorted[[1, 2]] = sorted(pts[[2, 3]], key=lambda x: x[1])
            pts = pts_sorted
        
        return pts.astype(np.int32), is_vertical
    
    @cached_property
    def xyxy(self) -> Tuple[int, int, int, int]:
        """轴对齐边界框 (x1, y1, x2, y2)"""
        x1, y1 = self.pts.min(axis=0)
        x2, y2 = self.pts.max(axis=0)
        return int(x1), int(y1), int(x2), int(y2)
    
    @cached_property
    def xywh(self) -> Tuple[int, int, int, int]:
        """(x, y, width, height) 格式"""
        x1, y1, x2, y2 = self.xyxy
        return x1, y1, x2 - x1, y2 - y1
    
    @cached_property
    def center(self) -> np.ndarray:
        """中心点"""
        return np.mean(self.pts, axis=0)
    
    @cached_property
    def centroid(self) -> np.ndarray:
        """质心（与 center 相同）"""
        return self.center
    
    @cached_property
    def structure(self) -> List[np.ndarray]:
        """结构点：四条边的中点"""
        p1 = ((self.pts[0] + self.pts[1]) / 2).astype(int)
        p2 = ((self.pts[2] + self.pts[3]) / 2).astype(int)
        p3 = ((self.pts[1] + self.pts[2]) / 2).astype(int)
        p4 = ((self.pts[3] + self.pts[0]) / 2).astype(int)
        return [p1, p2, p3, p4]
    
    @cached_property
    def font_size(self) -> float:
        """估计的字号（短边长度）"""
        [l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
        v1 = l1b - l1a
        v2 = l2b - l2a
        return min(np.linalg.norm(v2), np.linalg.norm(v1))
    
    @cached_property
    def aspect_ratio(self) -> float:
        """宽高比"""
        [l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
        v1 = l1b - l1a
        v2 = l2b - l2a
        norm_v = np.linalg.norm(v1)
        if norm_v == 0:
            return 1.0
        return np.linalg.norm(v2) / norm_v
    
    @cached_property
    def is_vertical(self) -> bool:
        """是否为竖排文本"""
        return self._is_vertical
    
    @cached_property
    def direction(self) -> str:
        """方向: 'h' (横向) 或 'v' (竖向)"""
        return 'v' if self._is_vertical else 'h'
    
    @cached_property
    def angle(self) -> float:
        """旋转角度（弧度）"""
        [l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
        v1 = l1b - l1a
        e2 = np.array([1, 0])
        norm = np.linalg.norm(v1)
        if norm == 0:
            return 0.0
        unit_vector = v1 / norm
        cos_angle = np.dot(unit_vector, e2)
        return np.fmod(np.arccos(np.clip(cos_angle, -1, 1)) + np.pi, np.pi)
    
    @cached_property
    def angle_degrees(self) -> float:
        """旋转角度（度）"""
        return np.rad2deg(self.angle) - 90
    
    @cached_property
    def polygon(self) -> Polygon:
        """Shapely 多边形对象"""
        return MultiPoint([tuple(p) for p in self.pts]).convex_hull
    
    @cached_property
    def area(self) -> float:
        """面积"""
        return self.polygon.area
    
    def distance_to(self, other: 'TextLine') -> float:
        """计算与另一个文本行的距离"""
        return self.polygon.distance(other.polygon)
    
    def poly_distance(self, other: 'TextLine') -> float:
        """计算两个框之间的距离，优先使用平行边的中点距离"""
        dir_a = self.direction
        dir_b = other.direction
        
        if dir_a == dir_b:
            if dir_a == 'h':
                self_top_mid = (self.pts[0] + self.pts[1]) / 2
                self_bottom_mid = (self.pts[2] + self.pts[3]) / 2
                other_top_mid = (other.pts[0] + other.pts[1]) / 2
                other_bottom_mid = (other.pts[2] + other.pts[3]) / 2
                
                distances = [
                    np.linalg.norm(self_top_mid - other_top_mid),
                    np.linalg.norm(self_top_mid - other_bottom_mid),
                    np.linalg.norm(self_bottom_mid - other_top_mid),
                    np.linalg.norm(self_bottom_mid - other_bottom_mid),
                ]
                return min(distances)
            else:
                self_left_mid = (self.pts[0] + self.pts[3]) / 2
                self_right_mid = (self.pts[1] + self.pts[2]) / 2
                other_left_mid = (other.pts[0] + other.pts[3]) / 2
                other_right_mid = (other.pts[1] + other.pts[2]) / 2
                
                distances = [
                    np.linalg.norm(self_left_mid - other_left_mid),
                    np.linalg.norm(self_left_mid - other_right_mid),
                    np.linalg.norm(self_right_mid - other_left_mid),
                    np.linalg.norm(self_right_mid - other_right_mid),
                ]
                return min(distances)
        
        return self.polygon.distance(other.polygon)
    
    def clip(self, width: int, height: int):
        """裁剪到图像边界"""
        self.pts[:, 0] = np.clip(self.pts[:, 0], 0, width)
        self.pts[:, 1] = np.clip(self.pts[:, 1], 0, height)


@dataclass
class TextBlock:
    """
    合并后的文本块
    
    整合自:
    - CTD: textblock.py 的 TextBlock
    - YOLO: utils.py 的 TextBlock
    """
    lines: List[TextLine] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    
    # 字体信息
    font_size: float = -1
    _angle: float = 0  # 度
    
    # 颜色
    fg_color: Tuple[int, int, int] = (0, 0, 0)
    bg_color: Tuple[int, int, int] = (255, 255, 255)
    
    # 方向
    _direction: str = 'auto'
    
    # 置信度
    prob: float = 1.0
    
    # 标签（用于 YOLO 分类）
    label: str = ""
    
    def __post_init__(self):
        # 如果 lines 是 np.ndarray 列表，转换为 TextLine
        if self.lines and isinstance(self.lines[0], np.ndarray):
            self.lines = [TextLine(pts=pts) for pts in self.lines]
    
    @cached_property
    def xyxy(self) -> Tuple[int, int, int, int]:
        """
        角度感知的边界框
        
        将所有点旋转到0°后计算 AABB，这样配合 rotation_angle 使用时，
        能得到紧凑的最小外接矩形效果。
        
        算法：
        1. 获取所有文本行的顶点
        2. 计算文本块的平均角度
        3. 将所有点围绕 minAreaRect 的中心旋转 -angle（旋转到0°）
        4. 计算旋转后的点的 AABB
        5. 返回以原始中心为基准的边界框坐标
        
        注意：使用 minAreaRect 的中心点，保证与 polygon 属性一致，
        避免标注框与检测框之间的位置偏移。
        """
        if not self.lines:
            return (0, 0, 0, 0)
        
        all_pts = np.vstack([line.pts for line in self.lines]).astype(np.float32)
        
        # 获取角度
        angle_deg = self.angle
        
        # 如果角度接近0，直接计算 AABB（保持原有行为）
        if abs(angle_deg) < 1:
            x1, y1 = all_pts.min(axis=0)
            x2, y2 = all_pts.max(axis=0)
            return int(x1), int(y1), int(x2), int(y2)
        
        # 使用 minAreaRect 的中心点，保证与 polygon 属性一致
        rect = cv2.minAreaRect(all_pts)
        cx, cy = rect[0]  # minAreaRect 返回 ((cx, cy), (w, h), angle)
        
        # 将所有点旋转到0°（旋转 -angle）
        angle_rad = np.deg2rad(-angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # 平移到原点 -> 旋转
        pts_centered = all_pts - np.array([cx, cy])
        pts_rotated = np.zeros_like(pts_centered)
        pts_rotated[:, 0] = pts_centered[:, 0] * cos_a - pts_centered[:, 1] * sin_a
        pts_rotated[:, 1] = pts_centered[:, 0] * sin_a + pts_centered[:, 1] * cos_a
        
        # 计算旋转后的 AABB（相对于中心的半宽高）
        half_w = (pts_rotated[:, 0].max() - pts_rotated[:, 0].min()) / 2
        half_h = (pts_rotated[:, 1].max() - pts_rotated[:, 1].min()) / 2
        
        # 返回以中心点为基准的边界框（使用 floor/ceil 确保完全包含内容）
        x1 = int(np.floor(cx - half_w))
        y1 = int(np.floor(cy - half_h))
        x2 = int(np.ceil(cx + half_w))
        y2 = int(np.ceil(cy + half_h))
        
        return x1, y1, x2, y2
    
    @cached_property
    def xywh(self) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.xyxy
        return x1, y1, x2 - x1, y2 - y1
    
    @cached_property
    def center(self) -> np.ndarray:
        xyxy = np.array(self.xyxy)
        return (xyxy[:2] + xyxy[2:]) / 2
    
    @property
    def angle(self) -> float:
        """旋转角度（度）"""
        if self._angle != 0:
            return self._angle
        if not self.lines:
            return 0
        # 计算平均角度
        angles = [line.angle_degrees for line in self.lines]
        avg = np.mean(angles)
        # 小角度归零
        if abs(avg) < 3:
            return 0
        return avg
    
    @angle.setter
    def angle(self, value: float):
        self._angle = value
    
    @property
    def direction(self) -> str:
        """排版方向: 'h' (横向) 或 'v' (竖向)"""
        if self._direction in ('h', 'v', 'hr', 'vr'):
            return self._direction
        
        # 自动判断
        if not self.lines:
            x1, y1, x2, y2 = self.xyxy
            return 'v' if (y2 - y1) > (x2 - x1) else 'h'
        
        # 根据面积最大的文本行判断
        max_area = 0
        max_direction = 'h'
        for line in self.lines:
            if line.area > max_area:
                max_area = line.area
                max_direction = line.direction
        return max_direction
    
    @direction.setter
    def direction(self, value: str):
        self._direction = value
    
    @cached_property
    def vertical(self) -> bool:
        return self.direction.startswith('v')
    
    @cached_property
    def horizontal(self) -> bool:
        return self.direction.startswith('h')
    
    @cached_property
    def min_rect(self) -> np.ndarray:
        """最小外接矩形"""
        if not self.lines:
            x1, y1, x2, y2 = self.xyxy
            return np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]])
        
        all_pts = np.vstack([line.pts for line in self.lines])
        rect = cv2.minAreaRect(all_pts.astype(np.float32))
        box = cv2.boxPoints(rect).astype(np.int32)
        return np.array([box])
    
    @cached_property
    def polygon(self) -> List[List[int]]:
        """四边形顶点列表"""
        return self.min_rect[0].tolist()
    
    @cached_property
    def area(self) -> float:
        """面积"""
        x1, y1, x2, y2 = self.xyxy
        return (x2 - x1) * (y2 - y1)
    
    @property
    def text(self) -> str:
        """合并后的文本"""
        if self.texts:
            return ' '.join(self.texts)
        return ' '.join([line.text for line in self.lines if line.text])
    
    def adjust_bbox(self, im_w: int = None, im_h: int = None):
        """调整边界框到图像范围内"""
        if im_w is None or im_h is None:
            return
        for line in self.lines:
            line.clip(im_w, im_h)
        # 清除缓存
        if 'xyxy' in self.__dict__:
            del self.__dict__['xyxy']
        if 'min_rect' in self.__dict__:
            del self.__dict__['min_rect']


@dataclass
class DetectionResult:
    """统一的检测结果"""
    blocks: List[TextBlock] = field(default_factory=list)
    mask: Optional[np.ndarray] = None
    
    # 原始文本行（合并前）
    raw_lines: List[TextLine] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.blocks)
    
    def __iter__(self):
        return iter(self.blocks)
    
    def to_legacy_format(self) -> dict:
        """
        转换为原有的 coords/polygons/angles 格式
        保持向后兼容
        """
        coords = []
        polygons = []
        angles = []
        
        for block in self.blocks:
            coords.append(block.xyxy)
            polygons.append(block.polygon)
            angles.append(block.angle)
        
        return {
            'coords': coords,
            'polygons': polygons,
            'angles': angles
        }
    
    @property
    def coords(self) -> List[Tuple[int, int, int, int]]:
        """向后兼容：坐标列表"""
        return [block.xyxy for block in self.blocks]
    
    @property
    def polygons(self) -> List[List[List[int]]]:
        """向后兼容：多边形列表"""
        return [block.polygon for block in self.blocks]
    
    @property
    def angles(self) -> List[float]:
        """向后兼容：角度列表"""
        return [block.angle for block in self.blocks]
