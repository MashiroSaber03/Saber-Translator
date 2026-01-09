"""
分镜检测模块 - 参考 manga-image-translator 的 Kumiko 库实现

使用传统 CV 方法检测漫画的分镜（Panel）边框
"""

import logging
import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger("PanelDetector")


@dataclass
class Panel:
    """分镜（Panel）数据结构"""
    x: int
    y: int
    w: int
    h: int
    
    @property
    def x1(self) -> int:
        return self.x
    
    @property
    def y1(self) -> int:
        return self.y
    
    @property
    def x2(self) -> int:
        return self.x + self.w
    
    @property
    def y2(self) -> int:
        return self.y + self.h
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)
    
    @property
    def area(self) -> int:
        return self.w * self.h
    
    def to_xywh(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)
    
    def to_xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)
    
    def contains_point(self, px: int, py: int) -> bool:
        """判断点是否在分镜内"""
        return self.x <= px <= self.x2 and self.y <= py <= self.y2


class PanelDetector:
    """
    分镜检测器 - 基于传统 CV 方法
    
    检测流程：
    1. Sobel 边缘检测
    2. 自适应二值化
    3. 形态学闭运算（填补小缺口）
    4. 轮廓提取
    5. 多边形简化
    6. 过滤小面板
    """
    
    DEFAULT_MIN_PANEL_RATIO = 1 / 10  # 最小面板面积占比
    
    def __init__(self, min_panel_ratio: float = None):
        """
        初始化分镜检测器
        
        Args:
            min_panel_ratio: 最小面板面积占比（相对于整页）
        """
        self.min_panel_ratio = min_panel_ratio or self.DEFAULT_MIN_PANEL_RATIO
    
    def detect_panels(self, img: np.ndarray) -> List[Panel]:
        """
        检测图像中的分镜
        
        Args:
            img: BGR 格式的图像数组
            
        Returns:
            Panel 列表，按面积从大到小排序
        """
        if img is None or img.size == 0:
            logger.warning("输入图像为空")
            return []
        
        img_h, img_w = img.shape[:2]
        min_panel_area = int(img_w * img_h * self.min_panel_ratio)
        
        # 1. 转灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Sobel 边缘检测
        sobel = self._apply_sobel(gray)
        
        # 3. 自适应二值化 + 形态学闭运算
        thresh = self._threshold_and_close(sobel)
        
        # 4. 提取轮廓
        contours = self._find_contours(thresh)
        
        # 5. 转换为 Panel 对象
        panels = []
        for contour in contours:
            panel = self._contour_to_panel(contour)
            if panel and panel.area >= min_panel_area:
                panels.append(panel)
        
        # 6. 按面积降序排序
        panels.sort(key=lambda p: p.area, reverse=True)
        
        logger.debug(f"检测到 {len(panels)} 个分镜")
        return panels
    
    def _apply_sobel(self, gray: np.ndarray) -> np.ndarray:
        """应用 Sobel 算子"""
        ddepth = cv2.CV_16S
        
        # X 方向梯度
        grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        # Y 方向梯度
        grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        
        # 转换为绝对值
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        
        # 合并梯度
        sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        
        return sobel
    
    def _threshold_and_close(self, sobel: np.ndarray) -> np.ndarray:
        """二值化 + 形态学闭运算"""
        # Otsu 自适应阈值
        _, thresh = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学闭运算：填补 1-2 像素的小缺口
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return thresh
    
    def _find_contours(self, thresh: np.ndarray) -> List[np.ndarray]:
        """提取外部轮廓"""
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def _contour_to_panel(self, contour: np.ndarray) -> Optional[Panel]:
        """将轮廓转换为 Panel"""
        # 多边形近似
        arclength = cv2.arcLength(contour, True)
        epsilon = 0.001 * arclength
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 获取外接矩形
        x, y, w, h = cv2.boundingRect(approx)
        
        if w <= 0 or h <= 0:
            return None
        
        return Panel(x=x, y=y, w=w, h=h)


def get_panels_from_array(img: np.ndarray, rtl: bool = True, min_panel_ratio: float = None) -> List[Tuple[int, int, int, int]]:
    """
    从图像数组检测分镜（兼容 manga-image-translator 接口）
    
    Args:
        img: BGR 格式的图像
        rtl: 是否从右到左阅读（暂未使用，用于排序）
        min_panel_ratio: 最小面板比例
        
    Returns:
        [(x, y, w, h), ...] 格式的分镜列表
    """
    detector = PanelDetector(min_panel_ratio=min_panel_ratio)
    panels = detector.detect_panels(img)
    return [p.to_xywh() for p in panels]
