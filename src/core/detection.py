"""
文本检测核心模块 (重构版)

使用统一检测器框架，保持向后兼容
"""

import logging
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from PIL import Image
from collections import Counter

from src.shared import constants
from src.core.detector import (
    get_detector, detect, detect_to_legacy_format,
    DETECTOR_CTD, DETECTOR_YOLO, DETECTOR_YOLOV5,
    DetectionResult, TextBlock
)

logger = logging.getLogger("CoreDetection")


class DetectionException(Exception):
    """检测过程失败异常"""
    pass


# ========== 主要检测接口 ==========

def get_bubble_detection_result(
    image_pil: Image.Image, 
    conf_threshold: float = 0.6,
    detector_type: str = None,
    expand_ratio: float = 0,
    expand_top: float = 0,
    expand_bottom: float = 0,
    expand_left: float = 0,
    expand_right: float = 0,
    edge_ratio_threshold: float = None
) -> dict:
    """
    使用指定检测器检测图像中的文本区域，返回完整检测结果（含角度信息）
    
    Args:
        image_pil: 输入的 PIL 图像对象
        conf_threshold: 检测的置信度阈值 (保留接口兼容性)
        detector_type: 检测器类型 ('ctd', 'yolo', 'yolov5', 'default')，默认使用 CTD
        expand_ratio: 整体扩展比例 (%)
        expand_top/bottom/left/right: 各边额外扩展比例 (%)
        edge_ratio_threshold: 边缘距离比例阈值，用于防止跨气泡错误合并
    
    Returns:
        dict: {
            'coords': List[Tuple[int, int, int, int]],  # 轴对齐边界框
            'polygons': List[List[List[int]]],  # 带角度的四边形
            'angles': List[float],  # 旋转角度（度）
            'raw_mask': Optional[np.ndarray]  # 模型生成的精确文字掩膜（仅 CTD/Default 支持）
        }
    """
    if detector_type is None:
        detector_type = constants.DEFAULT_DETECTOR
    
    if edge_ratio_threshold is None:
        edge_ratio_threshold = constants.CTD_EDGE_RATIO_THRESHOLD
    
    try:
        logger.debug(f"使用 {detector_type} 检测器")
        
        # 使用新的统一接口（merge_lines 由检测器自己决定）
        result = detect(
            image_pil,
            detector_type=detector_type,
            edge_ratio_threshold=edge_ratio_threshold,
            expand_ratio=expand_ratio,
            expand_top=expand_top,
            expand_bottom=expand_bottom,
            expand_left=expand_left,
            expand_right=expand_right
        )
        
        # 转换为旧格式
        legacy = result.to_legacy_format()
        
        # 保存模型生成的精确文字掩膜（仅 CTD/Default 支持）
        legacy['raw_mask'] = result.mask
        # 保存原始文本行（合并前的单行框，用于 debug 显示）
        legacy['raw_lines'] = result.raw_lines
        
        # 智能排序已在检测器的后处理中完成，这里不再排序
        
        logger.debug(f"获取 {len(legacy['coords'])} 个文本区域")
        return legacy
    
    except Exception as e:
        logger.error(f"检测过程出错 (检测器: {detector_type}): {e}", exc_info=True)
        return {'coords': [], 'polygons': [], 'angles': [], 'raw_mask': None, 'raw_lines': []}


def get_bubble_coordinates(
    image_pil: Image.Image, 
    conf_threshold: float = 0.6,
    detector_type: str = None,
    expand_ratio: float = 0,
    expand_top: float = 0,
    expand_bottom: float = 0,
    expand_left: float = 0,
    expand_right: float = 0,
    edge_ratio_threshold: float = None
) -> List[Tuple[int, int, int, int]]:
    """
    使用指定检测器检测图像中的文本区域并返回排序后的坐标列表
    （兼容旧接口，只返回 AABB 坐标）
    """
    result = get_bubble_detection_result(
        image_pil, conf_threshold, detector_type,
        expand_ratio, expand_top, expand_bottom, expand_left, expand_right,
        edge_ratio_threshold
    )
    return result.get('coords', [])


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


def detect_textlines_with_paddle_det(
    image_pil: Image.Image,
    bubble_coord: Tuple[int, int, int, int]
) -> List[Dict[str, Any]]:
    """使用 PaddleOCR det 模型检测气泡内的文本行（降级方案）"""
    x1, y1, x2, y2 = bubble_coord
    bubble_img = image_pil.crop((x1, y1, x2, y2))
    bubble_np = np.array(bubble_img.convert('RGB'))
    
    textlines = []
    
    try:
        from src.interfaces.paddle_ocr_onnx_interface import get_paddle_ocr_handler
        
        handler = get_paddle_ocr_handler()
        if not handler.initialized:
            handler.initialize("chinese")
        
        if handler.ocr is None:
            return textlines
        
        result, _ = handler.ocr(bubble_np)
        
        if result:
            for line in result:
                if len(line) >= 1 and line[0] is not None:
                    bbox = line[0]
                    if len(bbox) == 4:
                        polygon = [[int(p[0]) + x1, int(p[1]) + y1] for p in bbox]
                        angle = calculate_polygon_angle(polygon)
                        direction = angle_to_direction(angle)
                        
                        textlines.append({
                            'polygon': polygon,
                            'direction': direction,
                            'angle': angle
                        })
    except ImportError:
        pass
    except Exception as e:
        logger.error(f"使用 PaddleOCR det 检测文本行时出错: {e}")
    
    return textlines


def get_bubble_detection_result_with_auto_directions(
    image_pil: Image.Image,
    conf_threshold: float = 0.6,
    detector_type: str = None,
    expand_ratio: float = 0,
    expand_top: float = 0,
    expand_bottom: float = 0,
    expand_left: float = 0,
    expand_right: float = 0,
    edge_ratio_threshold: float = None
) -> Dict[str, Any]:
    """
    获取气泡检测结果，并返回每个气泡的自动排版方向
    """
    if detector_type is None:
        detector_type = constants.DEFAULT_DETECTOR
    
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
        
        # 使用新的统一接口
        detection_result = detect(
            image_pil,
            detector_type=detector_type,
            merge_lines=True,
            edge_ratio_threshold=edge_ratio_threshold,
            expand_ratio=0,  # 先不扩展，最后统一扩展
            expand_top=0,
            expand_bottom=0,
            expand_left=0,
            expand_right=0
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
            elif hasattr(block, 'vertical') and block.vertical is not None:
                auto_dir = 'v' if block.vertical else 'h'
            else:
                auto_dir = 'v' if (y2 - y1) > (x2 - x1) else 'h'
            
            result['auto_directions'].append(auto_dir)
        
        # 对于 YOLOv5，使用 PaddleOCR 降级检测
        if detector_type == constants.DETECTOR_YOLOV5:
            for i, coord in enumerate(result['coords']):
                if not result['textlines_per_bubble'][i]:
                    textlines_info = detect_textlines_with_paddle_det(image_pil, coord)
                    result['textlines_per_bubble'][i] = textlines_info
                    if textlines_info:
                        result['auto_directions'][i] = analyze_direction_from_textlines(textlines_info)
        
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


# ========== 测试代码 ==========

if __name__ == '__main__':
    from PIL import Image
    from src.shared.path_helpers import resource_path
    import os
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("--- 测试气泡检测核心逻辑 (重构版) ---")
    test_image_path = resource_path('pic/before1.png')
    if os.path.exists(test_image_path):
        print(f"加载测试图片: {test_image_path}")
        try:
            img_pil = Image.open(test_image_path)
            print("开始检测坐标...")
            coords = get_bubble_coordinates(img_pil, conf_threshold=0.5)
            print(f"检测完成，找到 {len(coords)} 个气泡坐标:")
            for i, coord in enumerate(coords):
                print(f"  - 气泡 {i+1}: {coord}")
        except Exception as e:
            print(f"测试过程中发生错误: {e}")
    else:
        print(f"错误：测试图片未找到 {test_image_path}")
