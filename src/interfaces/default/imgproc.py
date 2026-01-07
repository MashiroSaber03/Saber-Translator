"""
图像预处理工具
"""

import cv2
import numpy as np


def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    """
    保持宽高比缩放图像，并填充到 256 的倍数
    
    Args:
        img: 输入图像
        square_size: 目标尺寸
        interpolation: 插值方法
        mag_ratio: 放大比例
        
    Returns:
        resized: 缩放后的图像
        ratio: 缩放比例
        size_heatmap: 热图尺寸
        pad_w: 宽度填充
        pad_h: 高度填充
    """
    height, width, channel = img.shape
    target_size = mag_ratio * square_size
    ratio = target_size / max(height, width)

    target_h, target_w = int(round(height * ratio)), int(round(width * ratio))
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # 填充到 256 的倍数
    MULT = 256
    target_h32, target_w32 = target_h, target_w
    pad_h = pad_w = 0
    
    if target_h % MULT != 0:
        pad_h = (MULT - target_h % MULT)
        target_h32 = target_h + pad_h
    if target_w % MULT != 0:
        pad_w = (MULT - target_w % MULT)
        target_w32 = target_w + pad_w

    resized = np.zeros((target_h32, target_w32, channel), dtype=np.uint8)
    resized[0:target_h, 0:target_w, :] = proc

    size_heatmap = (int(target_w32 / 2), int(target_h32 / 2))
    return resized, ratio, size_heatmap, pad_w, pad_h


def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2):
    """
    调整坐标到原始图像尺寸
    
    Args:
        polys: 多边形坐标列表
        ratio_w: 宽度比例
        ratio_h: 高度比例
        ratio_net: 网络输出比例
        
    Returns:
        调整后的多边形坐标
    """
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys
