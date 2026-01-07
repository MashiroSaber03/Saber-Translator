"""
超长图片切割与拼接模块

用于处理极端长宽比的漫画图片（如长条漫画、双页漫画）
"""

import logging
from typing import Tuple, List, Optional, Any
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger("ImageRearrange")


# ========== 配置常量 ==========

REARRANGE_DOWNSCALE_RATIO_THRESHOLD = 2.5
REARRANGE_ASPECT_RATIO_THRESHOLD = 3.0
DEFAULT_TARGET_SIZE = 1536


@dataclass
class PatchInfo:
    """单个切片的信息"""
    index: int      # 切片索引
    top: int        # 在原图中的顶部位置
    bottom: int     # 在原图中的底部位置
    rel_top: float  # 相对顶部位置比例


@dataclass
class RearrangeContext:
    """重排上下文，保存切割信息以便后续坐标转换"""
    is_rearranged: bool = False
    original_height: int = 0
    original_width: int = 0
    transpose: bool = False
    
    patches_info: List[PatchInfo] = field(default_factory=list)
    down_scale_ratios: List[float] = field(default_factory=list)
    pad_sizes: List[Tuple[int, int]] = field(default_factory=list)


def square_pad_resize(img: np.ndarray, tgt_size: int) -> Tuple[np.ndarray, float, int, int]:
    """
    将图像填充成正方形并缩放到目标尺寸
    
    Args:
        img: 输入图像 (H, W, C)
        tgt_size: 目标尺寸
        
    Returns:
        img_padded: 处理后的图像
        down_scale_ratio: 缩放比例
        pad_h: 高度填充量
        pad_w: 宽度填充量
    """
    h, w = img.shape[:2]
    pad_h, pad_w = 0, 0

    # 填充成正方形
    if w < h:
        pad_w = h - w
        w += pad_w
    elif h < w:
        pad_h = w - h
        h += pad_h

    # 如果尺寸小于目标尺寸，继续填充
    pad_size = tgt_size - h
    if pad_size > 0:
        pad_h += pad_size
        pad_w += pad_size

    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # 缩放
    down_scale_ratio = tgt_size / img.shape[0]
    if down_scale_ratio < 1:
        img = cv2.resize(img, (tgt_size, tgt_size), interpolation=cv2.INTER_LINEAR)
    else:
        down_scale_ratio = 1.0

    return img, down_scale_ratio, pad_h, pad_w


def check_needs_rearrange(
    img: np.ndarray,
    tgt_size: int = DEFAULT_TARGET_SIZE,
    downscale_threshold: float = REARRANGE_DOWNSCALE_RATIO_THRESHOLD,
    aspect_threshold: float = REARRANGE_ASPECT_RATIO_THRESHOLD
) -> Tuple[bool, bool]:
    """
    检查图像是否需要重排处理
    
    Args:
        img: 输入图像
        tgt_size: 目标尺寸
        downscale_threshold: 缩放比阈值
        aspect_threshold: 长宽比阈值
        
    Returns:
        needs_rearrange: 是否需要重排
        transpose: 是否需要转置（横向长图）
    """
    h, w = img.shape[:2]
    
    transpose = False
    if h < w:
        transpose = True
        h, w = w, h
    
    asp_ratio = h / w if w > 0 else 0
    down_scale_ratio = h / tgt_size if tgt_size > 0 else 0
    
    needs_rearrange = down_scale_ratio > downscale_threshold and asp_ratio > aspect_threshold
    
    if needs_rearrange:
        logger.info(
            f"图像需要重排处理: 尺寸=({img.shape[1]}x{img.shape[0]}), "
            f"缩放比={down_scale_ratio:.2f}, 长宽比={asp_ratio:.2f}, 转置={transpose}"
        )
    
    return needs_rearrange, transpose


def slice_image_for_detection(
    img: np.ndarray,
    tgt_size: int = DEFAULT_TARGET_SIZE,
    verbose: bool = False
) -> Tuple[List[np.ndarray], RearrangeContext]:
    """
    将超长图片切割成多个独立切片
    
    Args:
        img: 输入图像 (H, W, C)
        tgt_size: 目标尺寸
        verbose: 是否输出调试信息
        
    Returns:
        patches: 切片列表，每个切片已经过 pad 和 resize
        context: 重排上下文
    """
    needs_rearrange, transpose = check_needs_rearrange(img, tgt_size)
    
    if not needs_rearrange:
        return [], RearrangeContext(
            is_rearranged=False,
            original_height=img.shape[0],
            original_width=img.shape[1]
        )
    
    original_h, original_w = img.shape[:2]
    
    # 如果是横向长图，先转置
    if transpose:
        img = np.transpose(img, (1, 0, 2))
    
    h, w = img.shape[:2]
    
    # 计算切片参数
    overlap_ratio = 0.2  # 20% 重叠
    effective_size = int(tgt_size * (1 - overlap_ratio))
    num_patches = max(1, int(np.ceil((h - tgt_size * overlap_ratio) / effective_size)))
    
    # 计算步长
    step = (h - tgt_size) / (num_patches - 1) if num_patches > 1 else 0
    
    patches_info = []
    patches = []
    down_scale_ratios = []
    pad_sizes = []
    
    for i in range(num_patches):
        top = int(i * step)
        bottom = min(top + tgt_size, h)
        
        # 如果最后一个切片太短，调整 top
        if bottom - top < tgt_size // 2 and i > 0:
            top = max(0, h - tgt_size)
            bottom = h
        
        patch = img[top:bottom, :, :]
        patch_resized, dsr, pad_h, pad_w = square_pad_resize(patch, tgt_size)
        
        patches_info.append(PatchInfo(index=i, top=top, bottom=bottom, rel_top=top / h))
        patches.append(patch_resized)
        down_scale_ratios.append(dsr)
        pad_sizes.append((pad_h, pad_w))
        
        if verbose:
            logger.info(f"切片 {i}: top={top}, bottom={bottom}, 缩放比={dsr:.4f}")
    
    context = RearrangeContext(
        is_rearranged=True,
        original_height=original_h,
        original_width=original_w,
        transpose=transpose,
        patches_info=patches_info,
        down_scale_ratios=down_scale_ratios,
        pad_sizes=pad_sizes
    )
    
    logger.info(f"图像已切割: {num_patches} 个切片")
    
    return patches, context


def transform_textlines_to_original(
    textlines: List[Any],
    patch_index: int,
    context: RearrangeContext
) -> List[Any]:
    """
    将切片中检测到的文本行坐标转换回原图坐标
    
    Args:
        textlines: TextLine 对象列表
        patch_index: 切片索引
        context: 重排上下文
        
    Returns:
        转换后的 TextLine 列表
    """
    if not context.is_rearranged or patch_index >= len(context.patches_info):
        return textlines
    
    patch_info = context.patches_info[patch_index]
    dsr = context.down_scale_ratios[patch_index] if patch_index < len(context.down_scale_ratios) else 1.0
    
    from src.core.detector.data_types import TextLine
    
    transformed = []
    for tl in textlines:
        pts = tl.pts.astype(np.float64)
        
        # 1. 反向缩放
        pts = pts / dsr
        
        # 2. 加上切片偏移
        pts[:, 1] += patch_info.top
        
        # 3. 如果转置过，交换 x, y 坐标
        if context.transpose:
            pts = pts[:, ::-1].copy()
        
        # 4. 裁剪到原图范围
        pts[:, 0] = np.clip(pts[:, 0], 0, context.original_width)
        pts[:, 1] = np.clip(pts[:, 1], 0, context.original_height)
        
        try:
            new_tl = TextLine(
                pts=pts.astype(np.int32),
                confidence=tl.confidence,
                text=tl.text,
                fg_color=tl.fg_color,
                bg_color=tl.bg_color
            )
            transformed.append(new_tl)
        except Exception as e:
            logger.debug(f"坐标转换失败: {e}")
    
    return transformed


def merge_masks_from_patches(
    masks: List[np.ndarray],
    context: RearrangeContext
) -> Optional[np.ndarray]:
    """
    将多个切片的掩码合并成原图大小的掩码
    
    Args:
        masks: 每个切片的掩码列表
        context: 重排上下文
        
    Returns:
        合并后的掩码 (uint8, 0-255)
    """
    if not masks or not context.is_rearranged:
        return None
    
    # 创建画布（转置后的空间）
    if context.transpose:
        canvas_h = context.original_width
        canvas_w = context.original_height
    else:
        canvas_h = context.original_height
        canvas_w = context.original_width
    
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    canvas_count = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    
    for i, mask in enumerate(masks):
        if i >= len(context.patches_info):
            break
        
        patch_info = context.patches_info[i]
        dsr = context.down_scale_ratios[i] if i < len(context.down_scale_ratios) else 1.0
        pad_h, pad_w = context.pad_sizes[i] if i < len(context.pad_sizes) else (0, 0)
        
        # 确保掩码是 2D
        if mask.ndim == 3:
            mask = mask.squeeze()
        
        patch_h = patch_info.bottom - patch_info.top
        patch_w = canvas_w
        
        # 反向缩放掩码
        if dsr < 1.0:
            padded_size = int(mask.shape[0] / dsr)
            mask_upscaled = cv2.resize(mask, (padded_size, padded_size), interpolation=cv2.INTER_LINEAR)
            
            valid_h = min(padded_size - pad_h, mask_upscaled.shape[0])
            valid_w = min(padded_size - pad_w, mask_upscaled.shape[1])
            mask_no_pad = mask_upscaled[:valid_h, :valid_w]
            
            if mask_no_pad.size > 0:
                mask_final = cv2.resize(mask_no_pad, (patch_w, patch_h), interpolation=cv2.INTER_LINEAR)
            else:
                mask_final = np.zeros((patch_h, patch_w), dtype=np.float32)
        else:
            mask_final = cv2.resize(mask, (patch_w, patch_h), interpolation=cv2.INTER_LINEAR)
        
        # 放到画布上
        top = patch_info.top
        bottom = min(top + mask_final.shape[0], canvas_h)
        actual_h = bottom - top
        actual_w = min(mask_final.shape[1], canvas_w)
        
        canvas[top:bottom, :actual_w] += mask_final[:actual_h, :actual_w]
        canvas_count[top:bottom, :actual_w] += 1
    
    # 平均重叠区域
    canvas_count = np.maximum(canvas_count, 1)
    canvas = canvas / canvas_count
    
    # 如果转置过，转回原始方向
    if context.transpose:
        canvas = canvas.T
    
    return np.clip(canvas * 255, 0, 255).astype(np.uint8)
