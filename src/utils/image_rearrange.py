"""
超长图片切割与拼接模块

用于处理极端长宽比的漫画图片（如长条漫画、双页漫画）
"""

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.core.detector.data_types import TextLine

logger = logging.getLogger("ImageRearrange")


# ========== 配置常量 ==========

REARRANGE_DOWNSCALE_RATIO_THRESHOLD = 2.5
REARRANGE_ASPECT_RATIO_THRESHOLD = 3.0
DEFAULT_TARGET_SIZE = 1536


@dataclass(slots=True)
class PatchInfo:
    """单个切片的信息"""
    top: int        # 在原图中的顶部位置
    bottom: int     # 在原图中的底部位置
    down_scale_ratio: float
    pad_height: int
    pad_width: int


@dataclass(slots=True)
class RearrangeContext:
    """重排上下文，保存切割信息以便后续坐标转换"""
    is_rearranged: bool = False
    original_height: int = 0
    original_width: int = 0
    transpose: bool = False

    patches_info: list[PatchInfo] = field(default_factory=list)


def square_pad_resize(
    img: np.ndarray,
    tgt_size: int,
) -> tuple[np.ndarray, float, int, int]:
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
    if (
        not isinstance(img, np.ndarray)
        or img.ndim != 3
        or img.shape[2] != 3
        or img.size == 0
    ):
        raise ValueError("长图切片输入必须是非空三通道图像")
    if isinstance(tgt_size, bool) or not isinstance(tgt_size, int) or tgt_size <= 0:
        raise ValueError("长图切片目标尺寸必须是正整数")
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
) -> tuple[bool, bool]:
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
    if (
        not isinstance(img, np.ndarray)
        or img.ndim != 3
        or img.shape[2] != 3
        or img.size == 0
    ):
        raise ValueError("长图检测输入必须是非空三通道图像")
    if isinstance(tgt_size, bool) or not isinstance(tgt_size, int) or tgt_size <= 0:
        raise ValueError("长图检测目标尺寸必须是正整数")
    for label, value in (
        ("缩放", downscale_threshold),
        ("长宽比", aspect_threshold),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"长图检测{label}阈值必须是正数")
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
) -> tuple[list[np.ndarray], RearrangeContext]:
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

    for i in range(num_patches):
        top = int(i * step)
        bottom = min(top + tgt_size, h)
        
        # 如果最后一个切片太短，调整 top
        if bottom - top < tgt_size // 2 and i > 0:
            top = max(0, h - tgt_size)
            bottom = h
        
        patch = img[top:bottom, :, :]
        patch_resized, dsr, pad_h, pad_w = square_pad_resize(patch, tgt_size)
        
        patches_info.append(
            PatchInfo(
                top=top,
                bottom=bottom,
                down_scale_ratio=dsr,
                pad_height=pad_h,
                pad_width=pad_w,
            )
        )
        patches.append(patch_resized)
        
    context = RearrangeContext(
        is_rearranged=True,
        original_height=original_h,
        original_width=original_w,
        transpose=transpose,
        patches_info=patches_info,
    )
    
    logger.info(f"图像已切割: {num_patches} 个切片")
    
    return patches, context


def transform_textlines_to_original(
    textlines: list[TextLine],
    patch_index: int,
    context: RearrangeContext
) -> list[TextLine]:
    """
    将切片中检测到的文本行坐标转换回原图坐标
    
    Args:
        textlines: TextLine 对象列表
        patch_index: 切片索引
        context: 重排上下文
        
    Returns:
        转换后的 TextLine 列表
    """
    if not context.is_rearranged:
        return list(textlines)
    if patch_index < 0 or patch_index >= len(context.patches_info):
        raise IndexError(f"切片索引越界: {patch_index}")

    patch_info = context.patches_info[patch_index]
    dsr = patch_info.down_scale_ratio
    if dsr <= 0:
        raise ValueError("长图切片缩放比例必须大于零")
    
    if not isinstance(textlines, list) or any(
        not isinstance(line, TextLine) for line in textlines
    ):
        raise TypeError("待还原文本行必须是 TextLine 列表")
    
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
        
        transformed.append(
            TextLine(
                pts=pts.astype(np.int32),
                confidence=tl.confidence,
                text=tl.text,
                fg_color=tl.fg_color,
                bg_color=tl.bg_color,
            )
        )
    
    return transformed


def merge_masks_from_patches(
    masks: list[np.ndarray | None],
    context: RearrangeContext
) -> np.ndarray | None:
    """
    将多个切片的掩码合并成原图大小的掩码
    
    Args:
        masks: 每个切片的掩码列表
        context: 重排上下文
        
    Returns:
        合并后的掩码 (uint8, 0-255)
    """
    if not context.is_rearranged:
        return None
    if context.original_width <= 0 or context.original_height <= 0:
        raise ValueError("长图切片上下文缺少原图尺寸")
    if len(masks) != len(context.patches_info):
        raise ValueError("切片掩码数量与切片上下文不一致")
    if not any(mask is not None for mask in masks):
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
        if mask is None:
            continue
        if not isinstance(mask, np.ndarray):
            raise TypeError("切片掩码必须是 numpy 数组")
        
        patch_info = context.patches_info[i]
        dsr = patch_info.down_scale_ratio
        pad_h = patch_info.pad_height
        pad_w = patch_info.pad_width
        if dsr <= 0 or pad_h < 0 or pad_w < 0:
            raise ValueError("长图切片上下文包含无效尺寸")
        
        # 确保掩码是 2D
        if mask.ndim == 3 and 1 in mask.shape:
            mask = np.squeeze(mask)
        if mask.ndim != 2:
            raise ValueError("切片掩码必须是二维数组")
        if not np.issubdtype(mask.dtype, np.number) or not np.isfinite(mask).all():
            raise ValueError("切片掩码必须包含有限数字")
        if np.any(mask < 0) or np.any(mask > 255):
            raise ValueError("切片掩码像素必须位于 0 到 255")
        
        # 切片在原图中的尺寸
        patch_h = patch_info.bottom - patch_info.top
        patch_w = canvas_w  # 切片宽度等于原图宽度（纵向切割）
        if patch_info.top < 0 or patch_info.bottom > canvas_h or patch_h <= 0:
            raise ValueError("长图切片位置超出原图范围")
        
        # 反向缩放掩码
        if dsr < 1.0:
            padded_size = int(round(mask.shape[0] / dsr))
            mask_upscaled = cv2.resize(
                mask,
                (padded_size, padded_size),
                interpolation=cv2.INTER_LINEAR,
            )
            valid_h = padded_size - pad_h
            valid_w = padded_size - pad_w
            valid_h = max(0, min(valid_h, mask_upscaled.shape[0]))
            valid_w = max(0, min(valid_w, mask_upscaled.shape[1]))
        else:
            mask_upscaled = mask
            valid_h = mask.shape[0] - pad_h
            valid_w = mask.shape[1] - pad_w
            valid_h = max(0, min(valid_h, mask.shape[0]))
            valid_w = max(0, min(valid_w, mask.shape[1]))
        if valid_h <= 0 or valid_w <= 0:
            raise ValueError("切片掩码有效区域为空")
        mask_final = cv2.resize(
            mask_upscaled[:valid_h, :valid_w],
            (patch_w, patch_h),
            interpolation=cv2.INTER_LINEAR,
        )
        
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
    
    return np.clip(canvas, 0, 255).astype(np.uint8)
