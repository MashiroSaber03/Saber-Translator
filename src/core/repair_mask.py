"""Shared automatic-mask geometry and manual overrides; no model imports."""

import math
import cv2
import numpy as np

def _validate_bubble_geometry(bubble_coords, bubble_polygons=None):
    if not isinstance(bubble_coords, list):
        raise ValueError("气泡坐标必须是数组")
    if bubble_polygons is not None and not isinstance(bubble_polygons, list):
        raise ValueError("气泡多边形必须是数组")
    if bubble_polygons is not None and len(bubble_polygons) != len(bubble_coords):
        raise ValueError("气泡多边形数量与坐标数量不匹配")
    for index, coords in enumerate(bubble_coords):
        if (
            not isinstance(coords, (list, tuple))
            or len(coords) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in coords
            )
        ):
            raise ValueError(f"气泡 {index} 坐标必须包含四个整数")
        x1, y1, x2, y2 = coords
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"气泡 {index} 坐标必须描述正面积区域")
        if bubble_polygons is None or not bubble_polygons[index]:
            continue
        polygon = bubble_polygons[index]
        if (
            not isinstance(polygon, list)
            or len(polygon) != 4
            or any(
                not isinstance(point, list)
                or len(point) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in point
                )
                for point in polygon
            )
        ):
            raise ValueError(f"气泡 {index} 多边形必须包含四个整数点")


def create_bubble_mask(image_size, bubble_coords, bubble_polygons=None):
    """创建黑色为修复区域的二值掩膜。

    Args:
        image_size: 图像尺寸 (height, width, channels) 或 (height, width)
        bubble_coords: 当前文本框坐标列表 [(x1, y1, x2, y2), ...]
        bubble_polygons: 可选，气泡多边形坐标列表 [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ...]
                        如果提供，将使用多边形而不是矩形来创建掩码
    """
    if len(image_size) < 2 or image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError("图像尺寸无效")
    if not bubble_coords:
        return np.ones(image_size[:2], dtype=np.uint8) * 255
    _validate_bubble_geometry(bubble_coords, bubble_polygons)

    # 创建全白掩码（全部保留）
    mask = np.ones(image_size[:2], dtype=np.uint8) * 255

    for i, coords in enumerate(bubble_coords):
        x1, y1, x2, y2 = coords
        # 有解析后的当前多边形时，不再混入轴对齐边缘。
        if bubble_polygons is not None:
            polygon = bubble_polygons[i]
            if polygon:
                pts = np.array(polygon, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 0)
            else:
                cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
        else:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

    return mask

def build_automatic_mask_layers(
    image_size,
    bubble_coords,
    bubble_polygons=None,
    precise_mask=None,
    mask_dilate_size=0,
    mask_box_expand_ratio=0,
    method='lama',
    regional_inpainting=False,
):
    """Return box range and processed automatic mask (255=selected).

    Preview and repair share these stages. Without a precise mask, preserve the
    existing text-box fallback, which does not use expansion or dilation.
    """
    _validate_bubble_geometry(bubble_coords, bubble_polygons)
    if len(image_size) < 2 or image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError("图像尺寸无效")
    if isinstance(mask_dilate_size, bool) or not isinstance(mask_dilate_size, int) or mask_dilate_size < 0:
        raise ValueError("mask_dilate_size 必须是非负整数")
    if (isinstance(mask_box_expand_ratio, bool) or not isinstance(mask_box_expand_ratio, (int, float))
            or not math.isfinite(float(mask_box_expand_ratio)) or mask_box_expand_ratio < 0):
        raise ValueError("mask_box_expand_ratio 必须是非负有限数字")
    if not isinstance(regional_inpainting, bool):
        raise ValueError("regional_inpainting 必须是布尔值")
    shape = tuple(image_size[:2])
    if precise_mask is not None and (not isinstance(precise_mask, np.ndarray) or precise_mask.ndim != 2
                                    or precise_mask.dtype != np.uint8 or precise_mask.shape != shape):
        raise ValueError("精确文字掩膜必须是与原图同尺寸的 uint8 单通道数组")
    if precise_mask is None:
        box_mask = result = 255 - create_bubble_mask(image_size, bubble_coords, bubble_polygons)
    else:
        box_mask = np.zeros(shape, dtype=np.uint8)
        for index, (x1, y1, x2, y2) in enumerate(bubble_coords):
            polygon = (bubble_polygons[index] if bubble_polygons is not None and bubble_polygons[index]
                       else [[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            points = np.asarray(polygon, dtype=np.float32)
            center = points.mean(axis=0)
            points = center + (points - center) * (1 + mask_box_expand_ratio / 100)
            cv2.fillPoly(box_mask, [np.rint(points).astype(np.int32)], 255)
        result = np.where((precise_mask >= 128) & (box_mask > 0), 255, 0).astype(np.uint8)
        if mask_dilate_size:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mask_dilate_size * 2 + 1,) * 2)
            result = cv2.dilate(result, kernel)
        if method == 'lama' and regional_inpainting:
            radius = max(1, round(2 * min(shape) / 1024))
            result = cv2.morphologyEx(np.pad(result, radius), cv2.MORPH_CLOSE,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2))
            result = result[radius:-radius, radius:-radius]
    return box_mask, result


def build_repair_mask(
    image_size, bubble_coords, bubble_polygons=None, precise_mask=None,
    user_mask=None, mask_dilate_size=0, mask_box_expand_ratio=0,
    method='lama', regional_inpainting=False,
):
    """Return the final binary mask after automatic processing and user overrides."""
    _, automatic = build_automatic_mask_layers(
        image_size, bubble_coords, bubble_polygons, precise_mask,
        mask_dilate_size, mask_box_expand_ratio, method, regional_inpainting,
    )
    if user_mask is not None and (not isinstance(user_mask, np.ndarray) or user_mask.ndim != 2
                                 or user_mask.dtype != np.uint8 or user_mask.shape != automatic.shape):
        raise ValueError("用户掩膜必须是与原图同尺寸的 uint8 单通道数组")
    return apply_user_mask(automatic, user_mask)


def apply_user_mask(automatic, user_mask):
    result = automatic.copy()
    if user_mask is not None:
        result[user_mask > 200] = 255
        result[user_mask < 50] = 0
    return result
