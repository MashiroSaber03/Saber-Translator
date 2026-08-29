import logging
import math
import numpy as np
from PIL import Image, ImageDraw
import cv2 # 需要 cv2 来创建掩码

from src.interfaces.lama_interface import clean_image_with_lama

from src.shared import constants

logger = logging.getLogger("CoreInpainting")


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
    logger.debug(f"创建气泡掩码: {len(bubble_coords)} 个")
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

def inpaint_bubbles(image_pil, bubble_coords, method=constants.DEFAULT_INPAINT_METHOD, fill_color=None, bubble_polygons=None, precise_mask=None, user_mask=None, mask_dilate_size=0, mask_box_expand_ratio=0, lama_model='lama_mpe', disable_resize=False):
    """
    根据指定方法修复或填充图像中的气泡区域。

    Args:
        image_pil (PIL.Image.Image): 原始 PIL 图像。
        bubble_coords (list): 气泡坐标列表 [(x1, y1, x2, y2), ...]。
        method (str): 修复方法 ('solid', 'lama')。
        fill_color (str | None): 'solid' 方法使用的填充颜色；LaMA 不接受。
        bubble_polygons (list): 可选，气泡多边形坐标列表 [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ...]
                               如果提供，将使用多边形而不是矩形来创建掩码和填充
        precise_mask (np.ndarray): 可选，模型生成的精确文字掩膜（textMask）。
                                   如果提供，将直接使用此掩膜而非根据坐标生成。
                                   仅 CTD/Default 检测器支持生成此掩膜。
        user_mask (np.ndarray): 可选，用户笔刷掩膜（userMask）。
                                白色(255)=用户标记需要修复的区域
                                黑色(0)=用户标记需要保留的区域
                                灰色(127)=未修改，使用自动检测结果
        mask_dilate_size (int): 掩膜膨胀大小（像素），用于扩大修复区域。
        mask_box_expand_ratio (int): 标注框区域扩大比例（%），用于扩大标注框的收录范围。
        lama_model (str): LAMA 模型选择 'lama_mpe' (速度优化) 或 'litelama' (通用)
        disable_resize (bool): 是否禁止 LaMA 自动缩放。

    Returns:
        PIL.Image.Image: 处理后的 PIL 图像。
    """
    if not isinstance(image_pil, Image.Image):
        raise ValueError("修复输入必须是 PIL 图像")
    if not isinstance(bubble_coords, list):
        raise ValueError("气泡坐标必须是数组")
    if not bubble_coords:
        logger.debug("无气泡坐标，跳过修复")
        return image_pil.copy()

    _validate_bubble_geometry(bubble_coords, bubble_polygons)

    if method not in {"solid", "lama"}:
        raise ValueError(f"不支持的修复方法: {method}")
    if method == "solid":
        if not isinstance(fill_color, str) or not (
            len(fill_color) == 7
            and fill_color.startswith("#")
            and all(
                character in "0123456789abcdefABCDEF"
                for character in fill_color[1:]
            )
        ):
            raise ValueError("填充颜色必须是 #RRGGBB")
    elif fill_color is not None:
        raise ValueError("LaMA 修复不接受填充颜色")
    if method == "lama" and lama_model not in {"lama_mpe", "litelama"}:
        raise ValueError("LaMA 模型必须是 lama_mpe 或 litelama")
    if not isinstance(disable_resize, bool):
        raise ValueError("disable_resize 必须是布尔值")
    if isinstance(mask_dilate_size, bool) or not isinstance(mask_dilate_size, int) or mask_dilate_size < 0:
        raise ValueError("mask_dilate_size 必须是非负整数")
    if (
        isinstance(mask_box_expand_ratio, bool)
        or not isinstance(mask_box_expand_ratio, (int, float))
        or not math.isfinite(float(mask_box_expand_ratio))
        or mask_box_expand_ratio < 0
    ):
        raise ValueError("mask_box_expand_ratio 必须是非负有限数字")

    converted_image = image_pil.convert('RGB')
    try:
        image_size = np.array(converted_image).shape
    finally:
        if converted_image is not image_pil:
            converted_image.close()

    # 1. 创建掩码 (黑色为修复区)
    if precise_mask is not None:
        # 使用模型生成的精确文字掩膜
        logger.debug("使用精确文字掩膜")
        
        if (
            not isinstance(precise_mask, np.ndarray)
            or precise_mask.ndim != 2
            or precise_mask.dtype != np.uint8
            or precise_mask.shape != tuple(image_size[:2])
        ):
            raise ValueError("精确文字掩膜必须是与原图同尺寸的 uint8 单通道数组")
        
        # 反转掩膜：文字区域（高值）变为需要修复的区域（低值）
        text_mask = 255 - precise_mask
        
        # 应用阈值，确保是二值掩膜
        _, text_mask = cv2.threshold(text_mask, 127, 255, cv2.THRESH_BINARY)
        
        # 只保留标注框内的文字掩膜（只修复框出来的区域）
        # 创建一个标注框区域的掩膜
        box_region_mask = np.ones_like(text_mask) * 255  # 白色表示保留
        expand_ratio = mask_box_expand_ratio / 100.0  # 转换为小数
        
        for index, (x1, y1, x2, y2) in enumerate(bubble_coords):
            polygon = (
                bubble_polygons[index]
                if bubble_polygons is not None and bubble_polygons[index]
                else [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            )
            points = np.asarray(polygon, dtype=np.float32)
            if expand_ratio > 0:
                center = points.mean(axis=0)
                points = center + (points - center) * (1.0 + expand_ratio)
            cv2.fillPoly(
                box_region_mask,
                [np.rint(points).astype(np.int32)],
                0,
            )
        
        if mask_box_expand_ratio > 0:
            logger.debug(f"标注框扩大 {mask_box_expand_ratio}%")
        
        # 合并掩膜：只有在标注框内且是文字区域的才需要修复
        # text_mask: 黑色=文字区域（需修复），白色=非文字区域
        # box_region_mask: 黑色=标注框内，白色=标注框外
        # 结果：只有两者都是黑色时才需要修复
        bubble_mask_np = np.maximum(text_mask, box_region_mask)
        
        # 掩膜膨胀处理（问题2：膨胀系数）
        if mask_dilate_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mask_dilate_size * 2 + 1, mask_dilate_size * 2 + 1))
            # 膨胀需要修复的区域（黑色区域），所以先反转，膨胀，再反转
            inverted = 255 - bubble_mask_np
            dilated = cv2.dilate(inverted, kernel, iterations=1)
            bubble_mask_np = 255 - dilated
            logger.debug(f"掩膜膨胀: {mask_dilate_size}px")
        
    else:
        # 使用坐标/多边形生成掩膜
        bubble_mask_np = create_bubble_mask(image_size, bubble_coords, bubble_polygons)
    
    # ✅ 2. 叠加用户掩膜（不受标注框限制）
    if user_mask is not None:
        logger.debug("叠加用户笔刷掩膜")
        if (
            not isinstance(user_mask, np.ndarray)
            or user_mask.ndim != 2
            or user_mask.dtype != np.uint8
            or user_mask.shape != tuple(image_size[:2])
        ):
            raise ValueError("用户掩膜必须是与原图同尺寸的 uint8 单通道数组")
        
        # 统计用于调试
        white_count = np.sum(user_mask > 200)
        black_count = np.sum(user_mask < 50)
        gray_count = np.sum((user_mask >= 50) & (user_mask <= 200))
        logger.debug(f"用户掩膜统计: 白色(修复)={white_count}px, 黑色(保留)={black_count}px, 灰色(未改)={gray_count}px")
        
        # 叠加逻辑：
        # - user_mask 白色(>200) → bubble_mask_np 设为黑色（强制修复，不受标注框限制）
        # - user_mask 黑色(<50) → bubble_mask_np 设为白色（强制保留，不受标注框限制）
        # - user_mask 灰色 → 保持 bubble_mask_np 原值
        
        # 用户标记需要修复的区域（白色）→ 强制设为黑色
        user_repair_mask = user_mask > 200
        bubble_mask_np[user_repair_mask] = 0
        
        # 用户标记需要保留的区域（黑色）→ 强制设为白色
        user_preserve_mask = user_mask < 50
        bubble_mask_np[user_preserve_mask] = 255
        
        # 统计最终掩膜
        final_repair_count = np.sum(bubble_mask_np < 128)
        logger.debug(f"最终掩膜修复区域: {final_repair_count}px ({final_repair_count * 100 / bubble_mask_np.size:.2f}%)")
    
    if not np.any(bubble_mask_np < 128):
        raise ValueError("修复掩膜为空")

    bubble_mask_pil = Image.fromarray(bubble_mask_np)
    result_img = None
    try:
        if method == 'lama':
            logger.debug(f"使用 LAMA 修复 (模型: {lama_model})")
            result_img = clean_image_with_lama(
                image_pil,
                bubble_mask_pil,
                lama_model=lama_model,
                disable_resize=disable_resize,
            )
            if not isinstance(result_img, Image.Image):
                raise RuntimeError("LaMA 修复未返回图像")
            if result_img.size != image_pil.size:
                raise RuntimeError("LaMA 修复结果尺寸与输入图像不一致")
            logger.debug("LAMA 修复成功")
        else:
            result_img = image_pil.copy()
            use_mask = precise_mask is not None or user_mask is not None
            logger.debug(f"纯色填充: {fill_color}")
            if use_mask:
                converted_result = result_img.convert('RGB')
                try:
                    result_np = np.array(converted_result)
                finally:
                    if converted_result is not result_img:
                        converted_result.close()

                r = int(fill_color[1:3], 16)
                g = int(fill_color[3:5], 16)
                b = int(fill_color[5:7], 16)

                fill_mask = bubble_mask_np < 128
                result_np[fill_mask] = [r, g, b]
                replacement = Image.fromarray(result_np)
                result_img.close()
                result_img = replacement
                logger.debug("精确掩膜填充完成")
            else:
                draw = ImageDraw.Draw(result_img)
                for i, (x1, y1, x2, y2) in enumerate(bubble_coords):
                    if bubble_polygons is not None:
                        polygon = bubble_polygons[i]
                        if polygon:
                            pts = [(p[0], p[1]) for p in polygon]
                            draw.polygon(pts, fill=fill_color)
                            continue
                    draw.rectangle(((x1, y1), (x2, y2)), fill=fill_color)
            logger.debug("纯色填充完成")

        return result_img
    except Exception:
        if isinstance(result_img, Image.Image):
            result_img.close()
        raise
    finally:
        bubble_mask_pil.close()
