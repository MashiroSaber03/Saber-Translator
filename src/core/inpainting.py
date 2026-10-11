import logging
import numpy as np
from PIL import Image

from src.interfaces.lama_interface import clean_image_with_lama

from src.shared import constants

logger = logging.getLogger("CoreInpainting")


from src.core.repair_mask import create_bubble_mask, build_repair_mask


def inpaint_bubbles(image_pil, bubble_coords, method=constants.DEFAULT_INPAINT_METHOD, fill_color=None, bubble_polygons=None, precise_mask=None, user_mask=None, mask_dilate_size=0, mask_box_expand_ratio=0, lama_model='lama_mpe', disable_resize=False, regional_inpainting=False):
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
                                   先按当前文本框裁切，再应用膨胀及局部修复设置。
                                   手动修复和保留标记最后覆盖自动区域。
        user_mask (np.ndarray): 可选，用户笔刷掩膜（userMask）。
                                白色(255)=用户标记需要修复的区域
                                黑色(0)=用户标记需要保留的区域
                                灰色(127)=未修改，使用自动检测结果
        mask_dilate_size (int): 掩膜膨胀大小（像素），用于扩大修复区域。
        mask_box_expand_ratio (int): 标注框区域扩大比例（%），用于扩大标注框的收录范围。
        lama_model (str): 'lama_mpe' (速度优化)、'litelama' (通用) 或 'lama_manga' (漫画)
        disable_resize (bool): 是否禁止 LaMA 自动缩放。
        regional_inpainting (bool): 是否按局部区域保留上下文并分别修复。

    Returns:
        PIL.Image.Image: 处理后的 PIL 图像。
    """
    if not isinstance(image_pil, Image.Image):
        raise ValueError("修复输入必须是 PIL 图像")
    if not isinstance(bubble_coords, list):
        raise ValueError("气泡坐标必须是数组")
    if not bubble_coords and user_mask is None:
        logger.debug("无气泡坐标，跳过修复")
        return image_pil.copy()

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
    if method == "lama" and lama_model not in {"lama_mpe", "litelama", "lama_manga"}:
        raise ValueError("LaMA 模型必须是 lama_mpe、litelama 或 lama_manga")
    if not isinstance(disable_resize, bool):
        raise ValueError("disable_resize 必须是布尔值")
    repair_mask = build_repair_mask(
        (image_pil.height, image_pil.width), bubble_coords, bubble_polygons,
        precise_mask, user_mask, mask_dilate_size, mask_box_expand_ratio,
        method, regional_inpainting,
    )
    if not np.any(repair_mask):
        return image_pil.copy()
    bubble_mask_pil = Image.fromarray(255 - repair_mask)
    result_img = None
    try:
        if method == 'lama':
            logger.debug(f"使用 LAMA 修复 (模型: {lama_model})")
            result_img = clean_image_with_lama(
                image_pil,
                bubble_mask_pil,
                lama_model=lama_model,
                disable_resize=disable_resize,
                regional_inpainting=regional_inpainting,
            )
            if not isinstance(result_img, Image.Image):
                raise RuntimeError("LaMA 修复未返回图像")
            if result_img.size != image_pil.size:
                raise RuntimeError("LaMA 修复结果尺寸与输入图像不一致")
            logger.debug("LAMA 修复成功")
        else:
            result_img = image_pil.convert('RGB')
            with Image.fromarray(repair_mask) as fill_mask:
                result_img.paste(fill_color, mask=fill_mask)

        return result_img
    except Exception:
        if isinstance(result_img, Image.Image):
            result_img.close()
        raise
    finally:
        bubble_mask_pil.close()
