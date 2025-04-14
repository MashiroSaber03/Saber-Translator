import logging
import os
import numpy as np
from PIL import Image, ImageDraw
import cv2 # 需要 cv2 来创建掩码

# 导入接口和常量
# 尝试导入接口，如果失败则标记为不可用
try:
    from src.interfaces.migan_interface import get_migan_inpainter
    _MIGAN_AVAILABLE = True
except ImportError:
    _MIGAN_AVAILABLE = False
    get_migan_inpainter = None # 定义一个空函数避免 NameError

try:
    from src.interfaces.lama_interface import clean_image_with_lama, is_lama_available
except ImportError:
    is_lama_available = lambda: False # 定义一个返回 False 的函数
    clean_image_with_lama = None # 定义一个空函数

from src.shared import constants
from src.shared.path_helpers import get_debug_dir, resource_path # 导入 resource_path 用于测试

logger = logging.getLogger("CoreInpainting")
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def is_migan_available():
    """检查 MI-GAN 功能是否可用 (基于导入成功)"""
    return _MIGAN_AVAILABLE

def create_bubble_mask(image_size, bubble_coords):
    """
    为气泡创建掩码图像 (黑色区域为修复区)。
    """
    logger.info(f"创建气泡掩码，图像大小：{image_size}, 气泡数量：{len(bubble_coords)}")
    if not bubble_coords:
        return np.ones(image_size[:2], dtype=np.uint8) * 255

    mask = np.ones(image_size[:2], dtype=np.uint8) * 255
    padding_ratio = 0.02
    min_padding = 1

    for x1, y1, x2, y2 in bubble_coords:
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0: continue

        padding_w = max(min_padding, int(width * padding_ratio))
        padding_h = max(min_padding, int(height * padding_ratio))

        inner_x1 = min(x1 + padding_w, x2 - 1)
        inner_y1 = min(y1 + padding_h, y2 - 1)
        inner_x2 = max(x2 - padding_w, x1 + 1)
        inner_y2 = max(y2 - padding_h, y1 + 1)

        if inner_x1 < inner_x2 and inner_y1 < inner_y2:
             cv2.rectangle(mask, (inner_x1, inner_y1), (inner_x2, inner_y2), 0, -1)

    try:
        debug_dir = get_debug_dir("inpainting_masks")
        cv2.imwrite(os.path.join(debug_dir, "bubble_mask_core.png"), mask)
    except Exception as save_e:
        logger.warning(f"保存修复掩码调试图像失败: {save_e}")

    return mask

def inpaint_bubbles(image_pil, bubble_coords, method='solid', fill_color=constants.DEFAULT_FILL_COLOR,
                    migan_strength=constants.DEFAULT_INPAINTING_STRENGTH, migan_blend_edges=True):
    """
    根据指定方法修复或填充图像中的气泡区域。

    Args:
        image_pil (PIL.Image.Image): 原始 PIL 图像。
        bubble_coords (list): 气泡坐标列表 [(x1, y1, x2, y2), ...]。
        method (str): 修复方法 ('solid', 'migan', 'lama')。
        fill_color (str): 'solid' 方法使用的填充颜色。
        migan_strength (float): MI-GAN 修复强度。
        migan_blend_edges (bool): MI-GAN 是否进行边缘融合。

    Returns:
        PIL.Image.Image: 处理后的 PIL 图像。
        PIL.Image.Image or None: 清理后的背景图像（如果修复成功），否则为 None。
    """
    if not bubble_coords:
        logger.info("没有气泡坐标，无需修复/填充。")
        return image_pil.copy(), None # 返回原图副本和无干净背景

    try:
        img_np = np.array(image_pil.convert('RGB'))
        image_size = img_np.shape[:2]
    except Exception as e:
         logger.error(f"无法将输入图像转换为 NumPy 数组: {e}", exc_info=True)
         return image_pil.copy(), None

    # 1. 创建掩码 (黑色为修复区)
    bubble_mask_np = create_bubble_mask(image_size, bubble_coords)
    bubble_mask_pil = Image.fromarray(bubble_mask_np)

    result_img = image_pil.copy()
    clean_background = None
    inpainting_successful = False # 标记修复是否成功

    # 2. 根据方法进行处理
    if method == 'lama' and is_lama_available() and clean_image_with_lama:
        logger.info("使用 LAMA 接口进行修复...")
        try:
            mask_np_inverted = 255 - bubble_mask_np
            inverted_mask_pil = Image.fromarray(mask_np_inverted)
            repaired_img = clean_image_with_lama(image_pil, inverted_mask_pil)
            if repaired_img:
                result_img = repaired_img
                clean_background = result_img.copy()
                setattr(result_img, '_lama_inpainted', True)
                inpainting_successful = True
                logger.info("LAMA 修复成功。")
            else:
                logger.error("LAMA 修复执行失败，未返回结果。将回退。")
        except Exception as e:
             logger.error(f"LAMA 修复过程中出错: {e}", exc_info=True)
             logger.info("LAMA 出错，将回退。")

    # 如果 LAMA 失败或未选择 LAMA，尝试 MI-GAN
    if not inpainting_successful and method == 'migan' and is_migan_available() and get_migan_inpainter:
        logger.info(f"使用 MI-GAN 接口进行修复 (强度: {migan_strength}, 融合: {migan_blend_edges})...")
        try:
            inpainter = get_migan_inpainter()
            if inpainter:
                repaired_img = inpainter.inpaint(
                    image_pil,
                    bubble_mask_pil,
                    blend_edges=migan_blend_edges,
                    strength=migan_strength
                )
                if repaired_img:
                    result_img = repaired_img
                    clean_background = result_img.copy()
                    setattr(result_img, '_migan_inpainted', True)
                    inpainting_successful = True
                    logger.info("MI-GAN 修复成功。")
                else:
                    logger.error("MI-GAN 修复执行失败，未返回结果。将回退。")
            else:
                 logger.error("无法获取 MI-GAN 实例。将回退。")
        except Exception as e:
             logger.error(f"MI-GAN 修复过程中出错: {e}", exc_info=True)
             logger.info("MI-GAN 出错，将回退。")

    # 如果修复未成功或选择了纯色填充
    if not inpainting_successful:
        logger.info(f"执行纯色填充，颜色: {fill_color}")
        # 确保在 result_img 上绘制（可能是原图副本，也可能是修复失败后的图）
        try:
            draw = ImageDraw.Draw(result_img)
            for x1, y1, x2, y2 in bubble_coords:
                if x1 < x2 and y1 < y2: # 检查坐标有效性
                    draw.rectangle(((x1, y1), (x2, y2)), fill=fill_color)
                else:
                    logger.warning(f"跳过无效坐标进行纯色填充: ({x1},{y1},{x2},{y2})")
            # 对于纯色填充，也生成一个"干净"背景的副本
            clean_background = result_img.copy()
            logger.info("纯色填充完成，已生成对应的'干净'背景。")
        except Exception as draw_e:
             logger.error(f"纯色填充时出错: {draw_e}", exc_info=True)
             # 如果绘制失败，至少返回原始图像副本
             result_img = image_pil.copy()
             clean_background = None


    # 保存调试图像
    try:
        debug_dir = get_debug_dir("inpainting_results")
        final_method = method if inpainting_successful else 'solid_fallback'
        result_img.save(os.path.join(debug_dir, f"inpainted_result_{final_method}.png"))
        if clean_background:
            # 将干净背景标记附加到主结果图像对象上
            setattr(result_img, '_clean_background', clean_background)
            setattr(result_img, '_clean_image', clean_background)
            clean_background.save(os.path.join(debug_dir, f"clean_background_{final_method}.png"))
    except Exception as save_e:
        logger.warning(f"保存修复结果调试图像失败: {save_e}")

    return result_img, clean_background

# --- 测试代码 ---
if __name__ == '__main__':
    from PIL import Image, ImageDraw # 需要导入 ImageDraw
    # 假设 detection 模块已完成
    try:
        from detection import get_bubble_coordinates
    except ImportError:
        print("错误：无法导入 detection 模块，请确保该模块已创建并包含 get_bubble_coordinates 函数。")
        get_bubble_coordinates = None # 设置为 None 以跳过依赖检测的测试

    print("--- 测试修复/填充核心逻辑 ---")
    test_image_path = resource_path('pic/before1.png')

    if os.path.exists(test_image_path) and get_bubble_coordinates:
        print(f"加载测试图片: {test_image_path}")
        try:
            img_pil = Image.open(test_image_path)
            print("获取气泡坐标...")
            coords = get_bubble_coordinates(img_pil)

            if coords:
                print(f"找到 {len(coords)} 个气泡。")

                # 测试纯色填充
                print("\n测试纯色填充...")
                filled_img, clean_solid = inpaint_bubbles(img_pil, coords, method='solid', fill_color='#FF0000')
                if filled_img:
                    save_path = get_debug_dir("test_result_solid.png")
                    filled_img.save(save_path)
                    print(f"纯色填充结果已保存到: {save_path}")
                if clean_solid:
                     save_path_clean = get_debug_dir("test_clean_solid.png")
                     clean_solid.save(save_path_clean)
                     print(f"纯色填充干净背景已保存到: {save_path_clean}")

                # 测试 MI-GAN
                print("\n测试 MI-GAN...")
                if is_migan_available():
                    migan_img, clean_migan = inpaint_bubbles(img_pil, coords, method='migan', migan_strength=1.5)
                    if migan_img:
                        save_path = get_debug_dir("test_result_migan.png")
                        migan_img.save(save_path)
                        print(f"MI-GAN 结果已保存到: {save_path}")
                    if clean_migan:
                         save_path_clean = get_debug_dir("test_clean_migan.png")
                         clean_migan.save(save_path_clean)
                         print(f"MI-GAN 干净背景已保存到: {save_path_clean}")
                else:
                    print("MI-GAN 不可用，跳过测试。")

                # 测试 LAMA
                print("\n测试 LAMA...")
                if is_lama_available():
                    lama_img, clean_lama = inpaint_bubbles(img_pil, coords, method='lama')
                    if lama_img:
                        save_path = get_debug_dir("test_result_lama.png")
                        lama_img.save(save_path)
                        print(f"LAMA 结果已保存到: {save_path}")
                    if clean_lama:
                         save_path_clean = get_debug_dir("test_clean_lama.png")
                         clean_lama.save(save_path_clean)
                         print(f"LAMA 干净背景已保存到: {save_path_clean}")
                else:
                    print("LAMA 不可用，跳过测试。")

            else:
                print("未找到气泡，无法测试修复。")
        except Exception as e:
            print(f"测试过程中发生错误: {e}")
    elif not get_bubble_coordinates:
         print("跳过修复测试，因为 detection 模块不可用。")
    else:
        print(f"错误：测试图片未找到 {test_image_path}")