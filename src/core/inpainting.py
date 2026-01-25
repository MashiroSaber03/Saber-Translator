import logging
import os
import numpy as np
from PIL import Image, ImageDraw
import cv2 # 需要 cv2 来创建掩码

# 导入接口和常量
# 尝试导入接口，如果失败则标记为不可用
try:
    from src.interfaces.lama_interface import clean_image_with_lama, is_lama_available
except ImportError:
    is_lama_available = lambda: False # 定义一个返回 False 的函数
    clean_image_with_lama = None # 定义一个空函数

from src.shared import constants
from src.shared.path_helpers import get_debug_dir, resource_path # 导入 resource_path 用于测试

logger = logging.getLogger("CoreInpainting")
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def create_bubble_mask(image_size, bubble_coords, bubble_polygons=None):
    """
    为气泡创建掩码图像 (黑色区域为修复区)。
    
    参考MI-GAN项目的掩码处理方法，更精细地创建文字区域掩码
    黑色区域（0）表示需要修复的区域
    白色区域（255）表示保留的区域
    
    Args:
        image_size: 图像尺寸 (height, width, channels) 或 (height, width)
        bubble_coords: 气泡 AABB 坐标列表 [(x1, y1, x2, y2), ...]
        bubble_polygons: 可选，气泡多边形坐标列表 [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ...]
                        如果提供，将使用多边形而不是矩形来创建掩码
    """
    logger.debug(f"创建气泡掩码: {len(bubble_coords)} 个")
    if not bubble_coords:
        return np.ones(image_size[:2], dtype=np.uint8) * 255

    # 创建全白掩码（全部保留）
    mask = np.ones(image_size[:2], dtype=np.uint8) * 255
    
    for i, (x1, y1, x2, y2) in enumerate(bubble_coords):
        # 计算气泡大小
        width = x2 - x1
        height = y2 - y1
        
        if width <= 0 or height <= 0: continue
        
        # 使用比例缩放的填充，更灵活地适应不同大小的气泡
        padding_ratio = 0.02  # 2%的填充比例
        min_padding = 1
        
        padding_w = max(min_padding, int(width * padding_ratio))
        padding_h = max(min_padding, int(height * padding_ratio))
        
        # 创建精确的文字区域掩码
        # 如果有多边形数据，使用多边形填充；否则使用矩形
        if bubble_polygons and i < len(bubble_polygons):
            polygon = bubble_polygons[i]
            if polygon and len(polygon) >= 3:
                # 转换为 numpy 数组，确保是整数
                pts = np.array(polygon, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 0)
            else:
                # 多边形无效，回退到矩形
                cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
        else:
            # 没有多边形数据，使用矩形
            cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)  # -1表示填充
        
        # 更精确的边缘处理，确保气泡边缘平滑
        # 外围添加一圈渐变区域，改善与背景的融合
        edge_mask = np.ones_like(mask) * 255
        
        # 对于边缘掩码，仍使用 AABB 来创建渐变区域
        cv2.rectangle(edge_mask, 
                     (max(0, x1-padding_w), max(0, y1-padding_h)), 
                     (min(mask.shape[1]-1, x2+padding_w), min(mask.shape[0]-1, y2+padding_h)), 
                     0, padding_w)
        
        # 使用高斯模糊创建边缘渐变效果，使修复效果更自然
        blur_size = max(3, padding_w*2+1)
        if blur_size % 2 == 0:  # 确保大小是奇数
            blur_size += 1
        edge_mask = cv2.GaussianBlur(edge_mask, (blur_size, blur_size), 0)
        
        # 合并主体掩码和边缘掩码，确保中心区域为0
        mask = np.minimum(mask, edge_mask)

    # 检查掩码是否覆盖了图像的大部分
    total_pixels = mask.size
    zeros = np.sum(mask == 0)
    black_ratio = zeros / total_pixels
    
    # 调整阈值为25%，更保守但还是可以允许适度的修复区域
    if black_ratio > 0.25:
        logger.warning(f"掩码黑色区域占比较高 ({black_ratio:.2%})，可能影响修复效果")
        # 如果黑色区域太大，尝试收缩掩码
        if black_ratio > 0.4:  # 如果超过40%，则收缩掩码
            logger.debug("执行掩码收缩")
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)

    try:
        debug_dir = get_debug_dir("inpainting_masks")
        cv2.imwrite(os.path.join(debug_dir, "bubble_mask_core.png"), mask)
    except Exception as save_e:
        logger.warning(f"保存修复掩码调试图像失败: {save_e}")

    return mask

def inpaint_bubbles(image_pil, bubble_coords, method='solid', fill_color=constants.DEFAULT_FILL_COLOR, bubble_polygons=None, precise_mask=None, mask_dilate_size=0, mask_box_expand_ratio=0, lama_model='lama_mpe'):
    """
    根据指定方法修复或填充图像中的气泡区域。

    Args:
        image_pil (PIL.Image.Image): 原始 PIL 图像。
        bubble_coords (list): 气泡坐标列表 [(x1, y1, x2, y2), ...]。
        method (str): 修复方法 ('solid', 'lama')。
        fill_color (str): 'solid' 方法使用的填充颜色。
        bubble_polygons (list): 可选，气泡多边形坐标列表 [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ...]
                               如果提供，将使用多边形而不是矩形来创建掩码和填充
        precise_mask (np.ndarray): 可选，模型生成的精确文字掩膜。
                                   如果提供，将直接使用此掩膜而非根据坐标生成。
                                   仅 CTD/Default 检测器支持生成此掩膜。
        mask_dilate_size (int): 掩膜膨胀大小（像素），用于扩大修复区域。
        mask_box_expand_ratio (int): 标注框区域扩大比例（%），用于扩大标注框的收录范围。
        lama_model (str): LAMA 模型选择 'lama_mpe' (速度优化) 或 'litelama' (通用)

    Returns:
        PIL.Image.Image: 处理后的 PIL 图像。
        PIL.Image.Image or None: 清理后的背景图像（如果修复成功），否则为 None。
    """
    if not bubble_coords:
        logger.debug("无气泡坐标，跳过修复")
        return image_pil.copy(), None # 返回原图副本和无干净背景

    try:
        img_np = np.array(image_pil.convert('RGB'))
        image_size = img_np.shape
    except Exception as e:
         logger.error(f"无法将输入图像转换为 NumPy 数组: {e}", exc_info=True)
         return image_pil.copy(), None

    # 1. 创建掩码 (黑色为修复区)
    if precise_mask is not None:
        # 使用模型生成的精确文字掩膜
        logger.debug("使用精确文字掩膜")
        
        # 确保掩膜尺寸与图像匹配
        if precise_mask.shape[:2] != image_size[:2]:
            precise_mask = cv2.resize(precise_mask, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
        
        # 模型输出的 mask 中，高值（白色）表示文字区域
        # 转换为修复掩膜格式：黑色(0)=需要修复，白色(255)=保留
        if precise_mask.max() <= 1.0:
            # 归一化的浮点掩膜，转换为 0-255
            precise_mask = (precise_mask * 255).astype(np.uint8)
        
        # 反转掩膜：文字区域（高值）变为需要修复的区域（低值）
        text_mask = 255 - precise_mask
        
        # 应用阈值，确保是二值掩膜
        _, text_mask = cv2.threshold(text_mask, 127, 255, cv2.THRESH_BINARY)
        
        # 只保留标注框内的文字掩膜（只修复框出来的区域）
        # 创建一个标注框区域的掩膜
        box_region_mask = np.ones_like(text_mask) * 255  # 白色表示保留
        img_h, img_w = text_mask.shape[:2]
        expand_ratio = mask_box_expand_ratio / 100.0  # 转换为小数
        
        for (x1, y1, x2, y2) in bubble_coords:
            # 计算扩大后的区域
            box_w = x2 - x1
            box_h = y2 - y1
            expand_w = int(box_w * expand_ratio / 2)
            expand_h = int(box_h * expand_ratio / 2)
            
            # 应用扩大并限制在图像范围内
            ex1 = max(0, int(x1) - expand_w)
            ey1 = max(0, int(y1) - expand_h)
            ex2 = min(img_w, int(x2) + expand_w)
            ey2 = min(img_h, int(y2) + expand_h)
            
            box_region_mask[ey1:ey2, ex1:ex2] = 0  # 标注框内区域设为黑色（需要处理）
        
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
        
        # 保存调试图像
        try:
            debug_dir = get_debug_dir("inpainting_masks")
            cv2.imwrite(os.path.join(debug_dir, "precise_mask_used.png"), bubble_mask_np)
        except Exception as save_e:
            logger.debug(f"保存精确掩膜调试图像失败: {save_e}")
    else:
        # 使用坐标/多边形生成掩膜
        bubble_mask_np = create_bubble_mask(image_size, bubble_coords, bubble_polygons)
    
    bubble_mask_pil = Image.fromarray(bubble_mask_np)

    result_img = image_pil.copy()
    clean_background = None
    inpainting_successful = False

    # 2. 根据方法进行处理
    if method == 'lama' and is_lama_available() and clean_image_with_lama:
        logger.debug(f"使用 LAMA 修复 (模型: {lama_model})")
        try:
            # 从 constants 读取是否禁用缩放的配置
            disable_resize = getattr(constants, 'LAMA_DISABLE_RESIZE', False)
            # 直接传递掩码，不需要在这里反转，因为clean_image_with_lama已经处理掩码反转
            repaired_img = clean_image_with_lama(image_pil, bubble_mask_pil, lama_model=lama_model, disable_resize=disable_resize)
            if repaired_img:
                result_img = repaired_img
                clean_background = result_img.copy()
                setattr(result_img, '_lama_inpainted', True)
                inpainting_successful = True
                logger.debug("LAMA 修复成功")
            else:
                logger.error("LAMA 修复执行失败，未返回结果。将回退。")
        except Exception as e:
             logger.error(f"LAMA 修复过程中出错: {e}", exc_info=True)
             logger.debug("LAMA 出错，回退到纯色填充")

    # 修复 Bug #3: 只在以下情况执行纯色填充：
    # 1. 用户明确选择了 solid 方法
    # 2. 用户选择了 lama 方法但 LAMA 修复失败（回退到纯色填充）
    should_do_solid_fill = (method == 'solid') or (method == 'lama' and not inpainting_successful)
    
    if should_do_solid_fill:
        use_precise = precise_mask is not None
        logger.debug(f"纯色填充: {fill_color}")
        # 确保在 result_img 上绘制（可能是原图副本，也可能是修复失败后的图）
        try:
            if use_precise:
                # 使用精确掩膜进行纯色填充
                # bubble_mask_np 中黑色(0)表示需要填充的区域
                result_np = np.array(result_img.convert('RGB'))
                
                # 解析填充颜色
                if isinstance(fill_color, str):
                    if fill_color.startswith('#'):
                        # 十六进制颜色
                        r = int(fill_color[1:3], 16)
                        g = int(fill_color[3:5], 16)
                        b = int(fill_color[5:7], 16)
                    else:
                        # 默认白色
                        r, g, b = 255, 255, 255
                else:
                    r, g, b = fill_color if len(fill_color) >= 3 else (255, 255, 255)
                
                # 在掩膜为黑色（需要修复）的地方填充颜色
                fill_mask = bubble_mask_np < 128  # 黑色区域为 True
                result_np[fill_mask] = [r, g, b]
                
                result_img = Image.fromarray(result_np)
                logger.debug("精确掩膜填充完成")
            else:
                # 使用坐标/多边形填充（原有逻辑）
                draw = ImageDraw.Draw(result_img)
                for i, (x1, y1, x2, y2) in enumerate(bubble_coords):
                    if x1 < x2 and y1 < y2: # 检查坐标有效性
                        # 如果有多边形数据，使用多边形填充
                        if bubble_polygons and i < len(bubble_polygons):
                            polygon = bubble_polygons[i]
                            if polygon and len(polygon) >= 3:
                                # 将多边形坐标转换为 PIL 需要的格式 [(x1, y1), (x2, y2), ...]
                                pts = [(int(p[0]), int(p[1])) for p in polygon]
                                draw.polygon(pts, fill=fill_color)
                            else:
                                # 多边形无效，回退到矩形
                                draw.rectangle(((x1, y1), (x2, y2)), fill=fill_color)
                        else:
                            # 没有多边形数据，使用矩形
                            draw.rectangle(((x1, y1), (x2, y2)), fill=fill_color)
                    else:
                        logger.warning(f"跳过无效坐标进行纯色填充: ({x1},{y1},{x2},{y2})")
            # 对于纯色填充，也生成一个"干净"背景的副本
            clean_background = result_img.copy()
            logger.debug("纯色填充完成")
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