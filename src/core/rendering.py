import logging
import math
import os
from PIL import Image, ImageDraw, ImageFont
import cv2 # 导入 cv2 备用

# 导入常量和路径助手
from src.shared import constants
from src.shared.path_helpers import resource_path, get_debug_dir # 导入 get_debug_dir

logger = logging.getLogger("CoreRendering")
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- 字体加载缓存 ---
_font_cache = {}

# --- 特殊字符的字体路径 ---
NOTOSANS_FONT_PATH = os.path.join('src', 'app', 'static', 'fonts', 'NotoSans-Medium.ttf')

# --- 需要使用特殊字体渲染的字符 ---
SPECIAL_CHARS = {'‼', '⁉'}

# --- 竖排标点符号映射表 ---
VERTICAL_PUNCTUATION_MAP = {
    # 中文标点
     '（': '︵', '）': '︶', 
    '【': '︻', '】': '︼', '「': '﹁', '」': '﹂', 
    '『': '﹃', '』': '﹄', '〈': '︿', '〉': '﹀',
    '"': '﹃', '"': '﹄', ''': '﹁', ''': '﹂',
    '《': '︽', '》': '︾', '［': '︹', '］': '︺',
    '｛': '︷', '｝': '︸', '〔': '︹', '〕': '︺',
    '—': '︱', '…': '︙', '～': '︴',
    
    # 英文标点
    '(': '︵', ')': '︶',
    '[': '︹', ']': '︺', '{': '︷', '}': '︸',
    '<': '︿', '>': '﹀', '-': '︱', '~': '︴'
}

# 特殊组合标点映射
SPECIAL_PUNCTUATION_PATTERNS = [
    ('...', '︙'),     # 连续三个点映射成竖直省略号
    ('…', '︙'),       # Unicode省略号映射成竖直省略号
    ('!!', '‼'),       # 连续两个感叹号映射成双感叹号
    ('!!!', '‼'),      # 连续三个感叹号映射成双感叹号
    ('！！', '‼'),     # 中文连续两个感叹号
    ('！！！', '‼'),   # 中文连续三个感叹号
    ('!?', '⁉'),       # 感叹号加问号映射成感叹问号组合
    ('?!', '⁉'),       # 问号加感叹号映射成感叹问号组合
    ('！？', '⁉'),     # 中文感叹号加问号
    ('？！', '⁉'),     # 中文问号加感叹号
]

def map_to_vertical_punctuation(text):
    """
    将文本中的标点符号映射为竖排标点符号
    
    Args:
        text (str): 原始文本
        
    Returns:
        str: 转换后的文本，标点符号已替换为竖排版本
    """
    # 首先处理特殊组合标点
    for pattern, replacement in SPECIAL_PUNCTUATION_PATTERNS:
        text = text.replace(pattern, replacement)
    
    # 然后处理单个标点
    result = ""
    i = 0
    while i < len(text):
        char = text[i]
        if char in VERTICAL_PUNCTUATION_MAP:
            result += VERTICAL_PUNCTUATION_MAP[char]
        else:
            result += char
        i += 1
    
    return result

def get_font(font_family_relative_path=constants.DEFAULT_FONT_RELATIVE_PATH, font_size=constants.DEFAULT_FONT_SIZE):
    """
    加载字体文件，带缓存。

    Args:
        font_family_relative_path (str): 字体的相对路径 (相对于项目根目录)。
        font_size (int): 字体大小。

    Returns:
        ImageFont.FreeTypeFont or ImageFont.ImageFont: 加载的字体对象，失败则返回默认字体。
    """
    # 确保 font_size 是整数
    try:
        font_size = int(font_size)
        if font_size <= 0:
             font_size = constants.DEFAULT_FONT_SIZE # 防止无效字号
    except (ValueError, TypeError):
         font_size = constants.DEFAULT_FONT_SIZE

    cache_key = (font_family_relative_path, font_size)
    if cache_key in _font_cache:
        return _font_cache[cache_key]

    font = None
    try:
        font_path_abs = resource_path(font_family_relative_path)
        if os.path.exists(font_path_abs):
            font = ImageFont.truetype(font_path_abs, font_size, encoding="utf-8")
            logger.info(f"成功加载字体: {font_path_abs} (大小: {font_size})")
        else:
            logger.warning(f"字体文件未找到: {font_path_abs} (相对路径: {font_family_relative_path})")
            raise FileNotFoundError()

    except Exception as e:
        logger.error(f"加载字体 {font_family_relative_path} (大小: {font_size}) 失败: {e}，尝试默认字体。")
        try:
            default_font_path_abs = resource_path(constants.DEFAULT_FONT_RELATIVE_PATH)
            if os.path.exists(default_font_path_abs):
                 font = ImageFont.truetype(default_font_path_abs, font_size, encoding="utf-8")
                 logger.info(f"成功加载默认字体: {default_font_path_abs} (大小: {font_size})")
            else:
                 logger.error(f"默认字体文件也未找到: {default_font_path_abs}")
                 font = ImageFont.load_default()
                 logger.warning("使用 Pillow 默认字体。")
        except Exception as e_default:
            logger.error(f"加载默认字体时出错: {e_default}", exc_info=True)
            font = ImageFont.load_default()
            logger.warning("使用 Pillow 默认字体。")

    _font_cache[cache_key] = font
    return font

def calculate_auto_font_size(text, bubble_width, bubble_height, text_direction='vertical',
                             font_family_relative_path=constants.DEFAULT_FONT_RELATIVE_PATH,
                             min_size=12, max_size=60, padding_ratio=1.0):
    """
    使用二分法计算最佳字体大小。
    """
    if not text or not text.strip() or bubble_width <= 0 or bubble_height <= 0:
        return constants.DEFAULT_FONT_SIZE

    W = bubble_width * padding_ratio
    H = bubble_height * padding_ratio
    N = len(text)
    c_w = 1.0
    l_h = 1.2

    if text_direction == 'vertical':
        W, H = H, W

    low = min_size
    high = max_size
    best_size = min_size

    while low <= high:
        mid = (low + high) // 2
        if mid == 0: break

        try:
            font = get_font(font_family_relative_path, mid)
            if font is None:
                high = mid - 1
                continue

            avg_char_width = mid * c_w
            avg_char_height = mid

            if text_direction == 'horizontal':
                chars_per_line = max(1, int(W / avg_char_width)) if avg_char_width > 0 else N # 避免除零
                lines_needed = math.ceil(N / chars_per_line) if chars_per_line > 0 else N
                total_height_needed = lines_needed * mid * l_h
                fits = total_height_needed <= H
            else: # vertical
                chars_per_column = max(1, int(H / avg_char_height)) if avg_char_height > 0 else N
                columns_needed = math.ceil(N / chars_per_column) if chars_per_column > 0 else N
                total_width_needed = columns_needed * mid * l_h
                fits = total_width_needed <= W

            if fits:
                best_size = mid
                low = mid + 1
            else:
                high = mid - 1

        except Exception as e:
            logger.error(f"计算字号 {mid} 时出错: {e}", exc_info=True)
            high = mid - 1

    result = max(min_size, best_size)
    logger.info(f"自动计算的最佳字体大小: {result}px (范围: {min_size}-{max_size})")
    return result

# --- 占位符，后续步骤会添加 ---
def draw_multiline_text_vertical(draw, text, font, x, y, max_height, fill=constants.DEFAULT_TEXT_COLOR, rotation_angle=constants.DEFAULT_ROTATION_ANGLE, bubble_width=None):
    if not text:
        return
    
    # 将标点符号转换为竖排样式
    text = map_to_vertical_punctuation(text)

    lines = []
    current_line = ""
    current_column_height = 0
    line_height = font.size + 5

    for char in text:
        bbox = font.getbbox(char)
        char_height = bbox[3] - bbox[1]

        if current_column_height + line_height <= max_height:
            current_line += char
            current_column_height += line_height
        else:
            lines.append(current_line)
            current_line = char
            current_column_height = line_height

    lines.append(current_line)

    column_width = font.size + 5
    
    # 计算文本段落的总宽度
    total_text_width = len(lines) * column_width
    
    # 在竖排文本中，x 是气泡的右边界
    if bubble_width is not None:
        # 如果传入了气泡宽度，使用气泡宽度计算居中位置
        bubble_center_x = x - bubble_width / 2
        current_x = bubble_center_x + total_text_width / 2
    else:
        # 如果没有传入气泡宽度，默认靠右对齐
        current_x = x
    
    # 计算文本垂直方向的总高度
    max_line_chars = 0
    for line in lines:
        max_line_chars = max(max_line_chars, len(line))
    total_text_height = max_line_chars * line_height
    
    # 计算垂直方向的居中偏移
    if total_text_height < max_height:
        vertical_offset = (max_height - total_text_height) / 2
        current_y = y + vertical_offset
    else:
        current_y = y  # 如果文本高度超过气泡高度，则不进行垂直居中
    
    # 如果需要旋转，先获取原始图像
    original_image = None
    if rotation_angle != 0:
        if hasattr(draw, '_image'):
            original_image = draw._image
        
        # 计算所有列的中心点，用于旋转
        if bubble_width is not None:
            center_x = x - bubble_width / 2  # 使用气泡中心点作为旋转中心
        else:
            center_x = x - (len(lines) * column_width) / 2  # 默认计算方式
        center_y = y + max_height / 2

    # 预加载NotoSans字体，用于特殊字符
    special_font = None
    font_size = font.size  # 获取当前字体大小

    for line in lines:
        line_start_y = current_y  # 使用计算出的居中起始位置
        for char in line:
            # 计算字符尺寸
            current_font = font
            # 检查是否为需要特殊字体的字符
            if char in SPECIAL_CHARS:
                if special_font is None:
                    try:
                        # 第一次遇到特殊字符时加载特殊字体
                        special_font = get_font(NOTOSANS_FONT_PATH, font_size)
                        logger.info(f"为特殊字符加载NotoSans字体，字号为 {font_size}")
                    except Exception as e:
                        logger.error(f"加载NotoSans字体失败: {e}，回退到普通字体")
                        special_font = font  # 如果加载失败，使用普通字体
                
                if special_font is not None:
                    current_font = special_font
            
            # 使用当前选定的字体计算字符尺寸
            bbox = current_font.getbbox(char)
            char_width = bbox[2] - bbox[0]
            char_height = bbox[3] - bbox[1]
            
            # 计算正确的位置（右上角对齐）
            text_x = current_x - char_width
            text_y = line_start_y
            
            if rotation_angle != 0 and original_image is not None:
                try:
                    # 直接在原始图像上绘制旋转后的文本
                    rotated_draw = ImageDraw.Draw(original_image)
                    
                    # 创建临时图像以执行旋转
                    import numpy as np
                    temp_img = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
                    temp_draw = ImageDraw.Draw(temp_img)
                    
                    # 在临时图像上绘制文本，使用适当的字体
                    temp_draw.text((text_x, text_y), char, font=current_font, fill=fill)
                    
                    # 旋转临时图像
                    rotated_temp = temp_img.rotate(
                        rotation_angle, 
                        resample=Image.Resampling.BICUBIC,
                        center=(center_x, center_y),
                        expand=False
                    )
                    
                    # 将旋转后的图像合成到原始图像
                    original_image.paste(rotated_temp, (0, 0), rotated_temp)
                except Exception as e:
                    logger.error(f"旋转渲染失败，回退到直接渲染: {e}")
                    draw.text((text_x, text_y), char, font=current_font, fill=fill)
            else:
                # 不旋转时，直接在原始 draw 对象上绘制
                draw.text((text_x, text_y), char, font=current_font, fill=fill)
                
            line_start_y += line_height
        current_x -= column_width

def draw_multiline_text_horizontal(draw, text, font, x, y, max_width, fill=constants.DEFAULT_TEXT_COLOR, rotation_angle=constants.DEFAULT_ROTATION_ANGLE):
    if not text:
        return

    lines = []
    current_line = ""
    current_line_width = 0

    for char in text:
        bbox = font.getbbox(char)
        char_width = bbox[2] - bbox[0]
        space_width = font.getbbox(' ')[2] - font.getbbox(' ')[0]

        if current_line_width + char_width <= max_width:
            current_line += char
            current_line_width += char_width
        else:
            lines.append(current_line)
            current_line = char
            current_line_width = char_width

    lines.append(current_line)

    current_y = y
    line_height = font.size + 5
    
    # 预加载NotoSans字体，用于特殊字符
    special_font = None
    font_size = font.size  # 获取当前字体大小
    
    # 如果需要旋转，先获取原始图像
    original_image = None
    if rotation_angle != 0:
        if hasattr(draw, '_image'):
            original_image = draw._image
            
        # 计算文本块的中心点，用于旋转
        center_x = x + max_width / 2
        center_y = y + (len(lines) * line_height) / 2

    for line in lines:
        current_x = x
        for char in line:
            # 检查是否为需要特殊字体的字符
            current_font = font
            if char in SPECIAL_CHARS:
                if special_font is None:
                    try:
                        # 第一次遇到特殊字符时加载特殊字体
                        special_font = get_font(NOTOSANS_FONT_PATH, font_size)
                    except Exception as e:
                        logger.error(f"加载NotoSans字体失败: {e}，回退到普通字体")
                        special_font = font  # 如果加载失败，使用普通字体
                
                if special_font is not None:
                    current_font = special_font

            # 使用当前选定的字体计算字符尺寸
            bbox = current_font.getbbox(char)
            char_width = bbox[2] - bbox[0]
            char_height = bbox[3] - bbox[1]
            
            if rotation_angle != 0 and original_image is not None:
                try:
                    # 直接在原始图像上绘制旋转后的文本
                    rotated_draw = ImageDraw.Draw(original_image)
                    
                    # 创建临时图像以执行旋转
                    import numpy as np
                    temp_img = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
                    temp_draw = ImageDraw.Draw(temp_img)
                    
                    # 在临时图像上绘制文本，使用适当的字体
                    temp_draw.text((current_x, current_y), char, font=current_font, fill=fill)
                    
                    # 旋转临时图像
                    rotated_temp = temp_img.rotate(
                        rotation_angle, 
                        resample=Image.Resampling.BICUBIC,
                        center=(center_x, center_y),
                        expand=False
                    )
                    
                    # 将旋转后的图像合成到原始图像
                    original_image.paste(rotated_temp, (0, 0), rotated_temp)
                except Exception as e:
                    logger.error(f"旋转渲染失败，回退到直接渲染: {e}")
                    draw.text((current_x, current_y), char, font=current_font, fill=fill)
            else:
                draw.text((current_x, current_y), char, font=current_font, fill=fill)
            
            current_x += char_width
        current_y += line_height

def render_all_bubbles(draw_image, all_texts, bubble_coords, bubble_styles):
    """
    在图像上渲染所有气泡的文本，使用各自的样式。

    Args:
        draw_image (PIL.Image.Image): 要绘制文本的 PIL 图像对象 (会被直接修改)。
        all_texts (list): 所有气泡的文本列表。
        bubble_coords (list): 气泡坐标列表 [(x1, y1, x2, y2), ...]。
        bubble_styles (dict): 包含每个气泡样式的字典，键为气泡索引(字符串),
                              值为样式字典 {'fontSize':, 'autoFontSize':, 'fontFamily':,
                              'textDirection':, 'position_offset':, 'textColor':, 'rotationAngle':}。
    """
    if not all_texts or not bubble_coords or len(all_texts) != len(bubble_coords):
        logger.warning(f"文本({len(all_texts) if all_texts else 0})、坐标({len(bubble_coords) if bubble_coords else 0})数量不匹配，无法渲染。")
        return

    draw = ImageDraw.Draw(draw_image)
    logger.info(f"开始渲染 {len(bubble_coords)} 个气泡的文本...")

    for i, (x1, y1, x2, y2) in enumerate(bubble_coords):
        # 确保索引有效
        if i >= len(all_texts):
            logger.warning(f"索引 {i} 超出文本列表范围，跳过。")
            continue

        style = bubble_styles.get(str(i), {}) # 获取当前气泡样式
        text = all_texts[i] if all_texts[i] is not None else "" # 处理 None 值

        # --- 获取样式参数 ---
        font_size_setting = style.get('fontSize', constants.DEFAULT_FONT_SIZE)
        auto_font_size = style.get('autoFontSize', False)
        # fontFamily 应该是相对路径，如 'src/app/static/fonts/...'
        font_family_rel = style.get('fontFamily', constants.DEFAULT_FONT_RELATIVE_PATH)
        text_direction = style.get('text_direction', constants.DEFAULT_TEXT_DIRECTION)
        position_offset = style.get('position_offset', {'x': 0, 'y': 0})
        text_color = style.get('text_color', constants.DEFAULT_TEXT_COLOR)
        rotation_angle = style.get('rotation_angle', constants.DEFAULT_ROTATION_ANGLE)

        # --- 处理字体大小 ---
        current_font_size = constants.DEFAULT_FONT_SIZE
        if auto_font_size:
            bubble_width = x2 - x1
            bubble_height = y2 - y1
            if 'calculated_font_size' in style and style['calculated_font_size']:
                 current_font_size = style['calculated_font_size']
            else:
                 current_font_size = calculate_auto_font_size(
                     text, bubble_width, bubble_height, text_direction, font_family_rel
                 )
                 style['calculated_font_size'] = current_font_size # 保存计算结果
        elif isinstance(font_size_setting, (int, float)) and font_size_setting > 0:
            current_font_size = int(font_size_setting)
        elif isinstance(font_size_setting, str) and font_size_setting.isdigit(): # 处理字符串形式的数字
             current_font_size = int(font_size_setting)

        # --- 加载字体 ---
        font = get_font(font_family_rel, current_font_size)
        if font is None:
            logger.error(f"气泡 {i}: 无法加载字体 {font_family_rel} (大小: {current_font_size})，跳过渲染。")
            continue

        # --- 计算绘制参数 ---
        offset_x = position_offset.get('x', 0)
        offset_y = position_offset.get('y', 0)
        draw_x = x1 + offset_x
        draw_y = y1 + offset_y
        vertical_draw_x = x2 + offset_x
        max_text_width = max(10, x2 - x1)
        max_text_height = max(10, y2 - y1)

        # --- 调用绘制函数 ---
        try:
            if text_direction == 'vertical':
                bubble_width = max_text_width  # 气泡的宽度，用于居中计算
                draw_multiline_text_vertical(draw, text, font, vertical_draw_x, draw_y, max_text_height, fill=text_color, rotation_angle=rotation_angle, bubble_width=bubble_width)
            elif text_direction == 'horizontal':
                draw_multiline_text_horizontal(draw, text, font, draw_x, draw_y, max_text_width, fill=text_color, rotation_angle=rotation_angle)
            else:
                logger.warning(f"气泡 {i}: 未知的文本方向 '{text_direction}'，跳过渲染。")
        except Exception as render_e:
             logger.error(f"渲染气泡 {i} 时出错: {render_e}", exc_info=True)

    logger.info("所有气泡文本渲染完成。")

def render_single_bubble(
    image, # 可能是带文字的 PIL Image，也可能是干净背景
    bubble_index,
    all_texts, # 包含所有气泡当前文本的列表
    bubble_coords,
    fontSize=constants.DEFAULT_FONT_SIZE, # 目标气泡的新字号 ('auto' 或数字)
    fontFamily=constants.DEFAULT_FONT_RELATIVE_PATH, # 目标气泡的新字体 (相对路径)
    text_direction=constants.DEFAULT_TEXT_DIRECTION, # 目标气泡的新方向
    position_offset={'x': 0, 'y': 0}, # 目标气泡的新偏移
    use_inpainting=False, # 全局设置 - 用于决定无背景时如何修复
    is_single_bubble_style=False, # 这个参数在此重构下不再需要
    text_color=constants.DEFAULT_TEXT_COLOR, # 目标气泡的新颜色
    rotation_angle=constants.DEFAULT_ROTATION_ANGLE, # 目标气泡的新旋转
    use_lama=False, # 全局设置 - 用于决定无背景时如何修复
    fill_color=constants.DEFAULT_FILL_COLOR # 全局设置 - 用于无背景时的纯色填充
    ):
    """
    使用新的文本和样式重新渲染单个气泡（通过更新样式并渲染所有气泡实现）。
    """
    logger.info(f"开始渲染单气泡 {bubble_index}，字体: {fontFamily}, 大小: {fontSize}, 方向: {text_direction}")

    if bubble_index < 0 or bubble_index >= len(bubble_coords):
        logger.error(f"无效的气泡索引 {bubble_index}")
        return image # 返回原始图像

    # --- 获取基础图像 (优先使用干净背景) ---
    img_pil = None
    clean_image_base = None
    if hasattr(image, '_clean_image') and isinstance(getattr(image, '_clean_image'), Image.Image):
        clean_image_base = getattr(image, '_clean_image').copy()
        img_pil = clean_image_base
    elif hasattr(image, '_clean_background') and isinstance(getattr(image, '_clean_background'), Image.Image):
        clean_image_base = getattr(image, '_clean_background').copy()
        img_pil = clean_image_base

    if img_pil is None:
        logger.warning(f"单气泡 {bubble_index} 渲染时未找到干净背景，将执行修复/填充...")
        target_coords = [bubble_coords[bubble_index]]
        
        # 导入修复相关模块
        from src.core.inpainting import inpaint_bubbles
        from src.interfaces.lama_interface import is_lama_available
        # from src.interfaces.migan_interface import is_migan_available
        
        inpainting_method = 'solid'
        if use_lama and is_lama_available(): inpainting_method = 'lama'
        # elif use_inpainting and is_migan_available(): inpainting_method = 'migan'
        img_pil, generated_clean_bg = inpaint_bubbles(
            image, target_coords, method=inpainting_method, fill_color=fill_color
        )
        if generated_clean_bg: clean_image_base = generated_clean_bg.copy()

    # --- 获取或创建样式字典 ---
    bubble_styles_to_use = {}
    if hasattr(image, '_bubble_styles') and isinstance(getattr(image, '_bubble_styles'), dict):
         bubble_styles_to_use = getattr(image, '_bubble_styles').copy()
         bubble_styles_to_use = {str(k): v for k, v in bubble_styles_to_use.items()}
         logger.debug(f"单气泡渲染：从图像加载了 {len(bubble_styles_to_use)} 个样式。")
    else:
         logger.warning("单气泡渲染：未找到保存的气泡样式，将创建默认样式。")
         for i in range(len(bubble_coords)):
             bubble_styles_to_use[str(i)] = {
                 'fontSize': constants.DEFAULT_FONT_SIZE, 'autoFontSize': False,
                 'fontFamily': constants.DEFAULT_FONT_RELATIVE_PATH, 'text_direction': constants.DEFAULT_TEXT_DIRECTION,
                 'position_offset': {'x': 0, 'y': 0}, 'text_color': constants.DEFAULT_TEXT_COLOR,
                 'rotation_angle': constants.DEFAULT_ROTATION_ANGLE
             }

    # --- 更新目标气泡的样式 ---
    target_style = bubble_styles_to_use.get(str(bubble_index), {}) # 获取现有或空字典
    is_auto_font_size = isinstance(fontSize, str) and fontSize.lower() == 'auto'
    target_font_rel = fontFamily # 假设 fontFamily 已经是相对路径
    target_style.update({
        'fontSize': fontSize, # 保存 'auto' 或数字
        'autoFontSize': is_auto_font_size,
        'fontFamily': target_font_rel,
        'text_direction': text_direction,
        'position_offset': position_offset,
        'text_color': text_color,
        'rotation_angle': rotation_angle
    })
    # 如果是自动字号，需要计算并保存 (确保 calculate_auto_font_size 已导入)
    if is_auto_font_size:
         bubble_width = bubble_coords[bubble_index][2] - bubble_coords[bubble_index][0]
         bubble_height = bubble_coords[bubble_index][3] - bubble_coords[bubble_index][1]
         text_to_render = all_texts[bubble_index] if bubble_index < len(all_texts) else ""
         calculated_size = calculate_auto_font_size(
             text_to_render, bubble_width, bubble_height, text_direction, target_font_rel
         )
         target_style['calculated_font_size'] = calculated_size
         target_style['fontSize'] = calculated_size # 更新 fontSize 为计算值
         logger.info(f"单气泡 {bubble_index}: 自动计算字号为 {calculated_size}px")

    bubble_styles_to_use[str(bubble_index)] = target_style
    logger.debug(f"单气泡渲染：更新气泡 {bubble_index} 的样式为: {target_style}")

    # --- 更新目标气泡的文本 ---
    # 确保 all_texts 长度足够
    if len(all_texts) <= bubble_index:
         all_texts.extend([""] * (bubble_index - len(all_texts) + 1))
    # 更新文本 (假设 all_texts 是从前端获取的最新列表)
    # logger.debug(f"单气泡渲染：使用文本列表: {all_texts}")

    # --- 调用核心渲染函数渲染所有气泡 ---
    render_all_bubbles(
        img_pil,
        all_texts, # 传递包含所有最新文本的列表
        bubble_coords,
        bubble_styles_to_use # 传递更新后的样式字典
    )

    # --- 准备返回值 ---
    img_with_bubbles_pil = img_pil
    # 附加必要的属性
    if hasattr(image, '_lama_inpainted'): setattr(img_with_bubbles_pil, '_lama_inpainted', True)
    if clean_image_base:
         setattr(img_with_bubbles_pil, '_clean_image', clean_image_base)
         setattr(img_with_bubbles_pil, '_clean_background', clean_image_base)
    # 附加更新后的样式
    setattr(img_with_bubbles_pil, '_bubble_styles', bubble_styles_to_use)

    return img_with_bubbles_pil

def re_render_text_in_bubbles(
    image, # 可能是带文字的 PIL Image，也可能是干净背景
    all_texts,
    bubble_coords,
    fontSize=constants.DEFAULT_FONT_SIZE,
    fontFamily=constants.DEFAULT_FONT_RELATIVE_PATH,
    text_direction=constants.DEFAULT_TEXT_DIRECTION,
    use_inpainting=False,
    blend_edges=True,
    inpainting_strength=constants.DEFAULT_INPAINTING_STRENGTH,
    use_lama=False,
    fill_color=constants.DEFAULT_FILL_COLOR,
    text_color=constants.DEFAULT_TEXT_COLOR,
    rotation_angle=constants.DEFAULT_ROTATION_ANGLE
    ):
    """
    使用新的文本和样式重新渲染气泡中的文字。
    """
    logger.info(f"开始重新渲染，字体: {fontFamily}, 大小: {fontSize}, 方向: {text_direction}")

    if not all_texts or not bubble_coords:
        logger.warning("缺少文本或坐标，无法重新渲染。")
        return image # 返回原始图像

    # --- 获取基础图像 (优先使用干净背景) ---
    img_pil = None
    clean_image_base = None
    if hasattr(image, '_clean_image') and isinstance(getattr(image, '_clean_image'), Image.Image):
        clean_image_base = getattr(image, '_clean_image').copy()
        img_pil = clean_image_base
        logger.info("重渲染：使用 _clean_image 作为基础。")
    elif hasattr(image, '_clean_background') and isinstance(getattr(image, '_clean_background'), Image.Image):
        clean_image_base = getattr(image, '_clean_background').copy()
        img_pil = clean_image_base
        logger.info("重渲染：使用 _clean_background 作为基础。")

    # 如果没有干净背景，则需要重新执行修复/填充
    if img_pil is None:
        logger.warning("重渲染时未找到干净背景，将重新执行修复/填充...")
        
        # 导入修复相关模块
        from src.core.inpainting import inpaint_bubbles
        from src.interfaces.lama_interface import is_lama_available
        # from src.interfaces.migan_interface import is_migan_available
        
        inpainting_method = 'solid'
        if use_lama and is_lama_available(): inpainting_method = 'lama'
        # elif use_inpainting and is_migan_available(): inpainting_method = 'migan'

        logger.info(f"重渲染时选择修复/填充方法: {inpainting_method}")
        img_pil, generated_clean_bg = inpaint_bubbles(
            image, bubble_coords, method=inpainting_method, fill_color=fill_color,
            migan_strength=inpainting_strength, migan_blend_edges=blend_edges
        )
        if generated_clean_bg: clean_image_base = generated_clean_bg.copy()

    # --- 准备样式字典 ---
    bubble_styles_to_use = {}
    
    # 检查图像是否已经有预定义的气泡样式字典
    if hasattr(image, '_bubble_styles') and isinstance(getattr(image, '_bubble_styles'), dict):
        # 优先使用预定义样式
        bubble_styles_to_use = getattr(image, '_bubble_styles')
        logger.info(f"使用图像预定义的气泡样式，共 {len(bubble_styles_to_use)} 个")
    else:
        # 没有预定义样式，使用全局设置创建新样式
        logger.info("没有找到预定义气泡样式，使用全局设置创建样式")
        
        # 获取全局设置参数
        is_auto_font_size_global = isinstance(fontSize, str) and fontSize.lower() == 'auto'
        font_family_rel = fontFamily
        
        logger.info(f"使用传入的全局颜色设置: {text_color}, 旋转角度: {rotation_angle}")
        
        # 为所有气泡创建新的样式字典，使用全局设置
        for i in range(len(bubble_coords)):
            bubble_styles_to_use[str(i)] = {
                'fontSize': fontSize,
                'autoFontSize': is_auto_font_size_global,
                'fontFamily': font_family_rel,
                'text_direction': text_direction,  # 全局文字方向
                'position_offset': {'x': 0, 'y': 0},  # 保持默认位置
                'text_color': text_color,  # 使用从请求获取的颜色
                'rotation_angle': rotation_angle  # 使用从请求获取的旋转角度
            }

    # --- 调用核心渲染函数 ---
    render_all_bubbles(
        img_pil, # 在获取的基础图像上绘制
        all_texts,
        bubble_coords,
        bubble_styles_to_use
    )

    # --- 准备返回值 ---
    img_with_bubbles_pil = img_pil
    # 附加必要的属性
    if hasattr(image, '_lama_inpainted'): setattr(img_with_bubbles_pil, '_lama_inpainted', True)
    if clean_image_base:
         setattr(img_with_bubbles_pil, '_clean_image', clean_image_base)
         setattr(img_with_bubbles_pil, '_clean_background', clean_image_base)
    setattr(img_with_bubbles_pil, '_bubble_styles', bubble_styles_to_use) # 附加更新后的样式

    return img_with_bubbles_pil

# --- 测试代码 ---
if __name__ == '__main__':
    print("--- 测试渲染核心逻辑 (字体加载和自动字号) ---")

    # 测试字体加载
    print("\n测试字体加载:")
    font_default = get_font()
    print(f"默认字体: {type(font_default)}")
    font_custom = get_font(constants.DEFAULT_FONT_RELATIVE_PATH, 30) # 使用常量
    print(f"宋体 30px: {type(font_custom)}")
    font_cached = get_font(constants.DEFAULT_FONT_RELATIVE_PATH, 30)
    print(f"宋体 30px (缓存): {type(font_cached)}")
    font_fail = get_font("non_existent.ttf", 20)
    print(f"无效字体: {type(font_fail)}")

    # 测试自动字号
    print("\n测试自动字号:")
    text_short = "短文本"
    text_long_v = "这是一段非常非常非常非常非常非常非常非常非常非常非常非常长的竖排测试文本内容"
    text_long_h = "This is a very very very very very very very very very very very very long horizontal test text content"
    bubble_w, bubble_h = 100, 200

    size_short = calculate_auto_font_size(text_short, bubble_w, bubble_h, 'vertical')
    print(f"短文本竖排 ({bubble_w}x{bubble_h}): {size_short}px")

    size_long_v = calculate_auto_font_size(text_long_v, bubble_w, bubble_h, 'vertical')
    print(f"长文本竖排 ({bubble_w}x{bubble_h}): {size_long_v}px")

    size_long_h = calculate_auto_font_size(text_long_h, bubble_w, bubble_h, 'horizontal')
    print(f"长文本横排 ({bubble_w}x{bubble_h}): {size_long_h}px")

    size_long_h_wide = calculate_auto_font_size(text_long_h, 300, 100, 'horizontal')
    print(f"长文本横排宽气泡 (300x100): {size_long_h_wide}px")