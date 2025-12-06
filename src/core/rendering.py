import logging
import math
import os
from typing import List, Optional, TYPE_CHECKING
from PIL import Image, ImageDraw, ImageFont
import cv2 # 导入 cv2 备用

# 导入常量和路径助手
from src.shared import constants
from src.shared.path_helpers import resource_path, get_debug_dir, get_font_path # 导入路径助手函数

# 类型提示（避免循环导入）
if TYPE_CHECKING:
    from src.core.config_models import BubbleState

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
        # 使用 get_font_path 统一处理字体路径，支持多种路径格式（包括自定义字体）
        font_path_abs = get_font_path(font_family_relative_path)
        if os.path.exists(font_path_abs):
            font = ImageFont.truetype(font_path_abs, font_size, encoding="utf-8")
            logger.info(f"成功加载字体: {font_path_abs} (大小: {font_size})")
        else:
            logger.warning(f"字体文件未找到: {font_path_abs} (相对路径: {font_family_relative_path})")
            raise FileNotFoundError()

    except Exception as e:
        logger.error(f"加载字体 {font_family_relative_path} (大小: {font_size}) 失败: {e}，尝试默认字体。")
        try:
            # 默认字体也使用 get_font_path 处理
            default_font_path_abs = get_font_path(constants.DEFAULT_FONT_RELATIVE_PATH)
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
    l_h = 1.05

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

# --- 竖排文本绘制函数（不含旋转，旋转在 render_all_bubbles 中统一处理） ---
def draw_multiline_text_vertical(draw, text, font, x, y, max_height,
                                 fill=constants.DEFAULT_TEXT_COLOR,
                                 stroke_enabled=constants.DEFAULT_STROKE_ENABLED,
                                 stroke_color=constants.DEFAULT_STROKE_COLOR,
                                 stroke_width=constants.DEFAULT_STROKE_WIDTH,
                                 bubble_width=None):
    """
    在指定位置绘制竖排多行文本（不含旋转）。
    旋转逻辑已移至 render_all_bubbles 函数中统一处理，使用外接圆方案优化性能。
    """
    if not text:
        return
    
    # 将标点符号转换为竖排样式
    text = map_to_vertical_punctuation(text)

    lines = []
    current_line = ""
    current_column_height = 0
    line_height_approx = font.size + 1  # 字间距为1像素

    for char in text:
        # 处理换行符：强制换列
        if char == '\n':
            if current_line:
                lines.append(current_line)
                current_line = ""
                current_column_height = 0
            continue
        
        if current_column_height + line_height_approx <= max_height:
            current_line += char
            current_column_height += line_height_approx
        else:
            lines.append(current_line)
            current_line = char
            current_column_height = line_height_approx
    lines.append(current_line)

    # 列宽基于字体大小估算
    column_width_approx = font.size + 3

    # 计算文本段落的总宽度
    total_text_width_for_centering = len(lines) * column_width_approx
    
    # 居中对齐
    if bubble_width is not None:
        bubble_center_x = x - bubble_width / 2
        current_x_base = bubble_center_x + total_text_width_for_centering / 2
    else:
        current_x_base = x

    # 计算垂直方向文本总高度，用于居中
    max_chars_in_line = 0
    if lines:
        max_chars_in_line = max(len(line) for line in lines if line)
    total_text_height_for_centering = max_chars_in_line * line_height_approx

    if total_text_height_for_centering < max_height:
        vertical_offset = (max_height - total_text_height_for_centering) / 2
        start_y_base = y + vertical_offset
    else:
        start_y_base = y

    # 预加载NotoSans字体，用于特殊字符
    special_font = None
    font_size = font.size

    current_x_col = current_x_base
    for line_idx, line in enumerate(lines):
        current_y_char = start_y_base
        for char_idx, char in enumerate(line):
            # 检查是否为需要特殊字体的字符
            current_font = font
            if char in SPECIAL_CHARS:
                if special_font is None:
                    try:
                        special_font = get_font(NOTOSANS_FONT_PATH, font_size)
                    except Exception as e:
                        logger.error(f"加载NotoSans字体失败: {e}，回退到普通字体")
                        special_font = font
                
                if special_font is not None:
                    current_font = special_font
            
            bbox = current_font.getbbox(char)
            char_width = bbox[2] - bbox[0]
            
            text_x_char = current_x_col - char_width
            text_y_char = current_y_char
            
            # 准备绘制参数
            text_draw_params = {
                "font": current_font,
                "fill": fill
            }
            if stroke_enabled:
                text_draw_params["stroke_width"] = int(stroke_width)
                text_draw_params["stroke_fill"] = stroke_color
            
            # 直接绘制（旋转在外层处理）
            draw.text((text_x_char, text_y_char), char, **text_draw_params)
                
            current_y_char += line_height_approx
        current_x_col -= column_width_approx

# --- 横排文本绘制函数（不含旋转，旋转在 render_all_bubbles 中统一处理） ---
def draw_multiline_text_horizontal(draw, text, font, x, y, max_width,
                                  fill=constants.DEFAULT_TEXT_COLOR,
                                  stroke_enabled=constants.DEFAULT_STROKE_ENABLED,
                                  stroke_color=constants.DEFAULT_STROKE_COLOR,
                                  stroke_width=constants.DEFAULT_STROKE_WIDTH,
                                  bubble_width=None,
                                  bubble_height=None):
    """
    在指定位置绘制横排多行文本（不含旋转）。
    旋转逻辑已移至 render_all_bubbles 函数中统一处理，使用外接圆方案优化性能。
    
    优化：一次遍历同时完成分行和记录字符宽度，避免重复调用 getbbox()。
    
    Args:
        bubble_width: 气泡宽度，用于水平居中
        bubble_height: 气泡高度，用于垂直居中
    """
    if not text:
        return

    # 一次遍历：分行 + 记录每个字符的宽度
    lines = []
    line_char_widths = []  # 每行的字符宽度列表
    current_line = ""
    current_line_widths = []
    current_line_width = 0

    for char in text:
        # 处理换行符：强制换行
        if char == '\n':
            if current_line:
                lines.append(current_line)
                line_char_widths.append(current_line_widths)
            current_line = ""
            current_line_widths = []
            current_line_width = 0
            continue
        
        bbox = font.getbbox(char)
        char_width = bbox[2] - bbox[0]

        if current_line_width + char_width <= max_width:
            current_line += char
            current_line_widths.append(char_width)
            current_line_width += char_width
        else:
            if current_line:
                lines.append(current_line)
                line_char_widths.append(current_line_widths)
            current_line = char
            current_line_widths = [char_width]
            current_line_width = char_width

    # 添加最后一行
    if current_line:
        lines.append(current_line)
        line_char_widths.append(current_line_widths)

    if not lines:
        return

    line_height = font.size + 5
    
    # 计算每行的总宽度（直接使用已记录的值，不再遍历）
    line_widths = [sum(widths) for widths in line_char_widths]
    
    # 计算垂直居中偏移
    total_text_height = len(lines) * line_height
    if bubble_height is not None and total_text_height < bubble_height:
        vertical_offset = (bubble_height - total_text_height) / 2
        current_y = y + vertical_offset
    else:
        current_y = y
    
    # 预加载NotoSans字体，用于特殊字符
    special_font = None
    font_size = font.size

    for line_idx, line in enumerate(lines):
        # 计算水平居中偏移
        if bubble_width is not None:
            horizontal_offset = (bubble_width - line_widths[line_idx]) / 2
            current_x = x + horizontal_offset
        else:
            current_x = x
        
        char_widths = line_char_widths[line_idx]
        for char_idx, char in enumerate(line):
            # 检查是否为需要特殊字体的字符
            current_font = font
            char_width = char_widths[char_idx]  # 使用缓存的宽度
            
            if char in SPECIAL_CHARS:
                if special_font is None:
                    try:
                        special_font = get_font(NOTOSANS_FONT_PATH, font_size)
                    except Exception as e:
                        logger.error(f"加载NotoSans字体失败: {e}，回退到普通字体")
                        special_font = font
                
                if special_font is not None:
                    current_font = special_font
                    # 特殊字符用特殊字体，需要重新计算宽度
                    bbox = current_font.getbbox(char)
                    char_width = bbox[2] - bbox[0]
            
            text_draw_params = {
                "font": current_font,
                "fill": fill
            }
            if stroke_enabled:
                text_draw_params["stroke_width"] = int(stroke_width)
                text_draw_params["stroke_fill"] = stroke_color
            
            # 直接绘制（旋转在外层处理）
            draw.text((current_x, current_y), char, **text_draw_params)
            
            current_x += char_width
        current_y += line_height

def render_all_bubbles(draw_image, all_texts, bubble_coords, bubble_states):
    """
    在图像上渲染所有气泡的文本，使用各自的样式。
    
    旋转优化：使用外接圆方案，每个气泡只创建一个临时图像进行旋转，
    而不是为每个字符创建临时图像，大幅提升旋转渲染性能。

    Args:
        draw_image (PIL.Image.Image): 要绘制文本的 PIL 图像对象 (会被直接修改)。
        all_texts (list): 所有气泡的文本列表。
        bubble_coords (list): 气泡坐标列表 [(x1, y1, x2, y2), ...]。
        bubble_states (dict): 包含每个气泡样式的字典，键为气泡索引(字符串),
                              值为样式字典 {'fontSize':, 'fontFamily':,
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

        style = bubble_states.get(str(i), {}) # 获取当前气泡样式
        text = all_texts[i] if all_texts[i] is not None else "" # 处理 None 值

        # --- 获取样式参数 ---
        font_size_setting = style.get('fontSize', constants.DEFAULT_FONT_SIZE)
        font_family_rel = style.get('fontFamily', constants.DEFAULT_FONT_RELATIVE_PATH)
        text_direction = style.get('text_direction', constants.DEFAULT_TEXT_DIRECTION)
        position_offset = style.get('position_offset', {'x': 0, 'y': 0})
        text_color = style.get('text_color', constants.DEFAULT_TEXT_COLOR)
        rotation_angle = style.get('rotation_angle', constants.DEFAULT_ROTATION_ANGLE)

        stroke_enabled = style.get('stroke_enabled', constants.DEFAULT_STROKE_ENABLED)
        stroke_color = style.get('stroke_color', constants.DEFAULT_STROKE_COLOR)
        stroke_width = style.get('stroke_width', constants.DEFAULT_STROKE_WIDTH)

        # --- 处理字体大小 ---
        bubble_width = x2 - x1
        bubble_height = y2 - y1
        
        # 直接使用保存的字号
        if isinstance(font_size_setting, (int, float)) and font_size_setting > 0:
            current_font_size = int(font_size_setting)
        elif isinstance(font_size_setting, str) and font_size_setting.isdigit():
            current_font_size = int(font_size_setting)
        else:
            current_font_size = constants.DEFAULT_FONT_SIZE

        # --- 加载字体 ---
        font = get_font(font_family_rel, current_font_size)
        if font is None:
            logger.error(f"气泡 {i}: 无法加载字体 {font_family_rel} (大小: {current_font_size})，跳过渲染。")
            continue

        # --- 计算绘制参数 ---
        offset_x = position_offset.get('x', 0)
        offset_y = position_offset.get('y', 0)
        max_text_width = max(10, bubble_width)
        max_text_height = max(10, bubble_height)

        # --- 调用绘制函数 ---
        try:
            if rotation_angle != 0:
                # ===== 旋转渲染：使用外接圆方案 =====
                # 计算外接圆直径（确保旋转后内容不被裁剪）
                diagonal = int(math.ceil(math.sqrt(bubble_width**2 + bubble_height**2)))
                # 增加一点边距，确保描边等不会被裁剪
                padding = max(10, int(stroke_width * 2) if stroke_enabled else 0)
                temp_size = diagonal + padding * 2
                
                # 创建外接圆大小的透明临时图像
                temp_img = Image.new('RGBA', (temp_size, temp_size), (0, 0, 0, 0))
                temp_draw = ImageDraw.Draw(temp_img)
                
                # 计算气泡在临时图像中的居中偏移
                temp_offset_x = (temp_size - bubble_width) // 2
                temp_offset_y = (temp_size - bubble_height) // 2
                
                # 在临时图像上绘制文字（相对于临时图像的坐标）
                if text_direction == 'vertical':
                    # 竖排时，x是右边界
                    temp_vertical_x = temp_offset_x + bubble_width
                    draw_multiline_text_vertical(
                        temp_draw, text, font, 
                        temp_vertical_x, temp_offset_y, max_text_height,
                        fill=text_color,
                        stroke_enabled=stroke_enabled,
                        stroke_color=stroke_color,
                        stroke_width=stroke_width,
                        bubble_width=max_text_width
                    )
                elif text_direction == 'horizontal':
                    draw_multiline_text_horizontal(
                        temp_draw, text, font,
                        temp_offset_x, temp_offset_y, max_text_width,
                        fill=text_color,
                        stroke_enabled=stroke_enabled,
                        stroke_color=stroke_color,
                        stroke_width=stroke_width,
                        bubble_width=max_text_width,
                        bubble_height=max_text_height
                    )
                else:
                    logger.warning(f"气泡 {i}: 未知的文本方向 '{text_direction}'，跳过渲染。")
                    continue
                
                # 以临时图像中心为旋转中心进行旋转
                # 注意：PIL的rotate是逆时针旋转，检测角度是顺时针，所以取反
                temp_center = temp_size // 2
                rotated_img = temp_img.rotate(
                    -rotation_angle,  # 取反以匹配检测角度方向
                    resample=Image.Resampling.BICUBIC,
                    center=(temp_center, temp_center),
                    expand=False
                )
                
                # 计算粘贴位置：气泡中心 - 临时图像半边长 + 位置偏移
                bubble_center_x = (x1 + x2) // 2
                bubble_center_y = (y1 + y2) // 2
                paste_x = bubble_center_x - temp_center + offset_x
                paste_y = bubble_center_y - temp_center + offset_y
                
                # 粘贴到原图（使用 alpha 通道作为蒙版）
                draw_image.paste(rotated_img, (paste_x, paste_y), rotated_img)
                
            else:
                # ===== 无旋转：直接在原图上绘制 =====
                draw_x = x1 + offset_x
                draw_y = y1 + offset_y
                vertical_draw_x = x2 + offset_x
                
                if text_direction == 'vertical':
                    draw_multiline_text_vertical(
                        draw, text, font, vertical_draw_x, draw_y, max_text_height,
                        fill=text_color,
                        stroke_enabled=stroke_enabled,
                        stroke_color=stroke_color,
                        stroke_width=stroke_width,
                        bubble_width=max_text_width
                    )
                elif text_direction == 'horizontal':
                    draw_multiline_text_horizontal(
                        draw, text, font, draw_x, draw_y, max_text_width,
                        fill=text_color,
                        stroke_enabled=stroke_enabled,
                        stroke_color=stroke_color,
                        stroke_width=stroke_width,
                        bubble_width=max_text_width,
                        bubble_height=max_text_height
                    )
                else:
                    logger.warning(f"气泡 {i}: 未知的文本方向 '{text_direction}'，跳过渲染。")
                    
        except Exception as render_e:
             logger.error(f"渲染气泡 {i} 时出错: {render_e}", exc_info=True)

    logger.info("所有气泡文本渲染完成。")

def render_single_bubble(
    image,
    bubble_index,
    all_texts,
    bubble_coords,
    fontSize=constants.DEFAULT_FONT_SIZE,
    fontFamily=constants.DEFAULT_FONT_RELATIVE_PATH,
    text_direction=constants.DEFAULT_TEXT_DIRECTION,
    position_offset={'x': 0, 'y': 0},
    use_inpainting=False,
    is_single_bubble_style=False,
    text_color=constants.DEFAULT_TEXT_COLOR,
    rotation_angle=constants.DEFAULT_ROTATION_ANGLE,
    use_lama=False,
    fill_color=constants.DEFAULT_FILL_COLOR,
    stroke_enabled=constants.DEFAULT_STROKE_ENABLED,
    stroke_color=constants.DEFAULT_STROKE_COLOR,
    stroke_width=constants.DEFAULT_STROKE_WIDTH
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
    bubble_states_to_use = {}
    if hasattr(image, '_bubble_states') and isinstance(getattr(image, '_bubble_states'), dict):
         bubble_states_to_use = getattr(image, '_bubble_states').copy()
         bubble_states_to_use = {str(k): v for k, v in bubble_states_to_use.items()}
         logger.debug(f"单气泡渲染：从图像加载了 {len(bubble_states_to_use)} 个样式。")
    else:
         logger.warning("单气泡渲染：未找到保存的气泡样式，将创建默认样式。")
         # 如果图像没有样式，为所有气泡创建基于全局默认的样式
         global_font_size_setting = constants.DEFAULT_FONT_SIZE
         global_font_family = constants.DEFAULT_FONT_RELATIVE_PATH
         global_text_dir = constants.DEFAULT_TEXT_DIRECTION
         global_text_color = constants.DEFAULT_TEXT_COLOR
         global_rot_angle = constants.DEFAULT_ROTATION_ANGLE
    
         for i in range(len(bubble_coords)):
             bubble_states_to_use[str(i)] = {
                 'fontSize': global_font_size_setting,
                 'fontFamily': global_font_family, 'text_direction': global_text_dir,
                 'position_offset': {'x': 0, 'y': 0}, 'text_color': global_text_color,
                 'rotation_angle': global_rot_angle,
                 'stroke_enabled': stroke_enabled,
                 'stroke_color': stroke_color,
                 'stroke_width': stroke_width
             }

    # --- 更新目标气泡的样式 ---
    target_style = bubble_states_to_use.get(str(bubble_index), {}).copy()
    target_font_rel = fontFamily
    
    # 直接使用传入的字号（已经在首次翻译时计算好了）
    actual_font_size = fontSize if isinstance(fontSize, int) and fontSize > 0 else constants.DEFAULT_FONT_SIZE
    
    target_style.update({
        'fontSize': actual_font_size,
        'fontFamily': target_font_rel,
        'text_direction': text_direction,
        'position_offset': position_offset,
        'text_color': text_color,
        'rotation_angle': rotation_angle,
        'stroke_enabled': stroke_enabled,
        'stroke_color': stroke_color,
        'stroke_width': stroke_width
    })

    bubble_states_to_use[str(bubble_index)] = target_style
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
        bubble_states_to_use # 传递更新后的样式字典
    )

    # --- 准备返回值 ---
    img_with_bubbles_pil = img_pil
    # 附加必要的属性
    if hasattr(image, '_lama_inpainted'): setattr(img_with_bubbles_pil, '_lama_inpainted', getattr(image, '_lama_inpainted', False))
    if clean_image_base:
         setattr(img_with_bubbles_pil, '_clean_image', clean_image_base)
         setattr(img_with_bubbles_pil, '_clean_background', clean_image_base)
    # 附加更新后的样式
    setattr(img_with_bubbles_pil, '_bubble_states', bubble_states_to_use)

    return img_with_bubbles_pil

def re_render_text_in_bubbles(
    image,
    all_texts,
    bubble_coords,
    fontSize=constants.DEFAULT_FONT_SIZE,
    fontFamily=constants.DEFAULT_FONT_RELATIVE_PATH,
    text_direction=constants.DEFAULT_TEXT_DIRECTION,
    use_inpainting=False,
    use_lama=False,
    fill_color=constants.DEFAULT_FILL_COLOR,
    text_color=constants.DEFAULT_TEXT_COLOR,
    rotation_angle=constants.DEFAULT_ROTATION_ANGLE,
    stroke_enabled=constants.DEFAULT_STROKE_ENABLED,
    stroke_color=constants.DEFAULT_STROKE_COLOR,
    stroke_width=constants.DEFAULT_STROKE_WIDTH
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
            image, bubble_coords, method=inpainting_method, fill_color=fill_color
        )
        if generated_clean_bg: clean_image_base = generated_clean_bg.copy()

    # --- 准备样式字典 ---
    bubble_states_to_use = {}
    
    # 检查图像是否已经有预定义的气泡样式字典
    if hasattr(image, '_bubble_states') and isinstance(getattr(image, '_bubble_states'), dict):
        # 优先使用预定义样式
        bubble_states_to_use = getattr(image, '_bubble_states').copy() # 深拷贝
        bubble_states_to_use = {str(k): v for k, v in bubble_states_to_use.items()}
        logger.info(f"使用图像预定义的气泡样式，共 {len(bubble_states_to_use)} 个")
        for i_str in bubble_states_to_use:
            if 'stroke_enabled' not in bubble_states_to_use[i_str]:
                bubble_states_to_use[i_str]['stroke_enabled'] = stroke_enabled
            if 'stroke_color' not in bubble_states_to_use[i_str]:
                bubble_states_to_use[i_str]['stroke_color'] = stroke_color
            if 'stroke_width' not in bubble_states_to_use[i_str]:
                bubble_states_to_use[i_str]['stroke_width'] = stroke_width
    else:
        # 没有预定义样式，使用全局设置创建新样式
        logger.info("没有找到预定义气泡样式，使用全局设置创建样式")
        
        font_family_rel = fontFamily
        # 直接使用传入的字号
        actual_font_size = fontSize if isinstance(fontSize, int) and fontSize > 0 else constants.DEFAULT_FONT_SIZE
        
        logger.info(f"使用传入的全局颜色设置: {text_color}, 旋转角度: {rotation_angle}")
        
        # 为所有气泡创建新的样式字典，使用全局设置
        for i in range(len(bubble_coords)):
            bubble_states_to_use[str(i)] = {
                'fontSize': actual_font_size,
                'fontFamily': font_family_rel,
                'text_direction': text_direction,
                'position_offset': {'x': 0, 'y': 0},
                'text_color': text_color,
                'rotation_angle': rotation_angle,
                'stroke_enabled': stroke_enabled,
                'stroke_color': stroke_color,
                'stroke_width': stroke_width
            }

    # --- 调用核心渲染函数 ---
    render_all_bubbles(
        img_pil, # 在获取的基础图像上绘制
        all_texts,
        bubble_coords,
        bubble_states_to_use
    )

    # --- 准备返回值 ---
    img_with_bubbles_pil = img_pil
    # 附加必要的属性
    if hasattr(image, '_lama_inpainted'): setattr(img_with_bubbles_pil, '_lama_inpainted', getattr(image, '_lama_inpainted', False))
    if clean_image_base:
         setattr(img_with_bubbles_pil, '_clean_image', clean_image_base)
         setattr(img_with_bubbles_pil, '_clean_background', clean_image_base)
    setattr(img_with_bubbles_pil, '_bubble_states', bubble_states_to_use) # 附加更新后的样式

    return img_with_bubbles_pil


# ============================================================
# 统一渲染函数（使用 BubbleState）
# ============================================================

def render_bubbles_unified(
    image: Image.Image,
    bubble_states: List["BubbleState"]
) -> Image.Image:
    """
    使用统一的 BubbleState 列表渲染所有气泡文本。
    
    这是新的核心渲染入口，所有渲染操作都应该通过此函数。
    它只依赖 BubbleState 列表，不再需要其他分散的参数。
    
    Args:
        image: 要绘制文本的 PIL 图像对象（会被直接修改）
        bubble_states: BubbleState 对象列表，包含每个气泡的完整状态
        
    Returns:
        处理后的图像（同一个对象，已被修改）
    """
    if not bubble_states:
        logger.warning("bubble_states 为空，跳过渲染。")
        return image
    
    draw = ImageDraw.Draw(image)
    logger.info(f"[统一渲染] 开始渲染 {len(bubble_states)} 个气泡...")
    
    for i, state in enumerate(bubble_states):
        text = state.translated_text
        if not text:
            continue
        
        x1, y1, x2, y2 = state.coords
        bubble_width = x2 - x1
        bubble_height = y2 - y1
        
        if bubble_width <= 0 or bubble_height <= 0:
            logger.warning(f"气泡 {i} 坐标无效: {state.coords}，跳过。")
            continue
        
        # 直接使用保存的字号
        current_font_size = state.font_size if state.font_size > 0 else constants.DEFAULT_FONT_SIZE
        
        # 加载字体
        font = get_font(state.font_family, current_font_size)
        if font is None:
            logger.error(f"气泡 {i}: 无法加载字体 {state.font_family}，跳过渲染。")
            continue
        
        # 计算绘制参数
        offset_x = state.position_offset.get('x', 0)
        offset_y = state.position_offset.get('y', 0)
        max_text_width = max(10, bubble_width)
        max_text_height = max(10, bubble_height)
        
        try:
            if state.rotation_angle != 0:
                # === 旋转渲染：使用外接圆方案 ===
                diagonal = int(math.ceil(math.sqrt(bubble_width**2 + bubble_height**2)))
                padding = max(10, int(state.stroke_width * 2) if state.stroke_enabled else 0)
                temp_size = diagonal + padding * 2
                
                temp_img = Image.new('RGBA', (temp_size, temp_size), (0, 0, 0, 0))
                temp_draw = ImageDraw.Draw(temp_img)
                
                temp_offset_x = (temp_size - bubble_width) // 2
                temp_offset_y = (temp_size - bubble_height) // 2
                
                if state.text_direction == 'vertical':
                    temp_vertical_x = temp_offset_x + bubble_width
                    draw_multiline_text_vertical(
                        temp_draw, text, font,
                        temp_vertical_x, temp_offset_y, max_text_height,
                        fill=state.text_color,
                        stroke_enabled=state.stroke_enabled,
                        stroke_color=state.stroke_color,
                        stroke_width=state.stroke_width,
                        bubble_width=max_text_width
                    )
                else:
                    draw_multiline_text_horizontal(
                        temp_draw, text, font,
                        temp_offset_x, temp_offset_y, max_text_width,
                        fill=state.text_color,
                        stroke_enabled=state.stroke_enabled,
                        stroke_color=state.stroke_color,
                        stroke_width=state.stroke_width,
                        bubble_width=max_text_width,
                        bubble_height=max_text_height
                    )
                
                temp_center = temp_size // 2
                rotated_img = temp_img.rotate(
                    -state.rotation_angle,
                    resample=Image.Resampling.BICUBIC,
                    center=(temp_center, temp_center),
                    expand=False
                )
                
                bubble_center_x = (x1 + x2) // 2
                bubble_center_y = (y1 + y2) // 2
                paste_x = bubble_center_x - temp_center + offset_x
                paste_y = bubble_center_y - temp_center + offset_y
                
                image.paste(rotated_img, (paste_x, paste_y), rotated_img)
                
            else:
                # === 无旋转：直接绘制 ===
                draw_x = x1 + offset_x
                draw_y = y1 + offset_y
                vertical_draw_x = x2 + offset_x
                
                if state.text_direction == 'vertical':
                    draw_multiline_text_vertical(
                        draw, text, font, vertical_draw_x, draw_y, max_text_height,
                        fill=state.text_color,
                        stroke_enabled=state.stroke_enabled,
                        stroke_color=state.stroke_color,
                        stroke_width=state.stroke_width,
                        bubble_width=max_text_width
                    )
                else:
                    draw_multiline_text_horizontal(
                        draw, text, font, draw_x, draw_y, max_text_width,
                        fill=state.text_color,
                        stroke_enabled=state.stroke_enabled,
                        stroke_color=state.stroke_color,
                        stroke_width=state.stroke_width,
                        bubble_width=max_text_width,
                        bubble_height=max_text_height
                    )
                    
        except Exception as render_e:
            logger.error(f"渲染气泡 {i} 时出错: {render_e}", exc_info=True)
    
    logger.info("[统一渲染] 所有气泡文本渲染完成。")
    return image


def render_single_bubble_unified(
    image: Image.Image,
    bubble_states: List["BubbleState"],
    bubble_index: int,
    use_clean_background: bool = True
) -> Image.Image:
    """
    使用统一的 BubbleState 重新渲染单个气泡。
    
    会在干净背景上重新渲染所有气泡，以确保其他气泡不受影响。
    
    Args:
        image: 当前图像（需要有 _clean_image 或 _clean_background 属性）
        bubble_states: 完整的 BubbleState 列表
        bubble_index: 要更新的气泡索引
        use_clean_background: 是否使用干净背景重渲染
        
    Returns:
        渲染后的图像
    """
    if bubble_index < 0 or bubble_index >= len(bubble_states):
        logger.error(f"无效的气泡索引 {bubble_index}")
        return image
    
    # 获取干净背景
    img_to_render = None
    clean_image_base = None
    
    if use_clean_background:
        if hasattr(image, '_clean_image') and isinstance(getattr(image, '_clean_image'), Image.Image):
            clean_image_base = getattr(image, '_clean_image').copy()
            img_to_render = clean_image_base
        elif hasattr(image, '_clean_background') and isinstance(getattr(image, '_clean_background'), Image.Image):
            clean_image_base = getattr(image, '_clean_background').copy()
            img_to_render = clean_image_base
    
    if img_to_render is None:
        logger.warning("未找到干净背景，将对当前图像执行修复...")
        # 仅修复目标气泡区域
        from src.core.inpainting import inpaint_bubbles
        target_state = bubble_states[bubble_index]
        target_coords = [list(target_state.coords)]
        
        inpaint_method = target_state.inpaint_method if target_state.inpaint_method else 'solid'
        img_to_render, generated_clean_bg = inpaint_bubbles(
            image, target_coords, method=inpaint_method, fill_color=target_state.fill_color
        )
        if generated_clean_bg:
            clean_image_base = generated_clean_bg.copy()
    
    # 渲染所有气泡
    render_bubbles_unified(img_to_render, bubble_states)
    
    # 附加属性
    if hasattr(image, '_lama_inpainted'):
        setattr(img_to_render, '_lama_inpainted', getattr(image, '_lama_inpainted', False))
    if clean_image_base:
        setattr(img_to_render, '_clean_image', clean_image_base)
        setattr(img_to_render, '_clean_background', clean_image_base)
    
    # 附加 BubbleState 列表
    setattr(img_to_render, '_bubble_states', bubble_states)
    
    return img_to_render


def re_render_with_states(
    image: Image.Image,
    bubble_states: List["BubbleState"],
    use_lama: bool = False,
    fill_color: str = constants.DEFAULT_FILL_COLOR,
    auto_font_size: bool = False
) -> Image.Image:
    """
    使用 BubbleState 列表重新渲染整个图像。
    
    这是给 re_render_image API 使用的统一函数。
    
    Args:
        image: 当前图像
        bubble_states: BubbleState 列表
        use_lama: 是否使用 LAMA 修复（如果没有干净背景）
        fill_color: 默认填充色（如果没有干净背景）
        auto_font_size: 是否为每个气泡自动计算字号
        
    Returns:
        渲染后的图像
    """
    if not bubble_states:
        logger.warning("bubble_states 为空，返回原图像。")
        return image
    
    # 获取干净背景
    img_to_render = None
    clean_image_base = None
    
    if hasattr(image, '_clean_image') and isinstance(getattr(image, '_clean_image'), Image.Image):
        clean_image_base = getattr(image, '_clean_image').copy()
        img_to_render = clean_image_base
        logger.info("re_render_with_states: 使用 _clean_image 作为基础。")
    elif hasattr(image, '_clean_background') and isinstance(getattr(image, '_clean_background'), Image.Image):
        clean_image_base = getattr(image, '_clean_background').copy()
        img_to_render = clean_image_base
        logger.info("re_render_with_states: 使用 _clean_background 作为基础。")
    
    if img_to_render is None:
        logger.warning("re_render_with_states: 未找到干净背景，将重新执行修复...")
        from src.core.inpainting import inpaint_bubbles
        from src.interfaces.lama_interface import is_lama_available
        
        bubble_coords = [list(s.coords) for s in bubble_states]
        inpainting_method = 'solid'
        if use_lama and is_lama_available():
            inpainting_method = 'lama'
        
        img_to_render, generated_clean_bg = inpaint_bubbles(
            image, bubble_coords, method=inpainting_method, fill_color=fill_color
        )
        if generated_clean_bg:
            clean_image_base = generated_clean_bg.copy()
    
    # 如果启用自动字号，为每个气泡计算字号
    if auto_font_size:
        logger.info("re_render_with_states: 启用自动字号计算...")
        for i, state in enumerate(bubble_states):
            if state.translated_text:
                x1, y1, x2, y2 = state.coords
                bubble_width = x2 - x1
                bubble_height = y2 - y1
                calculated_size = calculate_auto_font_size(
                    state.translated_text, bubble_width, bubble_height,
                    state.text_direction, state.font_family
                )
                state.font_size = calculated_size
                logger.debug(f"气泡 {i}: 自动计算字号为 {calculated_size}px")
    
    # 渲染所有气泡
    render_bubbles_unified(img_to_render, bubble_states)
    
    # 附加属性
    if hasattr(image, '_lama_inpainted'):
        setattr(img_to_render, '_lama_inpainted', getattr(image, '_lama_inpainted', False))
    if clean_image_base:
        setattr(img_to_render, '_clean_image', clean_image_base)
        setattr(img_to_render, '_clean_background', clean_image_base)
    
    # 附加 BubbleState 列表
    setattr(img_to_render, '_bubble_states', bubble_states)
    
    return img_to_render


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