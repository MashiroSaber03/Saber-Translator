import logging
import math
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# FreeType 字体回退支持
try:
    import freetype
    FREETYPE_AVAILABLE = True
except ImportError:
    FREETYPE_AVAILABLE = False
    logging.warning("freetype-py 未安装，将使用简化的字体回退机制")

# 导入常量和路径助手
from src.shared import constants
from src.shared.memory_errors import is_memory_allocation_error
from src.shared.path_helpers import get_font_path, resource_path

# 类型提示（避免循环导入）
if TYPE_CHECKING:
    from src.core.config_models import BubbleState

logger = logging.getLogger("CoreRendering")
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# =============================================================================
# 字体回退系统
# =============================================================================

# --- 字体缓存 ---
_font_cache = {}  # Pillow 字体缓存
_freetype_font_cache = {}  # FreeType 字体缓存
_font_file_handles = {}  # 保存文件句柄，防止被垃圾回收

# --- 字体路径 ---
FONTS_DIR = os.path.join('src', 'backend_v2', 'resources', 'fonts')

# --- 回退字体列表 ---
FALLBACK_FONTS = [
    os.path.join(FONTS_DIR, 'Arial_Unicode.ttf'),   # Unicode 全字符字体
    os.path.join(FONTS_DIR, 'msyh.ttc'),            # 微软雅黑
    os.path.join(FONTS_DIR, 'msgothic.ttc'),        # MS Gothic（日文哥特体）
]

# --- 特殊字符的字体路径（Pillow 回退用）---
NOTOSANS_FONT_PATH = os.path.join(FONTS_DIR, 'NotoSans-Medium.ttf')

# --- 需要使用特殊字体渲染的字符（Pillow 回退用）---
SPECIAL_CHARS = {'‼', '⁉', '⁇', '⁈', '︕', '︖', '︙', '⋮', '⋯'}

# =============================================================================
# 竖排标点符号处理系统
#
# 设计：竖排保留原字符并在渲染时整体旋转 90°，不再做字符替换。本节只保留若干
# 字符集合用于旋转判定：
#   - VERTICAL_FORM_CHARS：自身已经是竖排字形的字符，
#     遇到时保持原样、不再二次旋转
#   - UPRIGHT_IN_VERTICAL：竖排下保持直立的标点（感叹/问/句/逗/冒/分号等）
#   - EXTRA_VERTICAL_ROTATE_CHARS：Unicode 类别非 P 但应旋转的字符（⋯ ～）
# =============================================================================

# --- 已是竖排字形的字符集合（用户直接输入时不再二次旋转） ---
# 涵盖 CJK Compatibility Forms 的竖排变体、组合标点族、以及旋转对称字符。
# 注：· 仅放在 UPRIGHT_IN_VERTICAL 里，不在两个集合中重复。
VERTICAL_FORM_CHARS = {
    # 组合标点族（已是紧凑竖排字形）
    '‼', '⁉', '⁇', '⁈',
    # ASCII 旋转对称字符
    '|',
    # 竖向省略号
    '⋮', '︙',
    # 竖排句读（CJK Compatibility Forms）
    '︐', '︑', '︒', '︓', '︔', '︕', '︖',
    # 两点省略 / 破折号 / 连字符 / 下划 / 波浪（竖向）
    '︰', '︱', '︲', '︳', '︴',
    # 竖排括号（圆/花/龟甲/方头/书名/尖/方/白方头）
    '︵', '︶', '︷', '︸', '︹', '︺', '︻', '︼', '︽', '︾', '︿', '﹀',
    '︗', '︘', '﹇', '﹈',
    # 竖排引号（日式 corner bracket）
    '﹁', '﹂', '﹃', '﹄',
    # 着重号
    '﹅', '﹆',
    # 竖向虚线 / 波浪线装饰
    '﹉', '﹊', '﹋', '﹌', '﹍', '﹎', '﹏',
}

# --- 竖排下保持直立（不旋转）的标点集合 ---
# 这些字符旋转 90° 后字形会"躺倒"（叹号/问号的主笔画变横向、点变成侧面），
# 视觉上不自然。CJK 竖排惯例是：感叹/问号、句读点、中黑点等保持直立。
# 注意：冒号分号（: ; ： ；）走旋转路径——它们的竖排传统形态是"两点横向并列"
# （CJK Compatibility Forms 的 ︓ ︔），恰好是 90° 旋转的结果。
UPRIGHT_IN_VERTICAL = {
    # 感叹号 / 问号
    '!', '?', '！', '？', '¿', '¡', '؟',
    # 句点
    '.', '。',
    # 逗号 / 顿号
    ',', '，', '、',
    # 中黑点 / 两点省略
    '·', '・', '‥',
}

# --- 竖排下需要做低位标点垂直校正的字符集合 ---
# 这些标点在大多数字体里的墨水区域天然贴近基线，若直接按 current_y_char 绘制，
# 会落在单元格底部，看起来像“沉下去”。因此仍然需要把它们往单元格视觉中心拉回。
# 注意：感叹号 / 问号虽然也属于 UPRIGHT_IN_VERTICAL，但很多字体中它们本来就位于
# 字面中上部，再做同样的居中校正会被明显上提，甚至侵入上一格，所以明确排除。
VERTICAL_CENTER_ADJUST_PUNCTUATION = {
    '.', '。',
    ',', '，', '、',
    '·', '・', '‥',
}

# --- 需要以当前字体合成渲染的组合直立标点 ---
# 这些符号很多字体并不直接提供字形，现有逻辑会回退到别的字体绘制，导致位置和风格
# 与正文不一致。这里保留其“单字符语义”（单格、不断行），但绘制时改用当前字体把
# constituent 标点合成一个紧凑块。
COMBINED_UPRIGHT_SYMBOL_EXPANSIONS = {
    '‼': '!!',
    '⁉': '!?',
    '⁇': '??',
    '⁈': '?!',
}

# --- 竖排需要旋转但 Unicode 类别不是 P 的字符 ---
# 这些字符语义上是"线性延展"标点，但 unicodedata 把它们归为 Sm（数学符号），
# is_punctuation 返回 False。显式加入以触发 90° 旋转，与同族的 Pd 类保持一致。
EXTRA_VERTICAL_ROTATE_CHARS = {
    '⋯',   # U+22EF MIDLINE HORIZONTAL ELLIPSIS（居中省略号）
    '～',   # U+FF5E FULLWIDTH TILDE
    '−',   # U+2212 MINUS SIGN（视觉等同 -，应与破折号/连字符同步旋转）
}

# --- 单个线性延展标点也走块渲染路径 ---
# 连续段（长度 >= 2）已经会被包进 <E> 块；但单个 `… / — / ― / ─` 等此前仍走
# 单字符旋转路径，只做几何居中，容易比中文正文视觉中心偏上。把单个符号也交给
# render_ellipsis_block 处理，可以与连续块共享同一套视觉中心对齐逻辑。
SINGLE_LINEAR_BLOCK_CHARS = {
    '…', '⋯',
    '—', '–', '―', '─', '−',
    '～', '〜', '〰',
}

# --- 特殊组合标点映射 (保留用于组合符号处理) ---
SPECIAL_PUNCTUATION_PATTERNS = [
    ('...', '…'),      # 连续三个点先转为省略号
    ('..', '…'),       # 两个点也转为省略号
    ('!!!', '‼'),      # 连续三个感叹号映射成双感叹号
    ('!!', '‼'),       # 连续两个感叹号映射成双感叹号
    ('！！！', '‼'),   # 中文连续三个感叹号
    ('！！', '‼'),     # 中文连续两个感叹号
    ('???', '⁇'),      # 连续三个问号映射成双问号
    ('??', '⁇'),       # 连续两个问号映射成双问号
    ('？？？', '⁇'),   # 中文连续三个问号
    ('？？', '⁇'),     # 中文连续两个问号
    ('!?', '⁉'),       # 感叹号加问号映射成感叹问号组合
    ('?!', '⁈'),       # 问号加感叹号映射成问号感叹号组合
    ('！？', '⁉'),     # 中文感叹号加问号
    ('？！', '⁈'),     # 中文问号加感叹号
]


def is_punctuation(ch: str) -> bool:
    """
    检查字符是否为标点符号

    Args:
        ch: 单个字符

    Returns:
        是否为标点符号
    """
    import unicodedata

    cp = ord(ch)
    # ASCII 标点符号
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    # Unicode 标点类别
    cat = unicodedata.category(ch)
    if cat.startswith("P"):
        return True
    return False


# =============================================================================
# FreeType 字体回退系统
# =============================================================================

def get_cached_freetype_font(path: str) -> Optional["freetype.Face"]:
    """
    获取缓存的 FreeType 字体
    
    Args:
        path: 字体文件路径
        
    Returns:
        FreeType Face 对象，失败返回 None
    """
    if not FREETYPE_AVAILABLE:
        return None
    
    path = path.replace('\\', '/')
    if path not in _freetype_font_cache:
        try:
            # 使用 resource_path 处理打包后的路径
            abs_path = resource_path(path)
            if not os.path.exists(abs_path):
                abs_path = get_font_path(path)
            
            if os.path.exists(abs_path):
                # 保存文件句柄引用，防止被关闭
                file_handle = Path(abs_path).open('rb')
                _font_file_handles[path] = file_handle
                _freetype_font_cache[path] = freetype.Face(file_handle)
                logger.debug(f"FreeType 字体加载成功: {abs_path}")
            else:
                logger.warning(f"FreeType 字体未找到: {path}")
                return None
        except Exception as e:
            if is_memory_allocation_error(e):
                raise
            logger.error(f"FreeType 字体加载失败: {path} - {e}")
            return None
    
    return _freetype_font_cache.get(path)


def font_supports_char(font_path: str, char: str) -> bool:
    """
    检查字体是否支持某个字符
    
    Args:
        font_path: 字体文件路径
        char: 要检查的字符
        
    Returns:
        是否支持
    """
    if not FREETYPE_AVAILABLE:
        return True  # 无法检查时假设支持
    
    face = get_cached_freetype_font(font_path)
    if face is None:
        return False
    
    return face.get_char_index(char) != 0


def get_char_ink_offset(char: str, font: ImageFont.FreeTypeFont) -> Tuple[float, float]:
    """
    获取字符的墨水偏移量（实际墨水中心相对于边界框中心的偏移）
    
    Pillow 的 getbbox() 返回的是字符的度量边界，但实际墨水可能偏向一侧。
    这个函数通过渲染字符并分析像素来找到实际墨水的范围和偏移。
    
    Args:
        char: 要分析的字符
        font: 字体对象
        
    Returns:
        (x_offset, y_offset): 墨水中心相对于边界框中心的偏移
    """
    try:
        bbox = font.getbbox(char)
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        
        if bbox_width <= 0 or bbox_height <= 0:
            return (0.0, 0.0)
        
        # 在临时图像上渲染字符
        padding = 20
        img_size = max(bbox_width, bbox_height) + padding * 2
        img = Image.new('L', (img_size, img_size), 255)
        draw = ImageDraw.Draw(img)
        
        # 在中心位置绘制
        x = (img_size - bbox_width) // 2
        y = (img_size - bbox_height) // 2
        draw.text((x, y), char, font=font, fill=0)
        
        # 转换为 numpy 数组并找到实际墨水范围
        arr = np.array(img)
        non_white = np.where(arr < 250)  # 稍微放宽阈值
        
        if len(non_white[0]) == 0:
            return (0.0, 0.0)
        
        min_y, max_y = non_white[0].min(), non_white[0].max()
        min_x, max_x = non_white[1].min(), non_white[1].max()
        
        # 计算实际墨水中心
        ink_center_x = (min_x + max_x) / 2.0
        ink_center_y = (min_y + max_y) / 2.0
        
        # 边界框中心
        bbox_center_x = x + bbox_width / 2.0
        bbox_center_y = y + bbox_height / 2.0
        
        # 偏移量
        offset_x = ink_center_x - bbox_center_x
        offset_y = ink_center_y - bbox_center_y
        
        return (offset_x, offset_y)
        
    except Exception as e:
        if is_memory_allocation_error(e):
            raise
        logger.debug(f"获取字符 '{char}' 墨水偏移时出错: {e}")
        return (0.0, 0.0)


def compact_special_symbols(text: str) -> str:
    """
    预处理特殊符号

    处理逻辑：
    - 将连续的半角点（.. / ...）合并为省略号 …

    注意事项：
    - 不再做 … → ⋯ 的二次替换：该替换原本是横排下为解决 U+2026 贴底问题加的补丁，
      但会使字符码点从主字体常见的 U+2026 变为冷门的 U+22EF，在竖排旋转路径下触发
      字体回退，导致省略号的字形风格与正文不一致（用户感知为"字体变了"）。
    - 不再合并连续省略号，以保留原文的情感表达层次
      (例如: ...... 表示长时间沉默，不应被压缩为 ...)
    - 不删除标点后的空格，保留用户/AI输出的原始格式

    Args:
        text: 原始文本

    Returns:
        处理后的文本
    """
    if not text:
        return text

    # 将半角点合并为省略号
    text = text.replace('...', '…')
    text = text.replace('..', '…')

    return text


def CJK_Compatibility_Forms_translate(cdpt: str, direction: int) -> Tuple[str, int]:
    """
    决定字符在竖排渲染时的旋转角度（不再做字符替换）。

    新策略：保留原字符，竖排下对"线性延展"类标点（括号、破折号、波浪、引号、
    省略号等）整体旋转 90°；对"点状/句读"类标点（感叹/问/句/逗/冒/分号等）
    保持直立以避免字形躺倒。

    Args:
        cdpt: 单个字符
        direction: 0 = 横排，1 = 竖排

    Returns:
        Tuple[str, int]: (字符, 旋转角度)
        - 横排：始终 (cdpt, 0)
        - 竖排：
            * ー 返回 (ー, 90)
            * UPRIGHT_IN_VERTICAL（感叹/问/句/逗/冒/分号等）保持 0°
            * VERTICAL_FORM_CHARS（CJK 竖排字形 + 组合强调符号）保持 0°
            * 其余标点及 EXTRA_VERTICAL_ROTATE_CHARS 返回 (cdpt, 90)
            * 非标点（汉字/假名等）保持 0°
    """
    if direction == 0:
        return cdpt, 0

    if cdpt == 'ー':
        return cdpt, 90

    # 直立：CJK 竖排下这些标点保持原方向，旋转会导致字形躺倒
    if cdpt in UPRIGHT_IN_VERTICAL or cdpt in VERTICAL_FORM_CHARS:
        return cdpt, 0

    if is_punctuation(cdpt) or cdpt in EXTRA_VERTICAL_ROTATE_CHARS:
        return cdpt, 90

    return cdpt, 0


def get_vertical_center_adjusted_y(char: str, font: ImageFont.FreeTypeFont,
                                   current_y_char: float, line_height_approx: int) -> float:
    """
    计算竖排直立标点的垂直校正后 Y 坐标。

    仅对低位句读点执行校正；感叹号 / 问号等高位符号保持原始 baseline 位置，
    避免某些字体的 bbox 度量把它们额外上提，造成与上一格文字发生重叠。
    """
    if char not in VERTICAL_CENTER_ADJUST_PUNCTUATION:
        return current_y_char

    try:
        ink_bbox = font.getbbox(char)
        ink_mid_y = (ink_bbox[1] + ink_bbox[3]) / 2.0
        target_mid_y = line_height_approx / 2.0
        raw_shift = target_mid_y - ink_mid_y

        # 低位标点需要明显上提，但仍限制在合理范围内，避免异常字体度量导致越格。
        max_shift = line_height_approx * 0.75
        clamped_shift = max(-max_shift, min(max_shift, raw_shift))
        return current_y_char + clamped_shift
    except Exception as error:
        if is_memory_allocation_error(error):
            raise
        return current_y_char


def _close_images(*images: Optional[Image.Image]) -> None:
    seen: set[int] = set()
    for image in images:
        if image is None or id(image) in seen:
            continue
        seen.add(id(image))
        image.close()


def render_combined_upright_symbol(symbol: str, font: ImageFont.FreeTypeFont,
                                   fill, stroke_enabled: bool, stroke_color,
                                   stroke_width, canvas_image: Optional[Image.Image],
                                   current_x_col: int, current_y: float,
                                   line_width: int, line_height_unit: int) -> int:
    """
    使用当前字体为竖排组合标点合成一个紧凑块，并作为单个单元格绘制。

    目的：
    - 保留 `⁉ / ‼ / ⁇ / ⁈` 的“单格、不断行”语义；
    - 避免因为主字体缺少组合字形而回退到其它字体，导致视觉位置明显上飘。
    """
    content = COMBINED_UPRIGHT_SYMBOL_EXPANSIONS.get(symbol)
    if not content or canvas_image is None:
        return line_height_unit

    char_bboxes = []
    total_width = 0
    reference_top = None
    reference_bottom = None

    for char in content:
        bbox = font.getbbox(char)
        char_width = max(1, bbox[2] - bbox[0])
        char_bboxes.append((char, bbox, char_width))
        total_width += char_width
        reference_top = bbox[1] if reference_top is None else min(reference_top, bbox[1])
        reference_bottom = bbox[3] if reference_bottom is None else max(reference_bottom, bbox[3])

    if not char_bboxes or reference_top is None or reference_bottom is None:
        return line_height_unit

    spacing = max(1, int(font.size * 0.04))
    total_width += spacing * (len(char_bboxes) - 1)
    content_height = max(1, reference_bottom - reference_top)

    padding = max(10, int(stroke_width * 2) if stroke_enabled else 10)
    temp_w = total_width + padding * 2
    temp_h = content_height + padding * 2
    temp_img = Image.new('RGBA', (temp_w, temp_h), (0, 0, 0, 0))
    cropped = None
    resized = None
    try:
        temp_draw = ImageDraw.Draw(temp_img)

        text_params = {'font': font, 'fill': fill}
        if stroke_enabled:
            text_params['stroke_width'] = int(stroke_width)
            text_params['stroke_fill'] = stroke_color

        pen_x = padding
        draw_y = padding - reference_top
        for char, _bbox, char_width in char_bboxes:
            temp_draw.text((pen_x, draw_y), char, **text_params)
            pen_x += char_width + spacing

        temp_arr = np.array(temp_img)
        alpha = temp_arr[:, :, 3]
        non_zero = np.where(alpha > 10)
        if len(non_zero[0]) == 0:
            return line_height_unit

        min_y, max_y = non_zero[0].min(), non_zero[0].max()
        min_x, max_x = non_zero[1].min(), non_zero[1].max()
        cropped = temp_img.crop((min_x, min_y, max_x + 1, max_y + 1))

        scale_x = min(1.0, line_width / max(1, cropped.width))
        scale_y = min(1.0, line_height_unit / max(1, cropped.height))
        if scale_x < 1.0 or scale_y < 1.0:
            resized = cropped.resize(
                (
                    max(1, int(round(cropped.width * scale_x))),
                    max(1, int(round(cropped.height * scale_y))),
                ),
                resample=Image.Resampling.BICUBIC,
            )
        else:
            resized = cropped

        paste_x = int(
            (current_x_col - line_width) + (line_width - resized.width) / 2.0
        )
        paste_y = int(current_y + reference_top * scale_y)
        _paste_with_alpha(canvas_image, resized, paste_x, paste_y)
        return line_height_unit
    finally:
        _close_images(resized, cropped, temp_img)


def auto_add_horizontal_tags(text: str) -> str:
    """
    自动为竖排文本中的短英文单词或连续符号添加<H>标签，使其横向显示。
    
    处理规则：
    - 多词英文词组（如 "Tik Tok"）：整体横排显示
    - 独立的短英文单词（2个及以上字符）：添加<H>标签
    - 连续符号（!?）2-4个：横排显示
    
    渲染规则（在渲染时根据长度决定）：
    - 2个字符：横排显示
    - 3个及以上字符：竖排显示但每个字符旋转90度
    
    Args:
        text: 原始文本
        
    Returns:
        添加了<H>标签的文本
    """
    if not text:
        return text
    
    # 如果文本中已有<H>标签，则不进行处理，以尊重手动设置
    if '<H>' in text or '<h>' in text.lower():
        return text
    
    # 步骤1：为多词英文词组添加<H>标签（至少2个单词，用空格分隔）
    # 匹配：字母/数字 + 空格 + 字母/数字（可以重复多次）
    # 注意：移除了点号(.)以避免匹配省略号
    multi_word_pattern = r'[a-zA-Z0-9\uff21-\uff3a\uff41-\uff5a\uff10-\uff19_-]+(?:\s+[a-zA-Z0-9\uff21-\uff3a\uff41-\uff5a\uff10-\uff19_-]+)+'
    text = re.sub(multi_word_pattern, r'<H>\g<0></H>', text)
    
    # 步骤2：对剩余的独立英文单词添加<H>标签
    # 匹配2个及以上字符，排除已经在<H>标签内的内容
    word_pattern = r'(?<![a-zA-Z0-9\uff21-\uff3a\uff41-\uff5a\uff10-\uff19_-])([a-zA-Z0-9\uff21-\uff3a\uff41-\uff5a\uff10-\uff19_-]{2,})(?![a-zA-Z0-9\uff21-\uff3a\uff41-\uff5a\uff10-\uff19_-])'
    
    # 只替换不在<H>标签内的匹配
    def replace_word(match):
        # 检查匹配位置是否在<H>...</H>之间
        start_pos = match.start()
        # 简单检查：查找前面最近的<H>和</H>
        text_before = text[:start_pos]
        last_open = text_before.rfind('<H>')
        last_close = text_before.rfind('</H>')
        if last_open > last_close:
            # 在<H>标签内，不替换
            return match.group(0)
        return f'<H>{match.group(1)}</H>'
    
    text = re.sub(word_pattern, replace_word, text)
    
    # 步骤3：匹配连续符号（2-4个，同时支持半角和全角）
    symbol_pattern = r'[!?！？]{2,4}'
    text = re.sub(symbol_pattern, r'<H>\g<0></H>', text)
    
    return text

def process_text_for_vertical(text: str) -> str:
    """
    为竖排渲染预处理文本

    竖排不再做字符替换，字符级的旋转在 draw_multiline_text_vertical 中
    按 CJK_Compatibility_Forms_translate 的返回值逐字符决定。

    处理流程：
    1. 调用 compact_special_symbols 统一省略号格式（... → …）
    2. 处理特殊组合标点（如 !! → ‼，?? → ⁇，!? → ⁉，?! → ⁈）
    3. 把连续的"线性延展"标点（省略号 / 破折号 / 波浪，长度 >= 2）
       包成 <E>...</E> 块，渲染时整段旋转以消除逐字旋转导致的拼接缝
    4. 自动为英文/数字添加 <H> 横排标签

    Args:
        text: 原始文本

    Returns:
        预处理后的文本（尚未进行字符转换）
    """
    if not text:
        return text

    # 步骤1: 预处理特殊符号（合并 ... 为 …）
    text = compact_special_symbols(text)

    # 步骤2: 处理特殊组合标点（合并后由渲染层决定是否旋转/合成绘制）
    for pattern, replacement in SPECIAL_PUNCTUATION_PATTERNS:
        text = text.replace(pattern, replacement)

    # 步骤3: 聚合"线性延展"类标点的连续段为 <E> 块（≥ 2 个才打包）
    # 目的：这些字符单独旋转后相邻两个之间会出现明显拼接缝（例如 —— 出现 ~10px
    # 横向空隙，…… 出现点间断开）。打包成一段整体旋转，一次墨迹、无缝连接。
    # 各族内部连续才聚合，跨族不合并（如 —～ 不打包）。
    for pattern in (
        r'[…⋯]{2,}',          # 省略号族
        r'[—–―─]{2,}',        # 破折号族（EM / EN / HORIZONTAL BAR / BOX DRAWINGS LIGHT HORIZONTAL）
        r'[～〜〰]{2,}',        # 波浪族
    ):
        text = re.sub(pattern, lambda m: f'<E>{m.group(0)}</E>', text)

    # 步骤4: 自动为英文/数字添加 <H> 横排标签
    text = auto_add_horizontal_tags(text)

    return text


def get_font(font_family_relative_path=constants.DEFAULT_FONT_RELATIVE_PATH, font_size=constants.DEFAULT_FONT_SIZE):
    """
    加载当前 v2 文档已经解析出的字体文件，带缓存。

    Args:
        font_family_relative_path (str): 字体的相对路径 (相对于项目根目录)。
        font_size (int): 字体大小。

    Returns:
        ImageFont.FreeTypeFont: 加载的字体对象。

    Raises:
        ValueError: 字号不是正整数。
        FileNotFoundError: 字体文件不存在。
        OSError: 字体文件无法被 Pillow 读取。
    """
    try:
        font_size = int(font_size)
    except (ValueError, TypeError) as exc:
        raise ValueError("font_size must be a positive integer") from exc
    if font_size <= 0:
        raise ValueError("font_size must be a positive integer")

    cache_key = (font_family_relative_path, font_size)
    if cache_key in _font_cache:
        return _font_cache[cache_key]

    font_path_abs = get_font_path(font_family_relative_path)
    font = ImageFont.truetype(font_path_abs, font_size, encoding="utf-8")
    logger.info(f"成功加载字体: {font_path_abs} (大小: {font_size})")

    _font_cache[cache_key] = font
    return font

def calculate_auto_font_size(text, bubble_width, bubble_height, text_direction='vertical',
                             font_family_relative_path=constants.DEFAULT_FONT_RELATIVE_PATH,
                             min_size=12, max_size=80, padding_ratio=1.0):
    """
    使用二分法计算最佳字体大小。
    
    对于包含换行符的文本，会考虑换行符对布局的影响：
    - 竖排模式：每个换行符代表一个新列
    - 横排模式：每个换行符代表一个新行
    """
    if not text or not text.strip() or bubble_width <= 0 or bubble_height <= 0:
        return constants.DEFAULT_FONT_SIZE

    W = bubble_width * padding_ratio
    H = bubble_height * padding_ratio
    
    # 处理换行符：分割成段落，计算每个段落的字符数
    paragraphs = text.split('\n')
    # 过滤空段落后计算实际字符数（不包含换行符）
    paragraph_lengths = [len(p) for p in paragraphs if p]
    N = sum(paragraph_lengths)  # 总字符数（不含换行符）
    num_paragraphs = len(paragraph_lengths)  # 实际段落数
    
    if N == 0:
        return constants.DEFAULT_FONT_SIZE
    
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
                chars_per_line = max(1, int(W / avg_char_width)) if avg_char_width > 0 else N
                # 考虑换行符：每个段落至少占一行
                lines_needed = 0
                for length in paragraph_lengths:
                    if length > 0:
                        lines_needed += math.ceil(length / chars_per_line)
                    else:
                        lines_needed += 1  # 空段落也占一行
                # 至少需要 num_paragraphs 行（用户手动换行）
                lines_needed = max(lines_needed, num_paragraphs)
                total_height_needed = lines_needed * mid * l_h
                fits = total_height_needed <= H
            else: # vertical
                chars_per_column = max(1, int(H / avg_char_height)) if avg_char_height > 0 else N
                # 考虑换行符：每个段落至少占一列
                columns_needed = 0
                for length in paragraph_lengths:
                    if length > 0:
                        columns_needed += math.ceil(length / chars_per_column)
                    else:
                        columns_needed += 1  # 空段落也占一列
                # 至少需要 num_paragraphs 列（用户手动换行）
                columns_needed = max(columns_needed, num_paragraphs)
                total_width_needed = columns_needed * mid * l_h
                fits = total_width_needed <= W

            if fits:
                best_size = mid
                low = mid + 1
            else:
                high = mid - 1

        except Exception as e:
            if is_memory_allocation_error(e):
                raise
            logger.error(f"计算字号 {mid} 时出错: {e}", exc_info=True)
            high = mid - 1

    result = max(min_size, best_size)
    logger.info(f"自动计算的最佳字体大小: {result}px (范围: {min_size}-{max_size})")
    return result


def render_horizontal_block(content: str, font,
                           fill, stroke_enabled: bool, stroke_color, stroke_width: int,
                           canvas_image: Image.Image, current_x_col: int, current_y: int,
                           line_width: int, line_height_unit: int = None) -> int:
    """
    在竖排文本中渲染横排块（<H></H> 标签内的内容）
    
    使用"单位"系统：
    - 以中文字符高度为一个"单位"
    - 计算英文块需要多少个单位
    - 分配整数单位的空间
    - 在分配空间内垂直居中显示
    
    渲染规则：
    - 2个字符：横排显示
    - 3个及以上字符：竖排显示但每个字符旋转90度
    
    Args:
        line_height_unit: 一个"单位"的高度（中文字符高度），默认为字体大小 + 1
        
    Returns:
        占用的总高度（整数个单位）
    """
    font_size = font.size
    if not content or canvas_image is None:
        return font_size
    
    # 单位高度（中文字符高度）
    if line_height_unit is None:
        line_height_unit = font_size + 1
    
    # 全角→半角转换
    content = content.replace('！', '!').replace('？', '?')
    
    # 创建临时画布渲染横排内容
    h_width = int(font_size * len(content) * 1.5) + font_size * 2
    h_height = font_size * 3
    
    temp_img = Image.new('RGBA', (h_width, h_height), (0, 0, 0, 0))
    cropped = None
    rotated = None
    final_block = None
    try:
        temp_draw = ImageDraw.Draw(temp_img)

        # 横排渲染每个字符
        pen_x = font_size // 2
        pen_y = font_size

        text_params = {"font": font, "fill": fill}
        if stroke_enabled:
            text_params["stroke_width"] = int(stroke_width)
            text_params["stroke_fill"] = stroke_color

        for char in content:
            cdpt, _ = CJK_Compatibility_Forms_translate(char, 0)
            bbox = font.getbbox(cdpt)
            char_width = bbox[2] - bbox[0]
            temp_draw.text((pen_x, pen_y), cdpt, **text_params)
            pen_x += char_width

        temp_arr = np.array(temp_img)
        alpha = temp_arr[:, :, 3]
        non_zero = np.where(alpha > 0)
        if len(non_zero[0]) == 0:
            raise RuntimeError("横排文字块没有生成任何可见像素")

        min_y, max_y = non_zero[0].min(), non_zero[0].max()
        min_x, max_x = non_zero[1].min(), non_zero[1].max()
        cropped = temp_img.crop((min_x, min_y, max_x + 1, max_y + 1))

        if len(content) >= 3:
            rotated = cropped.rotate(
                -90,
                expand=True,
                resample=Image.Resampling.BICUBIC,
            )
            rotated_arr = np.array(rotated)
            rotated_alpha = rotated_arr[:, :, 3]
            rot_non_zero = np.where(rotated_alpha > 10)
            if len(rot_non_zero[0]) > 0:
                rot_min_y, rot_max_y = (
                    rot_non_zero[0].min(),
                    rot_non_zero[0].max(),
                )
                rot_min_x, rot_max_x = (
                    rot_non_zero[1].min(),
                    rot_non_zero[1].max(),
                )
                final_block = rotated.crop(
                    (rot_min_x, rot_min_y, rot_max_x + 1, rot_max_y + 1)
                )
            else:
                final_block = rotated
        else:
            final_block = cropped

        block_width, block_height = final_block.size
        units_needed = max(1, math.ceil(block_height / line_height_unit))
        allocated_height = units_needed * line_height_unit
        _, ref_ink_offset_y = get_char_ink_offset('我', font)
        cjk_ink_center_in_unit = line_height_unit / 2 + ref_ink_offset_y
        mid_unit_index = (units_needed - 1) / 2
        visual_center = (
            mid_unit_index * line_height_unit + cjk_ink_center_in_unit
        )
        vertical_offset = visual_center - block_height / 2
        paste_x = int(
            current_x_col - line_width + (line_width - block_width) / 2
        )
        paste_y = int(current_y + vertical_offset)
        _paste_with_alpha(canvas_image, final_block, paste_x, paste_y)
        return allocated_height
    finally:
        _close_images(final_block, rotated, cropped, temp_img)


def render_ellipsis_block(content: str, font,
                          fill, stroke_enabled: bool, stroke_color,
                          stroke_width, canvas_image: Optional[Image.Image],
                          current_x_col: int, current_y: float,
                          line_width: int, line_height_unit: int) -> int:
    """
    渲染"线性延展"标点的连续段：把 N 个横向字符合成一条横带，整体旋转 90°
    后作为 N 个单元格高度的连续竖向笔迹。

    适用场景：连续的省略号 `……`、破折号 `——`、波浪 `～～`（由预处理统一打包
    成 <E>...</E>）。相比逐字符旋转再逐格放置，一次性旋转可以保留字体对字符
    内部间距/形状的设计，并消除相邻字符间 line_height_approx 大于墨水高度
    造成的拼接缝（实测破折号独立渲染会露 10px 空隙）。

    Args:
        content: 连续同类字符串（长度 >= 1，通常 >= 2 才会走此路径）
        line_width: 当前列宽（用于水平居中）
        line_height_unit: 单元格高度（与普通字符一致）

    Returns:
        占用的总像素高度 = len(content) * line_height_unit
    """
    if not content or canvas_image is None:
        return len(content) * line_height_unit

    allocated_height = len(content) * line_height_unit

    # 1) 测量横向整串的墨水 bbox
    full_bbox = font.getbbox(content)
    content_w = max(1, full_bbox[2] - full_bbox[0])
    content_h = max(1, full_bbox[3] - full_bbox[1])

    # 2) 创建临时 RGBA 画布；留 padding 容纳描边与抗锯齿
    padding = max(10, int(stroke_width * 2) if stroke_enabled else 10)
    temp_w = content_w + padding * 2
    temp_h = content_h + padding * 2
    temp_img = Image.new('RGBA', (temp_w, temp_h), (0, 0, 0, 0))
    rotated = None
    cropped = None
    try:
        temp_draw = ImageDraw.Draw(temp_img)

        # 3) 绘制横向串；把 bbox 的左上角对齐到 (padding, padding)
        text_params = {'font': font, 'fill': fill}
        if stroke_enabled:
            text_params['stroke_width'] = int(stroke_width)
            text_params['stroke_fill'] = stroke_color
        draw_x = padding - full_bbox[0]
        draw_y = padding - full_bbox[1]
        temp_draw.text((draw_x, draw_y), content, **text_params)

        # 4) 整体旋转 90° 顺时针
        rotated = temp_img.rotate(
            -90,
            expand=True,
            resample=Image.Resampling.BICUBIC,
        )

        # 5) 按 alpha 裁剪到实际墨迹；裁剪失败时保留完整旋转图。
        try:
            rotated_arr = np.array(rotated)
            alpha = rotated_arr[:, :, 3]
            non_zero = np.where(alpha > 10)
            if len(non_zero[0]) > 0:
                min_y, max_y = non_zero[0].min(), non_zero[0].max()
                min_x, max_x = non_zero[1].min(), non_zero[1].max()
                cropped = rotated.crop(
                    (min_x, min_y, max_x + 1, max_y + 1)
                )
            else:
                cropped = rotated
        except Exception as error:
            if is_memory_allocation_error(error):
                raise
            cropped = rotated

        actual_w, actual_h = cropped.size
        _, ref_ink_offset_y = get_char_ink_offset('我', font)
        cjk_ink_center_in_unit = line_height_unit / 2 + ref_ink_offset_y
        units_needed = max(1, len(content))
        mid_unit_index = (units_needed - 1) / 2
        visual_center = (
            mid_unit_index * line_height_unit + cjk_ink_center_in_unit
        )
        paste_x = int(
            (current_x_col - line_width) + (line_width - actual_w) / 2.0
        )
        paste_y = int(current_y + visual_center - actual_h / 2.0)
        _paste_with_alpha(canvas_image, cropped, paste_x, paste_y)
        return allocated_height
    finally:
        _close_images(cropped, rotated, temp_img)


def _paste_with_alpha(canvas: Image.Image, overlay: Image.Image, x: int, y: int):
    """
    将带透明通道的图像正确粘贴到画布上
    """
    converted = overlay if overlay.mode == 'RGBA' else overlay.convert('RGBA')
    try:
        # 获取画布尺寸
        canvas_w, canvas_h = canvas.size
        overlay_w, overlay_h = converted.size
        
        # 边界检查
        if x >= canvas_w or y >= canvas_h or x + overlay_w <= 0 or y + overlay_h <= 0:
            return
        
        # 计算有效粘贴区域
        src_x1 = max(0, -x)
        src_y1 = max(0, -y)
        src_x2 = min(overlay_w, canvas_w - x)
        src_y2 = min(overlay_h, canvas_h - y)
        
        dst_x = max(0, x)
        dst_y = max(0, y)
        
        if src_x2 <= src_x1 or src_y2 <= src_y1:
            return
        
        # 裁剪 overlay 到有效区域
        cropped_overlay = converted.crop((src_x1, src_y1, src_x2, src_y2))
        try:
            if canvas.mode == 'RGBA':
                target_region = canvas.crop(
                    (
                        dst_x,
                        dst_y,
                        dst_x + cropped_overlay.width,
                        dst_y + cropped_overlay.height,
                    )
                )
                try:
                    composited = Image.alpha_composite(
                        target_region,
                        cropped_overlay,
                    )
                    try:
                        canvas.paste(composited, (dst_x, dst_y))
                    finally:
                        composited.close()
                finally:
                    target_region.close()
            else:
                canvas.paste(cropped_overlay, (dst_x, dst_y), cropped_overlay)
        finally:
            cropped_overlay.close()
    finally:
        if converted is not overlay:
            converted.close()

# --- 竖排文本绘制函数（支持单字符旋转）---
def draw_multiline_text_vertical(draw, text, font, x, y, max_height,
                                 fill=constants.DEFAULT_TEXT_COLOR,
                                 stroke_enabled=constants.DEFAULT_STROKE_ENABLED,
                                 stroke_color=constants.DEFAULT_STROKE_COLOR,
                                 stroke_width=constants.DEFAULT_STROKE_WIDTH,
                                 font_family_path=constants.DEFAULT_FONT_RELATIVE_PATH,
                                 line_spacing=constants.DEFAULT_LINE_SPACING,
                                 text_align=constants.DEFAULT_TEXT_ALIGN):
    """
    在指定位置绘制竖排多行文本。

    Word 逻辑：第一列贴气泡右边界开始，向左换列；每列按 text_align 独立做
    顶/中/底对齐（基于该列实际高度 line_heights[line_idx]）。

    关键特性：
    1. 逐字符调用 CJK_Compatibility_Forms_translate 进行标点转换
    2. 支持单字符旋转（如日文长音符号 ー 需要旋转90度）
    3. 气泡级别的旋转在 render_bubbles_unified 中统一处理

    Args:
        x: 气泡右边界（第一列贴该边界）
        y: 气泡顶边界（列内字符起点的参考原点）
        max_height: 气泡高度，限制每列最多能放多少字符
        line_spacing: 列间距倍数（影响 column_width_approx）
        text_align: 列内字符垂直对齐方式 'start'=顶 | 'center'=中 | 'end'=底
    """
    if not text:
        return
    
    # 预处理文本（省略号等）
    text = process_text_for_vertical(text)
    
    # 获取绘制的 Image 对象（用于单字符旋转时创建临时图像）
    # draw 对象是 ImageDraw.Draw，其 _image 属性指向原始 Image
    canvas_image = None
    if hasattr(draw, '_image'):
        canvas_image = draw._image

    lines = []
    line_heights = []  # 每列的实际高度（字符/横排块占用的垂直像素），与 lines 一一对应
    current_line = ""
    current_column_height = 0
    line_height_approx = font.size + 1  # 同一列内字符之间的字间距（不受 line_spacing 影响）

    # ===== 处理 <H></H> 标签的智能换行 =====
    # 先按 \n 分割段落，然后在每个段落内处理
    paragraphs = text.split('\n')

    for para_idx, paragraph in enumerate(paragraphs):
        # 非第一个段落时，先换列（实现回车换行效果）
        # 在竖排模式下，回车符(\n)应该对应新的一列
        if para_idx > 0:
            if current_line:
                lines.append(current_line)
                line_heights.append(current_column_height)
                current_line = ""
                current_column_height = 0

        if not paragraph:
            # 空段落，跳过（换列已在上面处理）
            continue

        # 分割段落为普通文本、横排块和省略号块
        parts = re.split(r'(<H>.*?</H>|<E>.*?</E>)', paragraph, flags=re.IGNORECASE | re.DOTALL)

        for part in parts:
            if not part:
                continue

            is_h_block = part.lower().startswith('<h>') and part.lower().endswith('</h>')
            is_e_block = part.lower().startswith('<e>') and part.lower().endswith('</e>')

            if is_h_block:
                # 横排块：计算其高度并作为整体处理
                content = part[3:-4]  # 去除 <H> 和 </H>
                if not content:
                    continue

                # 估算横排块的高度（使用单位系统）
                if len(content) >= 3:
                    # 3+ 字符旋转后变成竖排，高度 = 字符宽度之和
                    raw_height = sum(font.getbbox(c)[2] - font.getbbox(c)[0] for c in content)
                else:
                    # 2 字符横排，高度 = 字体高度
                    raw_height = font.size

                # 按单位计算占用高度（向上取整）
                units_needed = math.ceil(raw_height / line_height_approx)
                units_needed = max(1, units_needed)
                block_height = units_needed * line_height_approx

                # 检查是否能放入当前列
                if current_column_height + block_height <= max_height:
                    current_line += part  # 保持标签完整
                    current_column_height += block_height
                else:
                    # 需要换列
                    if current_line:
                        lines.append(current_line)
                        line_heights.append(current_column_height)
                    current_line = part
                    current_column_height = block_height
            elif is_e_block:
                # 省略号块：每个 … / ⋯ 占 1 个单元格高度，整段原子放置
                content = part[3:-4]  # 去除 <E> 和 </E>
                if not content:
                    continue
                block_height = len(content) * line_height_approx

                if current_column_height + block_height <= max_height:
                    current_line += part
                    current_column_height += block_height
                else:
                    if current_line:
                        lines.append(current_line)
                        line_heights.append(current_column_height)
                    current_line = part
                    current_column_height = block_height
            else:
                # 普通文本：逐字符处理
                for char in part:
                    if current_column_height + line_height_approx <= max_height:
                        current_line += char
                        current_column_height += line_height_approx
                    else:
                        lines.append(current_line)
                        line_heights.append(current_column_height)
                        current_line = char
                        current_column_height = line_height_approx

    # 添加最后一行
    if current_line:
        lines.append(current_line)
        line_heights.append(current_column_height)

    # 列宽基于字体大小估算（列间距受 line_spacing 影响：竖排的"行间距"即两列之间的距离）
    column_width_approx = int(font.size * line_spacing) + 3

    # Word 逻辑：水平方向始终从气泡右边开始（第一列贴气泡右边界，不再整体居中）
    # 传入的 x 是气泡右边界，current_x_col 表示当前列的右边界
    current_x_base = x

    # 垂直方向每列独立对齐（在主循环内按 line_heights[line_idx] 计算）
    # 预加载NotoSans字体，用于特殊字符
    special_font = None
    font_size = font.size

    # ===== 预计算每列的实际最大字符宽度 =====
    # 计算每列的最大字符实际宽度，用于精确居中对齐。
    line_max_widths = []
    for line in lines:
        max_char_width = font_size  # 默认使用 font_size
        for char in line:
            converted_char, _ = CJK_Compatibility_Forms_translate(char, 1)
            # 确定使用哪个字体
            actual_font = font
            if FREETYPE_AVAILABLE and not font_supports_char(font_family_path, converted_char):
                for fallback_path in FALLBACK_FONTS:
                    if font_supports_char(fallback_path, converted_char):
                        actual_font = get_font(fallback_path, font_size)
                        break
            # 获取字符宽度
            try:
                bbox = actual_font.getbbox(converted_char)
                char_width = bbox[2] - bbox[0]
                if char_width > max_char_width:
                    max_char_width = char_width
            except Exception as error:
                if is_memory_allocation_error(error):
                    raise
        line_max_widths.append(max_char_width)

    current_x_col = current_x_base
    for line_idx, line in enumerate(lines):
        # 每列独立按 text_align 计算垂直起点（Word 逻辑）
        col_height = line_heights[line_idx] if line_idx < len(line_heights) else 0
        if col_height < max_height:
            if text_align == 'start':
                col_vertical_offset = 0
            elif text_align == 'end':
                col_vertical_offset = max_height - col_height
            else:
                col_vertical_offset = (max_height - col_height) / 2
        else:
            col_vertical_offset = 0
        current_y_char = y + col_vertical_offset
        # 获取当前列的实际宽度
        line_width = line_max_widths[line_idx] if line_idx < len(line_max_widths) else font_size
        
        # ===== 分割行内容为普通文本、横排块和省略号块 =====
        parts = re.split(r'(<H>.*?</H>|<E>.*?</E>)', line, flags=re.IGNORECASE | re.DOTALL)

        for part in parts:
            if not part:
                continue

            is_horizontal_block = part.lower().startswith('<h>') and part.lower().endswith('</h>')
            is_ellipsis_block = part.lower().startswith('<e>') and part.lower().endswith('</e>')

            if is_horizontal_block:
                # ===== 渲染横排块 =====
                content = part[3:-4]  # 去除 <H> 和 </H>
                if content:
                    block_height = render_horizontal_block(
                        content=content,
                        font=font,
                        fill=fill,
                        stroke_enabled=stroke_enabled,
                        stroke_color=stroke_color,
                        stroke_width=stroke_width,
                        canvas_image=canvas_image,
                        current_x_col=current_x_col,
                        current_y=current_y_char,
                        line_width=line_width,
                        line_height_unit=line_height_approx  # 传递单位高度
                    )
                    current_y_char += block_height
            elif is_ellipsis_block:
                # ===== 渲染连续省略号块（整段旋转，消除拼接缝）=====
                content = part[3:-4]  # 去除 <E> 和 </E>
                if content:
                    block_height = render_ellipsis_block(
                        content=content,
                        font=font,
                        fill=fill,
                        stroke_enabled=stroke_enabled,
                        stroke_color=stroke_color,
                        stroke_width=stroke_width,
                        canvas_image=canvas_image,
                        current_x_col=current_x_col,
                        current_y=current_y_char,
                        line_width=line_width,
                        line_height_unit=line_height_approx,
                    )
                    current_y_char += block_height
            else:
                # ===== 渲染普通竖排字符 =====
                for char in part:
                    # 调用 CJK_Compatibility_Forms_translate 获取转换后的字符和旋转角度
                    converted_char, rot_degree = CJK_Compatibility_Forms_translate(char, 1)  # 1 = 竖排

                    if converted_char in SINGLE_LINEAR_BLOCK_CHARS and canvas_image is not None:
                        block_height = render_ellipsis_block(
                            content=converted_char,
                            font=font,
                            fill=fill,
                            stroke_enabled=stroke_enabled,
                            stroke_color=stroke_color,
                            stroke_width=stroke_width,
                            canvas_image=canvas_image,
                            current_x_col=current_x_col,
                            current_y=current_y_char,
                            line_width=line_width,
                            line_height_unit=line_height_approx,
                        )
                        current_y_char += block_height
                        continue

                    if (
                        rot_degree == 0
                        and converted_char in COMBINED_UPRIGHT_SYMBOL_EXPANSIONS
                        and canvas_image is not None
                    ):
                        block_height = render_combined_upright_symbol(
                            converted_char,
                            font,
                            fill,
                            stroke_enabled,
                            stroke_color,
                            stroke_width,
                            canvas_image,
                            current_x_col,
                            current_y_char,
                            line_width,
                            line_height_approx,
                        )
                        current_y_char += block_height
                        continue
                    
                    # ===== 使用字体回退系统 =====
                    current_font = font
                    
                    if FREETYPE_AVAILABLE:
                        if not font_supports_char(font_family_path, converted_char):
                            for fallback_path in FALLBACK_FONTS:
                                if font_supports_char(fallback_path, converted_char):
                                    try:
                                        current_font = get_font(fallback_path, font_size)
                                        logger.debug(f"字符 '{converted_char}' 使用回退字体: {os.path.basename(fallback_path)}")
                                        break
                                    except Exception as e:
                                        if is_memory_allocation_error(e):
                                            raise
                                        logger.warning(f"回退字体加载失败: {fallback_path} - {e}")
                                        continue
                    else:
                        if converted_char in SPECIAL_CHARS:
                            if special_font is None:
                                try:
                                    special_font = get_font(NOTOSANS_FONT_PATH, font_size)
                                except Exception as e:
                                    if is_memory_allocation_error(e):
                                        raise
                                    logger.error(f"加载NotoSans字体失败: {e}，回退到普通字体")
                                    special_font = font
                            if special_font is not None:
                                current_font = special_font
                    
                    # 准备绘制参数
                    text_draw_params = {
                        "font": current_font,
                        "fill": fill
                    }
                    if stroke_enabled:
                        text_draw_params["stroke_width"] = int(stroke_width)
                        text_draw_params["stroke_fill"] = stroke_color
                    
                    # 获取字符尺寸
                    bbox = current_font.getbbox(converted_char)
                    char_width = bbox[2] - bbox[0]
                    char_height = bbox[3] - bbox[1]
                    
                    if rot_degree != 0 and canvas_image is not None:
                        # ===== 需要旋转的字符 =====
                        # 创建临时图像用于旋转（尺寸足够容纳旋转后的字符）
                        # 对于90度旋转，宽高会互换，所以需要足够的空间
                        diagonal = int(math.ceil(math.sqrt(char_width**2 + char_height**2)))
                        padding = max(10, int(stroke_width * 2) if stroke_enabled else 0)
                        temp_size = diagonal + padding * 2
                        temp_size = int(temp_size)
                        
                        temp_img = Image.new(
                            'RGBA',
                            (temp_size, temp_size),
                            (0, 0, 0, 0),
                        )
                        rotated_img = None
                        cropped_rotated = None
                        rgb_rotated = None
                        try:
                            temp_draw = ImageDraw.Draw(temp_img)
                            temp_x = (temp_size - char_width) // 2
                            temp_y = (temp_size - char_height) // 2
                            temp_text_params = {
                                "font": current_font,
                                "fill": fill,
                            }
                            if stroke_enabled:
                                temp_text_params["stroke_width"] = int(
                                    stroke_width
                                )
                                temp_text_params["stroke_fill"] = stroke_color

                            temp_draw.text(
                                (temp_x, temp_y),
                                converted_char,
                                **temp_text_params,
                            )
                            rotated_img = temp_img.rotate(
                                -rot_degree,
                                resample=Image.Resampling.BICUBIC,
                                expand=False,
                            )
                            rotated_arr = np.array(rotated_img)
                            alpha_channel = rotated_arr[:, :, 3]
                            non_zero = np.where(alpha_channel > 10)
                            if len(non_zero[0]) > 0:
                                min_y, max_y = (
                                    non_zero[0].min(),
                                    non_zero[0].max(),
                                )
                                min_x, max_x = (
                                    non_zero[1].min(),
                                    non_zero[1].max(),
                                )
                                cropped_rotated = rotated_img.crop(
                                    (min_x, min_y, max_x + 1, max_y + 1)
                                )
                            else:
                                cropped_rotated = rotated_img

                            actual_width, actual_height = cropped_rotated.size
                            paste_x = int(
                                (current_x_col - line_width)
                                + (line_width - actual_width) / 2.0
                            )
                            paste_y = int(
                                current_y_char
                                + (line_height_approx - actual_height) / 2.0
                            )
                            try:
                                if canvas_image.mode == 'RGBA':
                                    canvas_image.paste(
                                        cropped_rotated,
                                        (paste_x, paste_y),
                                        cropped_rotated,
                                    )
                                else:
                                    rgb_rotated = cropped_rotated.convert('RGB')
                                    canvas_image.paste(
                                        rgb_rotated,
                                        (paste_x, paste_y),
                                        cropped_rotated,
                                    )
                            except Exception as e:
                                if is_memory_allocation_error(e):
                                    raise
                                logger.warning(
                                    f"旋转字符粘贴失败: {e}，回退到直接绘制"
                                )
                                text_x_char = current_x_col - char_width
                                draw.text(
                                    (text_x_char, current_y_char),
                                    converted_char,
                                    **text_draw_params,
                                )
                        finally:
                            _close_images(
                                rgb_rotated,
                                cropped_rotated,
                                rotated_img,
                                temp_img,
                            )
                    else:
                        # ===== 常规绘制（不需要旋转） =====
                        # ===== 水平居中计算 =====
                        # 计算字符在当前列中的水平居中位置，line_width 为该列的最大字符宽度。
                        # 使用预计算的 line_width（该列实际最大字符宽度）
                        text_x_char = (current_x_col - line_width) + round((line_width - char_width) / 2.0)
                        text_y_char = current_y_char

                        # ===== 墨水偏移校正（水平）=====
                        # Pillow 的 getbbox() 返回的边界框可能不等于实际墨水区域
                        # 反向补偿水平墨水偏移以实现真正的视觉居中。
                        ink_offset_x, _ = get_char_ink_offset(converted_char, current_font)
                        text_x_char -= ink_offset_x

                        # ===== 低位直立标点的垂直校正 =====
                        # 仅对句读点执行“往格子中心拉回”的校正；感叹号/问号等保持原始
                        # baseline 位置，避免某些字体的 bbox 度量把它们错误上提到上一格。
                        text_y_char = get_vertical_center_adjusted_y(
                            converted_char,
                            current_font,
                            current_y_char,
                            line_height_approx,
                        )

                        # 直接绘制
                        draw.text((text_x_char, text_y_char), converted_char, **text_draw_params)
                    
                    current_y_char += line_height_approx
        
        current_x_col -= column_width_approx

# --- 横排文本绘制函数（不含旋转，旋转在 render_bubbles_unified 中统一处理） ---
def draw_multiline_text_horizontal(draw, text, font, x, y, max_width,
                                  fill=constants.DEFAULT_TEXT_COLOR,
                                  stroke_enabled=constants.DEFAULT_STROKE_ENABLED,
                                  stroke_color=constants.DEFAULT_STROKE_COLOR,
                                  stroke_width=constants.DEFAULT_STROKE_WIDTH,
                                  bubble_width=None,
                                  font_family_path=constants.DEFAULT_FONT_RELATIVE_PATH,
                                  line_spacing=constants.DEFAULT_LINE_SPACING,
                                  text_align=constants.DEFAULT_TEXT_ALIGN):
    """
    在指定位置绘制横排多行文本（不含旋转）。
    旋转逻辑由 render_bubbles_unified 统一处理，使用外接圆方案优化性能。
    
    优化：一次遍历同时完成分行和记录字符宽度，避免重复调用 getbbox()。
    
    Args:
        bubble_width: 气泡宽度，用于按 text_align 计算每行水平对齐偏移（start=左/center=中/end=右）
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

    line_height = int(font.size * line_spacing) + 5

    # 计算每行的总宽度（直接使用已记录的值，不再遍历）
    line_widths = [sum(widths) for widths in line_char_widths]

    # Word 逻辑：垂直方向始终从气泡顶部开始（不居中，text_align 只管水平方向）
    current_y = y

    # 预加载NotoSans字体，用于特殊字符
    special_font = None
    font_size = font.size

    for line_idx, line in enumerate(lines):
        # 水平对齐偏移（start=左, center=中, end=右）
        if bubble_width is not None:
            if text_align == 'start':
                horizontal_offset = 0
            elif text_align == 'end':
                horizontal_offset = bubble_width - line_widths[line_idx]
            else:
                horizontal_offset = (bubble_width - line_widths[line_idx]) / 2
            current_x = x + horizontal_offset
        else:
            current_x = x
        
        char_widths = line_char_widths[line_idx]
        for char_idx, char in enumerate(line):
            # ===== 使用字体回退系统 =====
            current_font = font
            char_width = char_widths[char_idx]  # 使用缓存的宽度
            
            if FREETYPE_AVAILABLE:
                # 使用 FreeType 检查字体是否支持该字符
                if not font_supports_char(font_family_path, char):
                    # 主字体不支持，遍历回退字体列表
                    for fallback_path in FALLBACK_FONTS:
                        if font_supports_char(fallback_path, char):
                            try:
                                current_font = get_font(fallback_path, font_size)
                                # 使用回退字体时需要重新计算宽度
                                bbox = current_font.getbbox(char)
                                char_width = bbox[2] - bbox[0]
                                logger.debug(f"字符 '{char}' 使用回退字体: {os.path.basename(fallback_path)}")
                                break
                            except Exception as e:
                                if is_memory_allocation_error(e):
                                    raise
                                logger.warning(f"回退字体加载失败: {fallback_path} - {e}")
                                continue
            else:
                # FreeType 不可用时，回退到使用 SPECIAL_CHARS 检查
                if char in SPECIAL_CHARS:
                    if special_font is None:
                        try:
                            special_font = get_font(NOTOSANS_FONT_PATH, font_size)
                        except Exception as e:
                            if is_memory_allocation_error(e):
                                raise
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
            raise RuntimeError(
                f"气泡 {i}: 无法加载字体 {state.font_family}"
            )
        
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
                
                temp_img = Image.new(
                    'RGBA',
                    (temp_size, temp_size),
                    (0, 0, 0, 0),
                )
                rotated_img = None
                try:
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
                            font_family_path=state.font_family,
                            line_spacing=state.line_spacing,
                            text_align=state.text_align
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
                            font_family_path=state.font_family,
                            line_spacing=state.line_spacing,
                            text_align=state.text_align
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
                finally:
                    _close_images(rotated_img, temp_img)
                
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
                        font_family_path=state.font_family,
                        line_spacing=state.line_spacing,
                        text_align=state.text_align
                    )
                else:
                    draw_multiline_text_horizontal(
                        draw, text, font, draw_x, draw_y, max_text_width,
                        fill=state.text_color,
                        stroke_enabled=state.stroke_enabled,
                        stroke_color=state.stroke_color,
                        stroke_width=state.stroke_width,
                        bubble_width=max_text_width,
                        font_family_path=state.font_family,
                        line_spacing=state.line_spacing,
                        text_align=state.text_align
                    )
                    
        except Exception as render_e:
            logger.error(f"渲染气泡 {i} 时出错: {render_e}", exc_info=True)
            raise
    
    logger.info("[统一渲染] 所有气泡文本渲染完成。")
    return image
