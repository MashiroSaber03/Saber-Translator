import logging
import math
import os
import re
import functools
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING
from PIL import Image, ImageDraw, ImageFont
import cv2 # 导入 cv2 备用
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
from src.shared.path_helpers import resource_path, get_debug_dir, get_font_path # 导入路径助手函数

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
FONTS_DIR = os.path.join('src', 'app', 'static', 'fonts')
DEFAULT_FONT_PATH = constants.DEFAULT_FONT_RELATIVE_PATH

# --- 回退字体列表 ---
FALLBACK_FONTS = [
    os.path.join(FONTS_DIR, 'Arial_Unicode.ttf'),   # Unicode 全字符字体
    os.path.join(FONTS_DIR, 'msyh.ttc'),            # 微软雅黑
    os.path.join(FONTS_DIR, 'msgothic.ttc'),        # MS Gothic（日文哥特体）
]

# --- 特殊字符的字体路径（Pillow 回退用）---
NOTOSANS_FONT_PATH = os.path.join(FONTS_DIR, 'NotoSans-Medium.ttf')

# --- 需要使用特殊字体渲染的字符（Pillow 回退用）---
SPECIAL_CHARS = {'‼', '⁉', '︕', '︖', '︙', '⋮', '⋯'}

# =============================================================================
# 竖排标点符号处理系统
# =============================================================================

# --- CJK 横转竖标点符号映射表 ---
CJK_H2V = {
    # === 基础标点符号 ===
    "‥": "︰",      # 两点省略号
    "—": "︱",      # 破折号
    "―": "|",       # 水平线
    "–": "︲",      # 短破折号
    "_": "︴",      # 下划线
    
    # === 括号类 - 基础 ===
    "(": "︵",  ")": "︶",      # 英文圆括号
    "（": "︵", "）": "︶",     # 中文圆括号
    "{": "︷",  "}": "︸",      # 花括号
    "〔": "︹", "〕": "︺",     # 龟甲括号
    "【": "︻", "】": "︼",     # 方头括号
    "《": "︽", "》": "︾",     # 书名号
    "〈": "︿", "〉": "﹀",     # 单尖括号
    "[": "﹇",  "]": "﹈",      # 英文方括号
    
    # === 括号类 - 扩展 Unicode ===
    "⟨": "︿", "⟩": "﹀",       # 数学尖括号
    "⟪": "︿", "⟫": "﹀",       # 数学双尖括号
    "⦅": "︵", "⦆": "︶",       # 全角括号变体
    "❨": "︵", "❩": "︶",       # 装饰圆括号
    "❪": "︷", "❫": "︸",       # 装饰花括号
    "❬": "﹇", "❭": "﹈",       # 装饰方括号
    "❮": "︿", "❯": "﹀",       # 装饰尖括号
    
    # === 引号类 ===
    "「": "﹁", "」": "﹂",     # 日式单引号
    "『": "﹃", "』": "﹄",     # 日式双引号
    "﹑": "﹅",                 # 顿号变体
    "﹆": "﹆",                 # 保持不变
    '"': "﹁", '"': "﹂",       # 弯双引号
    "'": "﹁", "'": "﹂",       # 弯单引号
    "″": "﹂", "‴": "﹂",       # Prime 符号
    "‶": "﹁", "ⷷ": "﹁",       # Prime 变体
    
    # === 装饰线 ===
    "﹉": "﹉", "﹊": "﹊",     # 虚线
    "﹋": "﹋", "﹌": "﹌",     # 虚线
    "﹍": "﹍", "﹎": "﹎",     # 虚线
    "﹏": "﹏",                 # 波浪线
    
    # === 省略号类 ===
    "…": "⋮",      # 水平省略号 → 竖向三点
    "⋯": "︙",     # 居中省略号 → 竖向省略号
    "⋰": "⋮",     # 对角省略号 → 竖向三点
    "⋱": "⋮",     # 对角省略号 → 竖向三点
    
    # === 波浪线类 ===
    "~": "︴",      # ASCII 波浪线
    "〜": "︴",     # 日文波浪线
    "～": "︴",     # 全角波浪线
    "〰": "︴",     # 波浪破折号
    
    # === 感叹问号类（重要！专用竖排变体）===
    "!": "︕",      # 英文感叹号
    "?": "︖",      # 英文问号
    "！": "︕",     # 中文全角感叹号
    "？": "︖",     # 中文全角问号
    "؟": "︖",     # 阿拉伯问号
    "¿": "︖",     # 西班牙倒问号
    "¡": "︕",     # 西班牙倒感叹号
    
    # === 句点类 ===
    ".": "︒",      # 英文句点
    "。": "︒",     # 中文句号
    
    # === 分隔符类 ===
    ";": "︔",  "；": "︔",     # 分号
    ":": "︓",  "：": "︓",     # 冒号
    ",": "︐",  "，": "︐",     # 逗号
    "‚": "︐",  "„": "︐",      # 低引号
    "-": "︲",  "−": "︲",      # 连字符
    "・": "·",                  # 中黑点
}

# --- CJK 竖转横标点符号映射表 (反向映射) ---
CJK_V2H = {v: k for k, v in CJK_H2V.items()}

# --- 特殊组合标点映射 (保留用于组合符号处理) ---
SPECIAL_PUNCTUATION_PATTERNS = [
    ('...', '…'),      # 连续三个点先转为省略号
    ('..', '…'),       # 两个点也转为省略号
    ('!!!', '‼'),      # 连续三个感叹号映射成双感叹号
    ('!!', '‼'),       # 连续两个感叹号映射成双感叹号
    ('！！！', '‼'),   # 中文连续三个感叹号
    ('！！', '‼'),     # 中文连续两个感叹号
    ('!?', '⁉'),       # 感叹号加问号映射成感叹问号组合
    ('?!', '⁉'),       # 问号加感叹号映射成感叹问号组合
    ('！？', '⁉'),     # 中文感叹号加问号
    ('？！', '⁉'),     # 中文问号加感叹号
]

# --- 需要垂直居中校正的竖排标点符号 ---
# 这些是 CJK Compatibility Forms 的竖排标点，某些字体（如微软雅黑）对这些字符
# 的垂直位置处理不正确（偏上），需要在渲染时进行手动校正
VERTICAL_CENTER_PUNCTUATION = {
    # 竖排句读标点
    '︒', '︐', '︑', '︓', '︔', '︕', '︖', '︰',
    # 竖排括号
    '︵', '︶', '︷', '︸', '︹', '︺', '︻', '︼', '︽', '︾', '︿', '﹀',
    # 竖排引号
    '﹁', '﹂', '﹃', '﹄',
    # 竖排线类
    '︱', '︲', '︳', '︴',
    # 竖排省略号
    '︙', '⋮',
    # 其他竖排符号
    '﹅', '﹆', '﹇', '﹈',
    # 特殊组合标点（双感叹号、感叹问号等）
    '‼', '⁉',
}


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


def is_vertical_punctuation(ch: str) -> bool:
    """
    检查字符是否为竖排标点符号（需要垂直居中校正）
    
    某些字体（如微软雅黑）对 CJK Compatibility Forms 的竖排标点
    处理不正确，导致标点位置偏上。此函数用于识别这些需要校正的字符。
    
    Args:
        ch: 单个字符
        
    Returns:
        是否为需要垂直居中校正的竖排标点
    """
    return ch in VERTICAL_CENTER_PUNCTUATION


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
        logger.debug(f"获取字符 '{char}' 墨水偏移时出错: {e}")
        return (0.0, 0.0)


def compact_special_symbols(text: str) -> str:
    """
    预处理特殊符号
    
    处理逻辑：
    1. 替换半角省略号
    2. 合并连续的省略号为一个
    3. 将西文省略号(U+2026,贴底)替换为居中省略号(U+22EF)，解决横排省略号位置偏下的问题
    4. 删除标点符号后的空格
    
    Args:
        text: 原始文本
        
    Returns:
        处理后的文本
    """
    if not text:
        return text
    
    # 替换半角省略号
    text = text.replace('...', '…')
    text = text.replace('..', '…')
    
    # 合并连续的省略号为一个
    text = re.sub(r'…+', '…', text)
    
    # 将西文省略号(U+2026,贴底)替换为居中省略号(U+22EF)
    # 解决横排省略号位置偏下的问题
    text = text.replace('…', '⋯')
    
    # 删除标点符号后的空格
    # 只删除标点符号后的空格，不删除字母/数字后的空格
    pattern = r'([。，、！？；：…—～「」『』【】（）《》〈〉.,!?;:\-])[ 　]+'
    text = re.sub(pattern, r'\1', text)
    
    return text


def CJK_Compatibility_Forms_translate(cdpt: str, direction: int) -> Tuple[str, int]:
    """
    CJK兼容形式标点符号转换
    
    根据排版方向将标点符号转换为对应的形式。
    
    Args:
        cdpt: 单个字符
        direction: 排版方向，0 = 横排，1 = 竖排
        
    Returns:
        Tuple[str, int]: (转换后的字符, 旋转角度)
        旋转角度通常为 0，特殊情况（如日文长音符号）可能为 90
    """
    # 特殊处理：日文长音符号在竖排时需要旋转 90 度
    if cdpt == 'ー' and direction == 1:
        return 'ー', 90
    
    # 竖→横 转换
    if cdpt in CJK_V2H:
        if direction == 0:
            # 横排时，将竖排符号转为横排
            return CJK_V2H[cdpt], 0
        else:
            # 竖排时，保持不变
            return cdpt, 0
    
    # 横→竖 转换
    elif cdpt in CJK_H2V:
        if direction == 1:
            # 竖排时，将横排符号转为竖排
            return CJK_H2V[cdpt], 0
        else:
            # 横排时，保持不变
            return cdpt, 0
    
    return cdpt, 0


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
    
    注意：这个函数只做预处理，不做字符转换！
    字符转换（CJK_Compatibility_Forms_translate）在渲染函数中逐字符进行，
    这样才能正确处理需要旋转的字符（如日文长音符号 ー）。
    
    处理流程：
    1. 调用 compact_special_symbols 统一省略号格式
    2. 处理特殊组合标点（如 !! → ‼）
    3. 在竖排文本中，将省略号替换为竖排省略号符号
    4. 自动为英文/数字添加 <H> 横排标签
    
    Args:
        text: 原始文本
        
    Returns:
        预处理后的文本（尚未进行字符转换）
    """
    if not text:
        return text
    
    # 步骤1: 预处理特殊符号
    text = compact_special_symbols(text)
    
    # 步骤2: 处理特殊组合标点
    for pattern, replacement in SPECIAL_PUNCTUATION_PATTERNS:
        text = text.replace(pattern, replacement)
    
    # 步骤3: 在竖排文本中，将省略号替换为竖排省略号符号
    text = text.replace('…', '︙')
    text = text.replace('⋯', '︙')
    
    # 步骤4: 自动为英文/数字添加 <H> 横排标签
    text = auto_add_horizontal_tags(text)
    
    # 注意：不在此处进行字符转换！
    # 字符转换将在渲染函数 draw_multiline_text_vertical 中逐字符进行，
    # 这样才能正确获取和处理旋转角度。
    
    return text


def map_to_vertical_punctuation(text: str) -> str:
    """
    将文本中的标点符号映射为竖排标点符号
    
    这是对外公开的主要接口函数，内部调用 process_text_for_vertical。
    保持函数名不变以确保向后兼容。
    
    Args:
        text: 原始文本
        
    Returns:
        转换后的文本，标点符号已替换为竖排版本
    """
    return process_text_for_vertical(text)

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
            logger.error(f"计算字号 {mid} 时出错: {e}", exc_info=True)
            high = mid - 1

    result = max(min_size, best_size)
    logger.info(f"自动计算的最佳字体大小: {result}px (范围: {min_size}-{max_size})")
    return result


def render_horizontal_block(content: str, font, font_size: int, 
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
        line_height_unit: 一个"单位"的高度（中文字符高度），默认为 font_size + 1
        
    Returns:
        占用的总高度（整数个单位）
    """
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
    temp_draw = ImageDraw.Draw(temp_img)
    
    # 横排渲染每个字符
    pen_x = font_size // 2
    pen_y = font_size
    
    text_params = {"font": font, "fill": fill}
    if stroke_enabled:
        text_params["stroke_width"] = int(stroke_width)
        text_params["stroke_fill"] = stroke_color
    
    for char in content:
        # 横排模式，使用方向 0
        cdpt, _ = CJK_Compatibility_Forms_translate(char, 0)
        bbox = font.getbbox(cdpt)
        char_width = bbox[2] - bbox[0]
        
        temp_draw.text((pen_x, pen_y), cdpt, **text_params)
        pen_x += char_width
    
    # 获取实际渲染区域
    temp_arr = np.array(temp_img)
    alpha = temp_arr[:, :, 3]
    non_zero = np.where(alpha > 0)
    
    if len(non_zero[0]) == 0:
        return line_height_unit  # 返回一个单位
    
    min_y, max_y = non_zero[0].min(), non_zero[0].max()
    min_x, max_x = non_zero[1].min(), non_zero[1].max()
    
    # 裁剪有效区域
    cropped = temp_img.crop((min_x, min_y, max_x + 1, max_y + 1))
    
    # 根据字符数决定是否旋转
    if len(content) >= 3:
        # 3个及以上字符：旋转90度（顺时针）
        rotated = cropped.rotate(-90, expand=True, resample=Image.Resampling.BICUBIC)
        
        # 旋转后重新获取墨水边界（旋转可能会引入空白边缘）
        rotated_arr = np.array(rotated)
        rotated_alpha = rotated_arr[:, :, 3]
        rot_non_zero = np.where(rotated_alpha > 10)  # 阈值稍高以忽略抗锯齿产生的半透明像素
        
        if len(rot_non_zero[0]) > 0:
            rot_min_y, rot_max_y = rot_non_zero[0].min(), rot_non_zero[0].max()
            rot_min_x, rot_max_x = rot_non_zero[1].min(), rot_non_zero[1].max()
            final_block = rotated.crop((rot_min_x, rot_min_y, rot_max_x + 1, rot_max_y + 1))
        else:
            final_block = rotated
    else:
        # 2个字符：直接横排显示
        final_block = cropped
    
    block_width, block_height = final_block.size
    
    # ===== 单位系统计算 =====
    # 计算需要多少个"单位"才能容纳这个英文块
    units_needed = math.ceil(block_height / line_height_unit)
    units_needed = max(1, units_needed)  # 至少1个单位
    
    # 分配的总空间
    allocated_height = units_needed * line_height_unit
    
    # ===== 基于中文字符墨水中心的垂直居中 =====
    # 获取参考中文字符的墨水偏移
    _, ref_ink_offset_y = get_char_ink_offset('我', font)
    
    # 中文字符的墨水中心位置（相对于单元格顶部）
    # 对于一个单元格，中文的墨水中心 = 单元格高度/2 + 墨水偏移
    cjk_ink_center_in_unit = line_height_unit / 2 + ref_ink_offset_y
    
    # 对于分配的 N 个单位空间，计算"视觉中心"
    # 视觉中心 = 中间单位的墨水中心位置
    # 例如：3个单位，中间是第2个单位(索引1)，其墨水中心在 1*unit + cjk_ink_center_in_unit
    mid_unit_index = (units_needed - 1) / 2  # 0.5, 1, 1.5, 2, ...
    visual_center = mid_unit_index * line_height_unit + cjk_ink_center_in_unit
    
    # 英文块的墨水中心（相对于块顶部）
    block_ink_center = block_height / 2
    
    # 让英文块墨水中心对齐到视觉中心
    vertical_offset = visual_center - block_ink_center
    
    # 水平居中
    paste_x = int(current_x_col - line_width + (line_width - block_width) / 2)
    # 垂直位置
    paste_y = int(current_y + vertical_offset)
    
    # 使用正确的透明度混合
    _paste_with_alpha(canvas_image, final_block, paste_x, paste_y)
    
    # 返回分配的空间高度（整数个单位），确保后续文本位置正确
    return allocated_height


def _paste_with_alpha(canvas: Image.Image, overlay: Image.Image, x: int, y: int):
    """
    将带透明通道的图像正确粘贴到画布上
    """
    try:
        # 确保 overlay 是 RGBA
        if overlay.mode != 'RGBA':
            overlay = overlay.convert('RGBA')
        
        # 获取画布尺寸
        canvas_w, canvas_h = canvas.size
        overlay_w, overlay_h = overlay.size
        
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
        cropped_overlay = overlay.crop((src_x1, src_y1, src_x2, src_y2))
        
        if canvas.mode == 'RGBA':
            # RGBA 画布：使用 alpha_composite
            # 先提取目标区域
            target_region = canvas.crop((dst_x, dst_y, dst_x + cropped_overlay.width, dst_y + cropped_overlay.height))
            # 合成
            composited = Image.alpha_composite(target_region, cropped_overlay)
            # 粘贴回去
            canvas.paste(composited, (dst_x, dst_y))
        else:
            # RGB 画布：使用 overlay 的 alpha 作为遮罩
            canvas.paste(cropped_overlay, (dst_x, dst_y), cropped_overlay)
    except Exception as e:
        logger.warning(f"粘贴带透明度图像失败: {e}")

# --- 竖排文本绘制函数（支持单字符旋转）---
def draw_multiline_text_vertical(draw, text, font, x, y, max_height,
                                 fill=constants.DEFAULT_TEXT_COLOR,
                                 stroke_enabled=constants.DEFAULT_STROKE_ENABLED,
                                 stroke_color=constants.DEFAULT_STROKE_COLOR,
                                 stroke_width=constants.DEFAULT_STROKE_WIDTH,
                                 bubble_width=None,
                                 font_family_path=constants.DEFAULT_FONT_RELATIVE_PATH):
    """
    在指定位置绘制竖排多行文本。
    
    关键特性：
    1. 逐字符调用 CJK_Compatibility_Forms_translate 进行标点转换
    2. 支持单字符旋转（如日文长音符号 ー 需要旋转90度）
    3. 气泡级别的旋转在 render_all_bubbles 中统一处理
    """
    if not text:
        return
    
    # 预处理文本（省略号等）
    text = map_to_vertical_punctuation(text)
    
    # 获取绘制的 Image 对象（用于单字符旋转时创建临时图像）
    # draw 对象是 ImageDraw.Draw，其 _image 属性指向原始 Image
    canvas_image = None
    if hasattr(draw, '_image'):
        canvas_image = draw._image

    lines = []
    current_line = ""
    current_column_height = 0
    line_height_approx = font.size + 1  # 字间距为1像素

    # ===== 处理 <H></H> 标签的智能换行 =====
    # 先按 \n 分割段落，然后在每个段落内处理
    paragraphs = text.split('\n')
    
    for para_idx, paragraph in enumerate(paragraphs):
        # 非第一个段落时，先换列（实现回车换行效果）
        # 在竖排模式下，回车符(\n)应该对应新的一列
        if para_idx > 0:
            if current_line:
                lines.append(current_line)
                current_line = ""
                current_column_height = 0
        
        if not paragraph:
            # 空段落，跳过（换列已在上面处理）
            continue
        
        # 分割段落为普通文本和横排块
        parts = re.split(r'(<H>.*?</H>)', paragraph, flags=re.IGNORECASE | re.DOTALL)
        
        for part in parts:
            if not part:
                continue
            
            is_h_block = part.lower().startswith('<h>') and part.lower().endswith('</h>')
            
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
                        current_line = char
                        current_column_height = line_height_approx
    
    # 添加最后一行
    if current_line:
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
            except:
                pass
        line_max_widths.append(max_char_width)

    current_x_col = current_x_base
    for line_idx, line in enumerate(lines):
        current_y_char = start_y_base
        # 获取当前列的实际宽度
        line_width = line_max_widths[line_idx] if line_idx < len(line_max_widths) else font_size
        
        # ===== 分割行内容为普通文本和横排块 =====
        # 使用正则表达式分割 <H>...</H> 标签
        parts = re.split(r'(<H>.*?</H>)', line, flags=re.IGNORECASE | re.DOTALL)
        
        for part in parts:
            if not part:
                continue
            
            # 检查是否为横排块
            is_horizontal_block = part.lower().startswith('<h>') and part.lower().endswith('</h>')
            
            if is_horizontal_block:
                # ===== 渲染横排块 =====
                content = part[3:-4]  # 去除 <H> 和 </H>
                if content:
                    block_height = render_horizontal_block(
                        content=content,
                        font=font,
                        font_size=font_size,
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
            else:
                # ===== 渲染普通竖排字符 =====
                for char in part:
                    # 调用 CJK_Compatibility_Forms_translate 获取转换后的字符和旋转角度
                    converted_char, rot_degree = CJK_Compatibility_Forms_translate(char, 1)  # 1 = 竖排
                    
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
                                        logger.warning(f"回退字体加载失败: {fallback_path} - {e}")
                                        continue
                    else:
                        if converted_char in SPECIAL_CHARS:
                            if special_font is None:
                                try:
                                    special_font = get_font(NOTOSANS_FONT_PATH, font_size)
                                except Exception as e:
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
                        # 创建足够大的临时图像（带边距以避免裁剪）
                        padding = max(char_width, char_height)
                        temp_size = max(char_width, char_height) * 2 + padding * 2
                        temp_size = max(temp_size, font_size * 3)  # 确保足够大
                        temp_size = int(temp_size)
                        
                        temp_img = Image.new('RGBA', (temp_size, temp_size), (0, 0, 0, 0))
                        temp_draw = ImageDraw.Draw(temp_img)
                        
                        temp_x = temp_size // 2 - char_width // 2
                        temp_y = temp_size // 2 - char_height // 2
                        
                        temp_text_params = {
                            "font": current_font,
                            "fill": fill if canvas_image.mode == 'RGBA' else fill
                        }
                        if stroke_enabled:
                            temp_text_params["stroke_width"] = int(stroke_width)
                            temp_text_params["stroke_fill"] = stroke_color
                        
                        temp_draw.text((temp_x, temp_y), converted_char, **temp_text_params)
                        
                        rotated_img = temp_img.rotate(-rot_degree, resample=Image.Resampling.BICUBIC, expand=False)
                        
                        # 计算粘贴位置（水平居中）
                        # 使用预计算的 line_width（该列实际最大字符宽度）
                        paste_x = int((current_x_col - line_width) + round((line_width - temp_size) / 2.0))
                        paste_y = int(current_y_char)
                        
                        try:
                            if canvas_image.mode == 'RGBA':
                                canvas_image.paste(rotated_img, (paste_x, paste_y), rotated_img)
                            else:
                                # 如果主画布不是 RGBA，转换后粘贴
                                rgb_rotated = rotated_img.convert('RGB')
                                canvas_image.paste(rgb_rotated, (paste_x, paste_y))
                        except Exception as e:
                            logger.warning(f"旋转字符粘贴失败: {e}，回退到直接绘制")
                            # 回退：直接绘制（不旋转）
                            text_x_char = current_x_col - char_width
                            draw.text((text_x_char, current_y_char), converted_char, **text_draw_params)
                    else:
                        # ===== 常规绘制（不需要旋转） =====
                        # ===== 水平居中计算 =====
                        # 计算字符在当前列中的水平居中位置，line_width 为该列的最大字符宽度。
                        # 使用预计算的 line_width（该列实际最大字符宽度）
                        text_x_char = (current_x_col - line_width) + round((line_width - char_width) / 2.0)
                        text_y_char = current_y_char
                        
                        # ===== 墨水偏移校正（水平+垂直）=====
                        # Pillow 的 getbbox() 返回的边界框可能不等于实际墨水区域
                        # 对于某些字符（如竖排标点），实际墨水可能偏向边界框的一侧
                        # 需要校正以实现真正的视觉居中
                        ink_offset_x, ink_offset_y = get_char_ink_offset(converted_char, current_font)
                        text_x_char -= ink_offset_x  # 反向补偿水平墨水偏移
                        
                        # ===== 竖排标点符号垂直居中校正（使用墨水偏移）=====
                        # 获取参考汉字（如"我"）的墨水 y 偏移作为基准
                        # 所有标点的墨水中心应该与汉字的墨水中心对齐
                        if is_vertical_punctuation(converted_char):
                            # 计算参考汉字的墨水偏移（使用缓存避免重复计算）
                            if not hasattr(get_char_ink_offset, '_ref_y_offset'):
                                ref_font = get_font(font_family_path, font_size) if font_family_path else current_font
                                _, ref_y = get_char_ink_offset('我', ref_font)
                                get_char_ink_offset._ref_y_offset = ref_y
                            ref_y_offset = get_char_ink_offset._ref_y_offset
                            
                            # 垂直对齐：将标点的墨水中心与汉字的墨水中心对齐
                            # 如果 ink_offset_y > ref_y_offset，说明标点偏下，需要上移
                            # 如果 ink_offset_y < ref_y_offset，说明标点偏上，需要下移
                            vertical_correction = ref_y_offset - ink_offset_y
                            text_y_char += vertical_correction
                        
                        # 直接绘制
                        draw.text((text_x_char, text_y_char), converted_char, **text_draw_params)
                    
                    current_y_char += line_height_approx
        
        current_x_col -= column_width_approx

# --- 横排文本绘制函数（不含旋转，旋转在 render_all_bubbles 中统一处理） ---
def draw_multiline_text_horizontal(draw, text, font, x, y, max_width,
                                  fill=constants.DEFAULT_TEXT_COLOR,
                                  stroke_enabled=constants.DEFAULT_STROKE_ENABLED,
                                  stroke_color=constants.DEFAULT_STROKE_COLOR,
                                  stroke_width=constants.DEFAULT_STROKE_WIDTH,
                                  bubble_width=None,
                                  bubble_height=None,
                                  font_family_path=constants.DEFAULT_FONT_RELATIVE_PATH):
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
                                logger.warning(f"回退字体加载失败: {fallback_path} - {e}")
                                continue
            else:
                # FreeType 不可用时，回退到使用 SPECIAL_CHARS 检查
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
                        bubble_width=max_text_width,
                        font_family_path=font_family_rel
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
                        bubble_height=max_text_height,
                        font_family_path=font_family_rel
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
                        bubble_width=max_text_width,
                        font_family_path=font_family_rel
                    )
                elif text_direction == 'horizontal':
                    draw_multiline_text_horizontal(
                        draw, text, font, draw_x, draw_y, max_text_width,
                        fill=text_color,
                        stroke_enabled=stroke_enabled,
                        stroke_color=stroke_color,
                        stroke_width=stroke_width,
                        bubble_width=max_text_width,
                        bubble_height=max_text_height,
                        font_family_path=font_family_rel
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
                        bubble_width=max_text_width,
                        font_family_path=state.font_family
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
                        bubble_height=max_text_height,
                        font_family_path=state.font_family
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
                        bubble_width=max_text_width,
                        font_family_path=state.font_family
                    )
                else:
                    draw_multiline_text_horizontal(
                        draw, text, font, draw_x, draw_y, max_text_width,
                        fill=state.text_color,
                        stroke_enabled=state.stroke_enabled,
                        stroke_color=state.stroke_color,
                        stroke_width=state.stroke_width,
                        bubble_width=max_text_width,
                        bubble_height=max_text_height,
                        font_family_path=state.font_family
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