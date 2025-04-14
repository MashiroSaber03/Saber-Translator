"""
常量定义模块，用于存储应用程序中使用的各种常量
"""
import os

# --- 提示词相关 ---
DEFAULT_PROMPT = "你是一个好用的翻译助手。请将我的非中文语句段落连成一句或几句话并翻译成中文，我发给你所有的话都是需要翻译的内容，你只需要回答翻译结果。特别注意：翻译结果字数不能超过原文字数！翻译结果请符合中文的语言习惯。"
DEFAULT_TEXTBOX_PROMPT = "你是一个专业的外语老师。请将我提供的非中文内容连成一句或几句话并翻译成中文。同时要告诉我为什么这么翻译，这句话有哪些知识点。"
DEFAULT_PROMPT_NAME = "默认提示词"

# --- 模型与历史 ---
MAX_MODEL_HISTORY = 5
DEFAULT_MODEL_PROVIDER = 'siliconflow'
MODEL_HISTORY_FILE = 'model_history.json'
PROMPTS_FILE = 'prompts.json'
TEXTBOX_PROMPTS_FILE = 'textbox_prompts.json'

# --- 文件与目录 ---
# 默认字体路径现在指向 src/app/static/fonts/
DEFAULT_FONT_RELATIVE_PATH = os.path.join('src', 'app', 'static', 'fonts', 'STSONG.TTF')
DEFAULT_FONT_PATH = "static/STSONG.TTF"  # 保留旧变量以兼容现有代码
UPLOAD_FOLDER_NAME = 'uploads'
TEMP_FOLDER_NAME = 'temp'
UPLOAD_FOLDER = 'uploads'  # 保留旧变量以兼容现有代码
TEMP_FOLDER = 'temp'  # 保留旧变量以兼容现有代码

# --- 默认翻译与渲染参数 ---
DEFAULT_TARGET_LANG = 'zh'
DEFAULT_SOURCE_LANG = 'japan'
DEFAULT_FONT_SIZE = 30
DEFAULT_TEXT_DIRECTION = 'vertical'
DEFAULT_TEXT_COLOR = '#000000'
DEFAULT_ROTATION_ANGLE = 0
DEFAULT_FILL_COLOR = '#FFFFFF'
DEFAULT_INPAINTING_STRENGTH = 1.0

# --- OCR 相关 ---
SUPPORTED_LANGUAGES_OCR = {
    "japan": "MangaOCR",
    "en": "PaddleOCR",
    "korean": "PaddleOCR",
    "chinese": "PaddleOCR",
    "chinese_cht": "PaddleOCR",
    "french": "PaddleOCR",
    "german": "PaddleOCR",
    "russian": "PaddleOCR",
    "italian": "PaddleOCR",
    "spanish": "PaddleOCR"
}
PADDLE_LANG_MAP = {
    "en": "en",
    "korean": "korean",
    "chinese": "ch",
    "chinese_cht": "chinese_cht",
    "french": "french",
    "german": "german",
    "russian": "ru",
    "italian": "italian",
    "spanish": "spanish"
}