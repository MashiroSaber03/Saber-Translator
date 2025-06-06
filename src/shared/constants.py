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

# --- 新增自定义OpenAI服务商ID ---
CUSTOM_OPENAI_PROVIDER_ID = 'custom_openai'
# ---------------------------------

# --- 翻译服务相关 ---
# 百度翻译API引擎ID
BAIDU_TRANSLATE_ENGINE_ID = 'baidu_translate'
# 有道翻译API引擎ID
YOUDAO_TRANSLATE_ENGINE_ID = 'youdao_translate'

# --- 文件与目录 ---
# 默认字体路径现在指向 src/app/static/fonts/
DEFAULT_FONT_RELATIVE_PATH = os.path.join('src', 'app', 'static', 'fonts', 'msyh.ttc')
DEFAULT_FONT_PATH = "static/msyh.ttc"  # 保留旧变量以兼容现有代码
UPLOAD_FOLDER_NAME = 'uploads'
TEMP_FOLDER_NAME = 'temp'
UPLOAD_FOLDER = 'uploads'  # 保留旧变量以兼容现有代码
TEMP_FOLDER = 'temp'  # 保留旧变量以兼容现有代码

# --- 默认翻译与渲染参数 ---
DEFAULT_TARGET_LANG = 'zh'
DEFAULT_SOURCE_LANG = 'japan'
DEFAULT_FONT_SIZE = 30
DEFAULT_TEXT_DIRECTION = 'vertical'
DEFAULT_TEXT_COLOR = '#231816'
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

# --- 百度OCR相关 ---
BAIDU_OCR_VERSIONS = {
    "standard": "标准版",
    "high_precision": "高精度版"
}

# 百度OCR语言映射使用大写编码
# 参考文档: https://cloud.baidu.com/doc/OCR/s/zk3h7xz52
BAIDU_LANG_MAP = {
    "japan": "japanese",
    "japanese": "japanese",
    "en": "english",
    "english": "english", 
    "korean": "korean",
    "chinese": "chinese",
    "chinese_cht": "chinese",
    "french": "french",
    "german": "german",
    "russian": "russian",
    "italian": "italian",
    "spanish": "spanish"
}

# --- 百度翻译相关 ---
# 百度翻译语言映射
# 参考文档: https://fanyi-api.baidu.com/doc/21
BAIDU_TRANSLATE_LANG_MAP = {
    'zh': 'zh',       # 中文
    'en': 'en',       # 英语
    'ja': 'jp',       # 日语 (百度API使用jp)
    'ko': 'kor',      # 韩语
    'fr': 'fra',      # 法语
    'es': 'spa',      # 西班牙语
    'it': 'it',       # 意大利语
    'de': 'de',       # 德语
    'ru': 'ru',       # 俄语
    'pt': 'pt',       # 葡萄牙语
    'vi': 'vie',      # 越南语
    'th': 'th',       # 泰语
    'auto': 'auto',   # 自动检测
}

# 项目内部语言代码到百度翻译语言代码的映射
PROJECT_TO_BAIDU_TRANSLATE_LANG_MAP = {
    'zh': 'zh',
    'en': 'en',
    'japan': 'jp',
    'korean': 'kor',
    'chinese': 'zh',
    'chinese_cht': 'zh',
    'french': 'fra',
    'german': 'de',
    'russian': 'ru',
    'italian': 'it',
    'spanish': 'spa'
}

# --- AI 视觉 OCR 相关 ---
AI_VISION_OCR_ENGINE_ID = 'ai_vision'  # 定义唯一标识符

# 新增自定义OpenAI兼容服务商ID (确保这个值是唯一的)
CUSTOM_AI_VISION_PROVIDER_ID = 'custom_openai_vision' # 与翻译的自定义ID区分

DEFAULT_AI_VISION_OCR_PROMPT = """你是一个ocr助手，你需要将我发送给你的图片中的文字提取出来并返回给我，要求：
1、完整识别：我发送给你的图片中的文字都是需要识别的内容
2、非贪婪输出：不要返回任何其他解释和说明。"""

# 新增支持的 AI 视觉服务商列表
SUPPORTED_AI_VISION_PROVIDERS = {
    'siliconflow': 'SiliconFlow (硅基流动)',
    'volcano': '火山引擎',
    'gemini': 'Google Gemini',
    CUSTOM_AI_VISION_PROVIDER_ID: '自定义 OpenAI 兼容视觉服务'
}

# --- 有道翻译相关 ---
# 有道翻译语言映射
# 参考文档: https://ai.youdao.com/DOCSIRMA/html/trans/api/wbfy/index.html
YOUDAO_TRANSLATE_LANG_MAP = {
    'zh': 'zh-CHS',    # 中文简体
    'en': 'en',        # 英语
    'ja': 'ja',        # 日语
    'ko': 'ko',        # 韩语
    'fr': 'fr',        # 法语
    'es': 'es',        # 西班牙语
    'it': 'it',        # 意大利语
    'de': 'de',        # 德语
    'ru': 'ru',        # 俄语
    'pt': 'pt',        # 葡萄牙语
    'vi': 'vi',        # 越南语
    'auto': 'auto',    # 自动检测
}

# 项目内部语言代码到有道翻译语言代码的映射
PROJECT_TO_YOUDAO_TRANSLATE_LANG_MAP = {
    'zh': 'zh-CHS',
    'en': 'en',
    'japan': 'ja',
    'korean': 'ko',
    'chinese': 'zh-CHS',
    'chinese_cht': 'zh-TW',
    'french': 'fr',
    'german': 'de',
    'russian': 'ru',
    'italian': 'it',
    'spanish': 'es'
}

# --- 新增 JSON 格式提示词 ---
DEFAULT_TRANSLATE_JSON_PROMPT = """你是一个专业的翻译引擎。请将用户提供的文本翻译成简体中文。

当文本中包含特殊字符（如大括号{}、引号""、反斜杠\等）时，请在输出中保留它们但不要将它们视为JSON语法的一部分。

请严格按照以下 JSON 格式返回结果，不要添加任何额外的解释或对话:
{
  "translated_text": "[翻译后的文本放在这里]"
}"""

DEFAULT_AI_VISION_OCR_JSON_PROMPT = """你是一个OCR助手。请将我发送给你的图片中的所有文字提取出来。

当文本中包含特殊字符（如大括号{}、引号""、反斜杠\等）时，请在输出中保留它们但不要将它们视为JSON语法的一部分。如果需要，你可以使用转义字符\\来表示这些特殊字符。

请严格按照以下 JSON 格式返回结果，不要添加任何额外的解释或对话:
{
  "extracted_text": "[这里放入所有识别到的文字，可以包含换行符以大致保留原始分段，但不要包含任何其他非文本内容]"
}"""

# --- rpm (Requests Per Minute) Limiting ---
DEFAULT_rpm_TRANSLATION = 0  # 0 表示无限制
DEFAULT_rpm_AI_VISION_OCR = 0 # 0 表示无限制

# --- 文本描边默认值 ---
DEFAULT_TEXT_STROKE_ENABLED = False
DEFAULT_TEXT_STROKE_COLOR = '#FFFFFF' # 默认白色描边
DEFAULT_TEXT_STROKE_WIDTH = 1         # 默认1像素宽度
# ------------------------