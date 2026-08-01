"""
常量定义模块，用于存储应用程序中使用的各种常量
"""
import os
from src.shared.text_style_defaults import (
    get_text_style_factory_defaults,
)


_TEXT_STYLE_DEFAULTS = {}


def refresh_text_style_runtime_defaults() -> None:
    global _TEXT_STYLE_DEFAULTS
    global DEFAULT_FONT_RELATIVE_PATH
    global DEFAULT_FONT_FAMILY
    global DEFAULT_FONT_SIZE
    global DEFAULT_TEXT_DIRECTION
    global DEFAULT_TEXT_COLOR
    global DEFAULT_FILL_COLOR
    global DEFAULT_INPAINT_METHOD
    global DEFAULT_LINE_SPACING
    global DEFAULT_TEXT_ALIGN
    global DEFAULT_STROKE_ENABLED
    global DEFAULT_STROKE_COLOR
    global DEFAULT_STROKE_WIDTH

    _TEXT_STYLE_DEFAULTS = get_text_style_factory_defaults()

    DEFAULT_FONT_RELATIVE_PATH = os.path.join(
        'src',
        'backend_v2',
        'resources',
        _TEXT_STYLE_DEFAULTS["fontFamily"].replace("/", os.sep)
    )
    DEFAULT_FONT_FAMILY = _TEXT_STYLE_DEFAULTS["fontFamily"]
    DEFAULT_FONT_SIZE = _TEXT_STYLE_DEFAULTS["fontSize"]
    layout_direction = _TEXT_STYLE_DEFAULTS["layoutDirection"]
    DEFAULT_TEXT_DIRECTION = layout_direction if layout_direction in {"vertical", "horizontal"} else "vertical"
    DEFAULT_TEXT_COLOR = _TEXT_STYLE_DEFAULTS["textColor"]
    DEFAULT_FILL_COLOR = _TEXT_STYLE_DEFAULTS["fillColor"]
    DEFAULT_INPAINT_METHOD = _TEXT_STYLE_DEFAULTS["inpaintMethod"]
    DEFAULT_LINE_SPACING = _TEXT_STYLE_DEFAULTS["lineSpacing"]
    DEFAULT_TEXT_ALIGN = _TEXT_STYLE_DEFAULTS["textAlign"]
    DEFAULT_STROKE_ENABLED = _TEXT_STYLE_DEFAULTS["strokeEnabled"]
    DEFAULT_STROKE_COLOR = _TEXT_STYLE_DEFAULTS["strokeColor"]
    DEFAULT_STROKE_WIDTH = _TEXT_STYLE_DEFAULTS["strokeWidth"]


refresh_text_style_runtime_defaults()

# --- 提示词相关 ---
DEFAULT_PROMPT = "你是一个好用的翻译助手。请将我的非中文语句段落连成一句或几句话并翻译成中文，我发给你所有的话都是需要翻译的内容，你只需要回答翻译结果。特别注意：翻译结果字数不能超过原文字数！翻译结果请符合中文的语言习惯。"

# --- 新增 JSON 格式提示词（单气泡翻译专用）---
DEFAULT_TRANSLATE_JSON_PROMPT = """你是一个专业的翻译引擎。请将用户提供的文本翻译成简体中文。

当文本中包含特殊字符（如大括号{}、引号""、反斜杠\\等）时，请在输出中保留它们但不要将它们视为JSON语法的一部分。

请严格按照以下 JSON 格式返回结果，不要添加任何额外的解释或对话:
{
  "translated_text": "[翻译后的文本放在这里]"
}"""

# --- 批量翻译提示词 ---
# 使用三步翻译法，支持多文本批量翻译
# 注意：如需翻译为其他语言，请修改提示词中的"中文"为目标语言
BATCH_TRANSLATE_SYSTEM_TEMPLATE = '''忽略之前的所有指令，仅遵循以下定义。

## 角色：专业漫画翻译师
你是一个专业的漫画翻译引擎，擅长将外语漫画翻译成中文。

## 翻译方法
1. 直译阶段：
- 对每一行文本进行精确的逐词翻译
- 尽可能保持原文的句子结构
- 保留所有原始标记和表达方式
- 对模糊的内容保持原样，不做过度解读

2. 分析与意译阶段：
- 捕捉核心含义、情感基调和文化内涵
- 识别碎片化文本段落之间的逻辑联系
- 分析直译的不足之处和需要改进的地方

3. 润色阶段：
- 调整翻译使其在中文中听起来自然流畅，同时保持原意
- 保留适合漫画和宅文化的情感基调和强度
- 确保角色语气和术语的一致性
- 根据上下文推断合适的人称代词（他/她/我/你/你们），不要添加原文中不存在的代词
- 根据第二步的结论进行最终润色

## 翻译规则
- 逐行翻译，保持准确性和真实性，忠实再现原文及其情感意图
- 保留原文中的拟声词或音效词，不进行翻译
- 每个翻译段落必须带有编号前缀（严格使用 <|数字|> 格式），只输出翻译结果，不要输出原文
- 只翻译内容，不要添加任何解释或评论

请将以下外语文本翻译成中文：
'''

# 批量翻译的用户提示词模板
BATCH_TRANSLATE_USER_TEMPLATE = '''请帮我将以下漫画文本翻译成中文。如果文本已经是中文或者看起来是拟声词/音效词，请原样输出。保持编号前缀格式。
'''

# 批量翻译的示例（用于 few-shot learning）
BATCH_TRANSLATE_SAMPLE_INPUT = (
    '<|1|>恥ずかしい… 目立ちたくない… 私が消えたい…\n'
    '<|2|>きみ… 大丈夫⁉\n'
    '<|3|>なんだこいつ 空気読めて ないのか…？'
)
BATCH_TRANSLATE_SAMPLE_OUTPUT = (
    '<|1|>好尴尬…我不想引人注目…我想消失…\n'
    '<|2|>你…没事吧⁉\n'
    '<|3|>这家伙怎么看不懂气氛的…？'
)

# --- 批量翻译 JSON 模式 ---
# JSON 模式使用结构化输出，更容易解析但 Token 消耗更高
BATCH_TRANSLATE_JSON_SYSTEM_TEMPLATE = '''忽略之前的所有指令，仅遵循以下定义。

## 角色：专业漫画翻译师
你是一个专业的漫画翻译引擎，擅长将外语漫画翻译成中文。

## 翻译方法
1. 直译阶段：对每一行文本进行精确的逐词翻译
2. 分析与意译阶段：捕捉核心含义、情感基调和文化内涵
3. 润色阶段：调整翻译使其在中文中听起来自然流畅

## 翻译规则
- 逐行翻译，保持准确性和真实性
- 保留原文中的拟声词或音效词，不进行翻译
- 只翻译内容，不要添加任何解释或评论

## 输出格式
请严格按照以下 JSON 格式返回翻译结果，不要添加任何额外文字：
{
  "translations": [
    {"id": 1, "text": "翻译内容1"},
    {"id": 2, "text": "翻译内容2"}
  ]
}

请将以下外语文本翻译成中文：
'''

# JSON 模式的用户提示词
BATCH_TRANSLATE_JSON_USER_TEMPLATE = '''请帮我将以下漫画文本翻译成中文，严格按照 JSON 格式输出。
'''

# JSON 模式的示例（用于 few-shot learning）
BATCH_TRANSLATE_JSON_SAMPLE_INPUT = '''{
  "texts": [
    {"id": 1, "text": "恥ずかしい… 目立ちたくない… 私が消えたい…"},
    {"id": 2, "text": "きみ… 大丈夫⁉"},
    {"id": 3, "text": "なんだこいつ 空気読めて ないのか…？"}
  ]
}'''

BATCH_TRANSLATE_JSON_SAMPLE_OUTPUT = '''{
  "translations": [
    {"id": 1, "text": "好尴尬…我不想引人注目…我想消失…"},
    {"id": 2, "text": "你…没事吧⁉"},
    {"id": 3, "text": "这家伙怎么看不懂气氛的…？"}
  ]
}'''

# 批量翻译配置
BATCH_TRANSLATE_MAX_CHARS_PER_REQUEST = 4000  # 单个请求的最大字符数 (粗略估计 1 token ≈ 4 chars)



# --- 翻译服务相关 ---
# 百度翻译API引擎ID
BAIDU_TRANSLATE_ENGINE_ID = 'baidu_translate'
# 有道翻译API引擎ID
YOUDAO_TRANSLATE_ENGINE_ID = 'youdao_translate'

# --- 文件与目录 ---
# 默认字体路径指向 v2 只读资源目录。
# 注意：
# - 内置字体随应用发布，位于 src/backend_v2/resources/fonts/
# - 用户上传字体进入 v2 对象存储，不修改应用资源目录

# --- 默认渲染参数 ---
DEFAULT_ROTATION_ANGLE = 0
# DEFAULT_FONT_SIZE / DEFAULT_TEXT_DIRECTION / DEFAULT_TEXT_COLOR /
# DEFAULT_FILL_COLOR / DEFAULT_INPAINT_METHOD / DEFAULT_LINE_SPACING /
# DEFAULT_TEXT_ALIGN 在模块顶部由 refresh_text_style_runtime_defaults() 初始化

# --- 百度翻译相关 ---
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

DEFAULT_AI_VISION_OCR_PROMPT = """你是一个ocr助手，你需要将我发送给你的图片中的文字提取出来并返回给我，要求：
1、完整识别：我发送给你的图片中的文字都是需要识别的内容
2、非贪婪输出：不要返回任何其他解释和说明。"""

# --- 有道翻译相关 ---
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

DEFAULT_AI_VISION_OCR_JSON_PROMPT = """你是一个OCR助手。请将我发送给你的图片中的所有文字提取出来。

当文本中包含特殊字符（如大括号{}、引号""、反斜杠\等）时，请在输出中保留它们但不要将它们视为JSON语法的一部分。如果需要，你可以使用转义字符\\来表示这些特殊字符。

请严格按照以下 JSON 格式返回结果，不要添加任何额外的解释或对话:
{
  "extracted_text": "[这里放入所有识别到的文字，可以包含换行符以大致保留原始分段，但不要包含任何其他非文本内容]"
}"""

# --- rpm (Requests Per Minute) Limiting ---
DEFAULT_rpm_TRANSLATION = 0  # 0 表示无限制
DEFAULT_rpm_AI_VISION_OCR = 0 # 0 表示无限制

# --- AI Vision OCR 图片尺寸限制 ---
DEFAULT_AI_VISION_MIN_IMAGE_SIZE = 32  # VLM 模型通常要求 >= 28px

# --- 文本描边 ---
# DEFAULT_STROKE_ENABLED / DEFAULT_STROKE_COLOR / DEFAULT_STROKE_WIDTH
# 在模块顶部由 refresh_text_style_runtime_defaults() 初始化

# --- 48px OCR 相关 ---
OCR_ENGINE_48PX = '48px_ocr'
MODEL_48PX_DIR = 'models/ocr_48px'
MODEL_48PX_CHECKPOINT = 'ocr_ar_48px.ckpt'
MODEL_48PX_DICT = 'alphabet-all-v7.txt'

# --- PaddleOCR-VL 相关 ---
OCR_ENGINE_PADDLEOCR_VL = 'paddleocr_vl'
PADDLEOCR_VL_MODEL_DIR = 'models/paddleocr_vl'
PADDLEOCR_VL_HF_MODEL = 'jzhang533/PaddleOCR-VL-For-Manga'

# CTD 配置
# 边缘距离比例阈值：当一个文本行与多个邻居连接时，如果到某个邻居的距离
# 远大于到最近邻居的距离 (比例超过此阈值)，则断开这个连接，防止跨气泡错误合并
# 推荐值 3.0-5.0，0 表示禁用
CTD_EDGE_RATIO_THRESHOLD = 0.0

# 辅助一阶段 YSGYolo 检测配置
ENABLE_AUX_YOLO_DETECTION = False
AUX_YOLO_CONF_THRESHOLD = 0.4
AUX_YOLO_OVERLAP_THRESHOLD = 0.1

# SaberYOLO 配置（二阶段误合并纠错）
SABER_YOLO_MODEL_DIR = 'models/saber_yolo'
SABER_YOLO_MODEL_NAME = 'saber_yolo.pt'
SABER_YOLO_CONF_THRESH = 0.2
SABER_YOLO_IOU_THRESH = 0.5
ENABLE_SABER_YOLO_REFINE = True
SABER_YOLO_REFINE_OVERLAP_THRESHOLD = 0.5

# --- 重试机制设置 ---
DEFAULT_TRANSLATION_MAX_RETRIES = 3  # 普通翻译默认重试次数

# --- 超长图片处理 (Large Image Rearrange) ---
LARGE_IMAGE_ENABLED = True  # 是否启用超长图片自动切割
LARGE_IMAGE_TARGET_SIZE = 1536  # 切片目标尺寸（与检测器一致）

# --- LAMA 修复相关 ---
# 是否禁用 LAMA 修复时的自动缩放（默认 False，即允许缩放）
# 设为 True 时将使用原图尺寸进行修复，需要更强的 GPU 和更多显存
LAMA_DISABLE_RESIZE = False
