"""
常量定义模块，用于存储应用程序中使用的各种常量
"""
import os

# --- 提示词相关 ---
DEFAULT_PROMPT = "你是一个好用的翻译助手。请将我的非中文语句段落连成一句或几句话并翻译成中文，我发给你所有的话都是需要翻译的内容，你只需要回答翻译结果。特别注意：翻译结果字数不能超过原文字数！翻译结果请符合中文的语言习惯。"
DEFAULT_TEXTBOX_PROMPT = "你是一个专业的外语老师。请将我提供的非中文内容连成一句或几句话并翻译成中文。同时要告诉我为什么这么翻译，这句话有哪些知识点。"
DEFAULT_PROMPT_NAME = "默认提示词"

# --- 批量翻译提示词 ---
# 使用三步翻译法，支持多文本批量翻译
# 注意：如需翻译为其他语言，请修改提示词中的"中文"为目标语言
BATCH_TRANSLATE_SYSTEM_TEMPLATE = '''忽略之前的所有指令，仅遵循以下定义。

## 角色：专业漫画翻译师
你是一个专业的漫画翻译引擎，擅长将日文漫画翻译成中文。

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

请将以下日文文本翻译成中文：
'''

# 批量翻译的用户提示词模板
BATCH_TRANSLATE_USER_TEMPLATE = '''请帮我将以下漫画文本翻译成中文。如果文本已经是中文或者看起来是拟声词/音效词，请原样输出。保持编号前缀格式。
'''

# 批量翻译的示例（用于 few-shot learning，日译中）
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
你是一个专业的漫画翻译引擎，擅长将日文漫画翻译成中文。

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

请将以下日文文本翻译成中文：
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
# 注意：
# - 临时文件实际存储在 data/temp/ 目录
# - PDF 上传直接处理为 base64，不保存到磁盘
# - 字体上传保存到 src/app/static/fonts/ 目录
# - 图片上传在前端转为 base64，不经过后端文件系统

# --- 默认翻译与渲染参数 ---
DEFAULT_TARGET_LANG = 'zh'
DEFAULT_SOURCE_LANG = 'japan'
DEFAULT_FONT_SIZE = 30
DEFAULT_TEXT_DIRECTION = 'vertical'
DEFAULT_TEXT_COLOR = '#231816'
DEFAULT_ROTATION_ANGLE = 0
DEFAULT_FILL_COLOR = '#FFFFFF'

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
    "spanish": "PaddleOCR",
    "portuguese": "PaddleOCR",
    "arabic": "PaddleOCR",
    "thai": "PaddleOCR",
    "greek": "PaddleOCR"
}

# PaddleOCR v5 ONNX 语言映射（前端语言代码 -> ONNX 模型目录名）
# 注意：使用 ONNX 版本后，映射值对应 models/paddle_ocr_onnx/languages/ 下的目录名
PADDLE_V5_LANG_MAP = {
    # 中日文 - 使用 chinese 模型（PP-OCRv5 chinese 模型支持中文和日文）
    "japanese": "chinese",      # 日文
    "japan": "chinese",         # 兼容旧代码
    "chinese": "chinese",       # 简体中文
    "ch": "chinese",            # 简体中文
    "chinese_cht": "chinese",   # 繁体中文
    
    # 英文专用模型
    "en": "english",            # 英文
    "english": "english",
    
    # 韩语专用模型
    "korean": "korean",         # 韩语
    
    # 拉丁语系模型（覆盖 32 种欧洲语言）
    "french": "latin",          # 法语
    "german": "latin",          # 德语
    "spanish": "latin",         # 西班牙语
    "italian": "latin",         # 意大利语
    "portuguese": "latin",      # 葡萄牙语
    "latin": "latin",           # 拉丁语系通用
    
    # 斯拉夫语系模型（俄语、乌克兰语、保加利亚语、白俄罗斯语）
    "russian": "eslav",         # 俄语
    "eslav": "eslav",           # 东斯拉夫语系
    "cyrillic": "eslav",        # 西里尔语系 -> 映射到 eslav
    
    # 以下语言需要单独下载模型才能使用
    # 运行: python download_paddle_onnx_models.py thai greek
    # "thai": "thai",           # 泰语 (需下载)
    # "th": "thai",
    # "greek": "greek",         # 希腊语 (需下载)
    # "el": "greek",
    # "arabic": "arabic",       # 阿拉伯语 (v5 不支持)
}

# 前端显示的语言列表（按模型分组）
PADDLE_V5_LANGUAGES = {
    "default": {
        "japanese": "日语",
        "en": "英语",
        "chinese": "简体中文",
        "chinese_cht": "繁体中文",
    },
    "specialized": {
        "korean": "韩语",
        "french": "法语",
        "german": "德语",
        "spanish": "西班牙语",
        "italian": "意大利语",
        "portuguese": "葡萄牙语",
        "russian": "俄语",
        "arabic": "阿拉伯语",
        "thai": "泰语",
        "greek": "希腊语",
    }
}

# 保留旧的映射以兼容旧代码
PADDLE_LANG_MAP = PADDLE_V5_LANG_MAP

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

# --- 高质量翻译默认提示词 ---
DEFAULT_HQ_TRANSLATE_PROMPT = """你是一个漫画翻译助手，我会提供原始漫画图片和导出的只含原文不含译文JSON翻译文件，帮我将原文翻译成中文
翻译要求：
1.仅修改JSON中的"translated"译文，保持其他所有结构和字段不变。
2.json中的"imagelndex"序号是每张图片的页码，在翻译前先根据json文件中每页图片的原文内容和所有漫画图片对我给你的所有图片进行排序，在进行上下文对比时要严格按照"imagelndex"序号进行对比，你在翻译时要按照"imagelndex"的顺序进行顺序翻译，从而使得每句翻译足够连贯，上下文不会突兀
3.通过json中每页图片的"original"原文内容和所有的漫画图片明确每句话在那页图的哪个位置，并结合漫画图像和上下文语境，让翻译更加连贯自然，符合角色语气和场景
4.json中的"bubblelndex"标号可能不正确，你需要根据图片自行判断每句话的正确顺序，从而输出符合上下文连贯的翻译，但在输出翻译时不要修改原本的"bubblelndex"标号
5.保留原文的表达意图和情感，但使用更地道、流畅的表达方式
6.注意专有名词和术语的一致性翻译
7.如遇幽默、双关语或文化特定内容，请尽量找到目标语言中恰当的表达
8.不要添加原文中不存在的信息或解释
9.保持简洁明了，符合气泡空间限制
10.译文字数尽量不要超过原文
11.你需要直接返回修改后的完整JSON文件，无需解释每处修改原因。"""

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
DEFAULT_STROKE_ENABLED = True
DEFAULT_STROKE_COLOR = '#FFFFFF'
DEFAULT_STROKE_WIDTH = 3

# --- 48px OCR 相关 ---
OCR_ENGINE_48PX = '48px_ocr'
MODEL_48PX_DIR = 'models/ocr_48px'
MODEL_48PX_CHECKPOINT = 'ocr_ar_48px.ckpt'
MODEL_48PX_DICT = 'alphabet-all-v7.txt'

# --- PaddleOCR-VL 相关 ---
OCR_ENGINE_PADDLEOCR_VL = 'paddleocr_vl'
PADDLEOCR_VL_MODEL_DIR = 'models/paddleocr_vl'
PADDLEOCR_VL_HF_MODEL = 'jzhang533/PaddleOCR-VL-For-Manga'

# --- 文本检测器相关 ---
DETECTOR_CTD = 'ctd'
DETECTOR_YOLO = 'yolo'
DETECTOR_YOLOV5 = 'yolov5'  # 旧版 YOLOv5 检测器
DETECTOR_DEFAULT = 'default'  # DBNet ResNet34 (detect-20241225.ckpt)
DEFAULT_DETECTOR = DETECTOR_DEFAULT

SUPPORTED_DETECTORS = {
    DETECTOR_DEFAULT: 'Default (DBNet ResNet34)',
    DETECTOR_CTD: 'CTD (Comic Text Detector)',
    DETECTOR_YOLO: 'YSGYolo',
    DETECTOR_YOLOV5: 'YOLOv5 (旧版)'
}

# CTD 配置
# 边缘距离比例阈值：当一个文本行与多个邻居连接时，如果到某个邻居的距离
# 远大于到最近邻居的距离 (比例超过此阈值)，则断开这个连接，防止跨气泡错误合并
# 推荐值 3.0-5.0，0 表示禁用
CTD_EDGE_RATIO_THRESHOLD = 0.0

# YOLOv5 配置 (旧版)
YOLOV5_CONF_THRESH = 0.6

# YSGYolo 配置
YOLO_MODEL_DIR = 'models/yolo'
YOLO_DEFAULT_MODEL = 'ysgyolo_1.2_OS1.0.pt'
YOLO_CONF_THRESH = 0.3
YOLO_IOU_THRESH = 0.5
YOLO_MASK_DILATE_SIZE = 2

# Default (DBNet ResNet34) 配置
DEFAULT_MODEL_DIR = 'models/default'
DEFAULT_MODEL_NAME = 'detect-20241225.ckpt'
DEFAULT_DETECT_SIZE = 1536
DEFAULT_TEXT_THRESHOLD = 0.5
DEFAULT_BOX_THRESHOLD = 0.7
DEFAULT_UNCLIP_RATIO = 2.2

# 48px OCR 不兼容检测器（输出多行合并框的检测器）
OCR_48PX_INCOMPATIBLE_DETECTORS = {DETECTOR_YOLOV5}

# --- 重试机制设置 ---
DEFAULT_TRANSLATION_MAX_RETRIES = 3  # 普通翻译默认重试次数
DEFAULT_HQ_TRANSLATION_MAX_RETRIES = 2  # 高质量翻译默认重试次数
DEFAULT_PROOFREADING_MAX_RETRIES = 2  # AI校对默认重试次数

# --- 用户设置 ---
USER_SETTINGS_FILE = 'user_settings.json'

# --- 超长图片处理 (Large Image Rearrange) ---
# 当图片满足以下条件时自动启用切割检测:
#   1. 缩放比 (down_scale_ratio) > LARGE_IMAGE_DOWNSCALE_THRESHOLD
#   2. 长宽比 (aspect_ratio) > LARGE_IMAGE_ASPECT_THRESHOLD
LARGE_IMAGE_ENABLED = True  # 是否启用超长图片自动切割
LARGE_IMAGE_DOWNSCALE_THRESHOLD = 2.5  # 缩放比阈值
LARGE_IMAGE_ASPECT_THRESHOLD = 3.0  # 长宽比阈值
LARGE_IMAGE_TARGET_SIZE = 1536  # 切片目标尺寸（与检测器一致）