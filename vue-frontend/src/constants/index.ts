/**
 * 常量定义文件
 * 与后端 constants.js 保持一致
 */

// ============================================================
// 默认提示词常量
// ============================================================

/** 默认提示词名称 */
export const DEFAULT_PROMPT_NAME = '默认提示词'

/** 默认气泡填充颜色（白色） */
export const DEFAULT_FILL_COLOR = '#FFFFFF'

/** AI 视觉 OCR 默认提示词（普通模式） */
export const DEFAULT_AI_VISION_OCR_PROMPT = `你是一个ocr助手，你需要将我发送给你的图片中的文字提取出来并返回给我，要求：
1、完整识别：我发送给你的图片中的文字都是需要识别的内容
2、非贪婪输出：不要返回任何其他解释和说明。`

/** 翻译提示词默认值（批量翻译模式）- 与后端 BATCH_TRANSLATE_SYSTEM_TEMPLATE 保持一致 */
export const DEFAULT_TRANSLATE_PROMPT = `忽略之前的所有指令，仅遵循以下定义。

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

请将以下日文文本翻译成中文：`

/** 翻译提示词默认值（JSON格式模式）- 批量翻译 JSON 模式 */
export const DEFAULT_TRANSLATE_JSON_PROMPT = `忽略之前的所有指令，仅遵循以下定义。

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

请将以下日文文本翻译成中文：`

/** AI 视觉 OCR 默认提示词（JSON格式模式） */
export const DEFAULT_AI_VISION_OCR_JSON_PROMPT = `你是一个OCR助手。请将我发送给你的图片中的所有文字提取出来。
当文本中包含特殊字符（如大括号{}、引号""、反斜杠\\等）时，请在输出中保留它们但不要将它们视为JSON语法的一部分。如果需要，你可以使用转义字符\\来表示这些特殊字符。
请严格按照以下 JSON 格式返回结果，不要添加任何额外的解释或对话:
{
  "extracted_text": "[这里放入所有识别到的文字，可以包含换行符以大致保留原始分段，但不要包含任何其他非文本内容]"
}`

/** 高质量翻译模式默认提示词 */
export const DEFAULT_HQ_TRANSLATE_PROMPT = `你是一个漫画翻译助手，我会提供原始漫画图片和导出的只含原文不含译文JSON翻译文件，帮我将原文翻译成中文
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
11.你需要直接返回修改后的完整JSON文件，无需解释每处修改原因。`

/** AI校对的默认提示词 */
export const DEFAULT_PROOFREADING_PROMPT = `你是一个专业的漫画校对助手，请帮我校对漫画翻译结果。我会给你漫画未经翻译的原图、由漫画原文和已有的翻译文本组成的JSON文件，请根据图片内容和上下文关系，检查并改进翻译质量。
校对要点：
1. json中的"imagelndex"序号是每张图片的页码，在翻译前先根据json文件中每页图片的原文内容和所有漫画图片对我给你的所有图片进行排序，在进行上下文对比时要严格按照"imagelndex"序号进行对比，你在翻译时要按照"imagelndex"的顺序进行顺序翻译，从而使得每句翻译足够连贯，上下文不会突兀
2. 通过json中每页图片的"original"原文内容和所有的漫画图片明确每句话在那页图的哪个位置，并结合漫画图像和上下文语境，让翻译更加连贯自然，符合角色语气和场景
3. json中的"bubblelndex"标号可能不正确，你需要根据图片自行判断每句话的正确顺序，从而输出符合上下文连贯的翻译，但在输出翻译时不要修改原本的"bubblelndex"标号
4. 针对特殊术语或专有名词的翻译进行统一
5. 重点关注易错点的人称和语气词
6. 修正任何语法或表达错误
请直接返回修改后的JSON数据，保持原有格式，只需更新"translated"字段的内容。`

// ============================================================
// RPM 默认值常量
// ============================================================

/** 翻译服务默认 RPM 限制（0 表示无限制） */
export const DEFAULT_RPM_TRANSLATION = 0

/** AI 视觉 OCR 默认 RPM 限制（0 表示无限制） */
export const DEFAULT_RPM_AI_VISION_OCR = 0

/** AI 视觉 OCR 默认最小图片尺寸（像素），VLM 模型通常要求 >= 28px */
export const DEFAULT_AI_VISION_OCR_MIN_IMAGE_SIZE = 32

// ============================================================
// 自定义服务商 ID 常量
// ============================================================

/** 自定义 AI 视觉 OCR 服务商 ID（前端使用） */
export const CUSTOM_AI_VISION_PROVIDER_ID_FRONTEND = 'custom_openai_vision'

// ============================================================
// 文本描边默认值常量
// ============================================================

/** 默认启用描边 */
export const DEFAULT_STROKE_ENABLED = true

/** 默认描边颜色（白色） */
export const DEFAULT_STROKE_COLOR = '#FFFFFF'

/** 默认描边宽度 */
export const DEFAULT_STROKE_WIDTH = 3

// ============================================================
// 编辑模式字号预设常量
// ============================================================

/** 字号预设列表 */
export const FONT_SIZE_PRESETS = [16, 20, 24, 28, 32, 36, 40, 48, 56, 64]

/** 字号滑块步进值 */
export const FONT_SIZE_STEP = 2

/** 字号最小值 */
export const FONT_SIZE_MIN = 10

/** 字号最大值 */
export const FONT_SIZE_MAX = 100

/** 用户自定义字号预设存储键 */
export const FONT_SIZE_CUSTOM_PRESETS_KEY = 'customFontSizePresets'

// ============================================================
// 编辑模式视图常量
// ============================================================

/** 编辑模式视图类型 */
export const EDIT_VIEW_MODE = {
  /** 双图对照 */
  DUAL: 'dual',
  /** 仅原图 */
  ORIGINAL: 'original',
  /** 仅翻译图 */
  TRANSLATED: 'translated'
} as const

export type EditViewMode = (typeof EDIT_VIEW_MODE)[keyof typeof EDIT_VIEW_MODE]

// ============================================================
// 重试机制默认值常量
// ============================================================

/** 普通翻译默认重试次数 */
export const DEFAULT_TRANSLATION_MAX_RETRIES = 3

/** 高质量翻译默认重试次数 */
export const DEFAULT_HQ_TRANSLATION_MAX_RETRIES = 2

/** AI校对默认重试次数 */
export const DEFAULT_PROOFREADING_MAX_RETRIES = 2

// ============================================================
// 笔刷大小常量
// ============================================================

/** 笔刷最小大小 */
export const BRUSH_MIN_SIZE = 5

/** 笔刷最大大小 */
export const BRUSH_MAX_SIZE = 200

/** 笔刷默认大小 */
export const BRUSH_DEFAULT_SIZE = 30

// ============================================================
// 编辑模式事件命名空间常量
// ============================================================

/** 编辑模式事件命名空间 */
export const EDIT_MODE_EVENT_NS = '.editModeUi'

// ============================================================
// 布局模式存储键常量
// ============================================================

/** 编辑模式布局存储键 */
export const LAYOUT_MODE_KEY = 'edit_mode_layout'

// ============================================================
// localStorage 存储键常量
// ============================================================

/** 翻译设置存储键 */
export const STORAGE_KEY_TRANSLATION_SETTINGS = 'translationSettings'

/** 阅读器设置存储键 */
export const STORAGE_KEY_READER_SETTINGS = 'readerSettings'

/** 漫画笔记存储键前缀 */
export const STORAGE_KEY_MANGA_NOTES_PREFIX = 'manga_notes_'

/** 服务商配置缓存存储键 */
export const STORAGE_KEY_PROVIDER_CONFIGS = 'providerConfigs'

/** 模型使用历史存储键 */
export const STORAGE_KEY_MODEL_HISTORY = 'modelHistory'

// ============================================================
// 翻译服务商常量
// ============================================================

/**
 * 翻译服务商列表
 * 包含所有支持的翻译服务商
 */
export const TRANSLATE_PROVIDERS = [
  { value: 'siliconflow', label: 'SiliconFlow', type: 'cloud' },
  { value: 'deepseek', label: 'DeepSeek', type: 'cloud' },
  { value: 'volcano', label: '火山引擎', type: 'cloud' },
  { value: 'caiyun', label: '彩云小译', type: 'cloud' },
  { value: 'baidu_translate', label: '百度翻译', type: 'cloud' },
  { value: 'youdao_translate', label: '有道翻译', type: 'cloud' },
  { value: 'gemini', label: 'Google Gemini', type: 'cloud' },
  { value: 'ollama', label: 'Ollama (本地)', type: 'local' },
  { value: 'sakura', label: 'Sakura (本地)', type: 'local' },
  { value: 'custom_openai', label: '自定义 OpenAI 兼容服务', type: 'custom' }
] as const

/** 翻译服务商值类型 */
export type TranslateProviderValue = (typeof TRANSLATE_PROVIDERS)[number]['value']

/**
 * 高质量翻译服务商列表
 * 仅包含支持高质量翻译的服务商
 */
export const HQ_TRANSLATE_PROVIDERS = [
  { value: 'siliconflow', label: 'SiliconFlow' },
  { value: 'deepseek', label: 'DeepSeek' },
  { value: 'volcano', label: '火山引擎' },
  { value: 'gemini', label: 'Google Gemini' },
  { value: 'custom_openai', label: '自定义 OpenAI 兼容服务' }
] as const

/** 高质量翻译服务商值类型 */
export type HqTranslateProviderValue = (typeof HQ_TRANSLATE_PROVIDERS)[number]['value']

/**
 * 本地翻译服务商列表
 * 不需要 API Key 的本地服务
 */
export const LOCAL_TRANSLATE_PROVIDERS = ['ollama', 'sakura'] as const

/**
 * 不支持 RPM 限制的服务商列表
 * 这些服务商不需要设置请求频率限制
 */
export const NO_RPM_LIMIT_PROVIDERS = [
  'ollama',
  'sakura',
  'caiyun',
  'baidu_translate',
  'youdao_translate'
] as const

/**
 * 支持获取模型列表的服务商
 * 这些服务商可以通过 API 获取可用模型列表
 */
export const SUPPORTS_FETCH_MODELS_PROVIDERS = [
  'siliconflow',
  'deepseek',
  'volcano',
  'gemini',
  'custom_openai'
] as const

// ============================================================
// OCR 引擎常量
// ============================================================

/**
 * OCR 引擎列表
 * 包含所有支持的 OCR 引擎
 */
export const OCR_ENGINES = [
  { value: 'manga_ocr', label: 'MangaOCR (本地)', type: 'local', description: '日语漫画专用' },
  { value: 'paddle_ocr', label: 'PaddleOCR (本地)', type: 'local', description: '多语言支持' },
  { value: 'baidu_ocr', label: '百度OCR (云端)', type: 'cloud', description: '需要 API Key' },
  { value: 'ai_vision', label: 'AI视觉OCR (云端)', type: 'cloud', description: '支持多服务商' }
] as const

/** OCR 引擎值类型 */
export type OcrEngineValue = (typeof OCR_ENGINES)[number]['value']

/**
 * 文字检测器列表
 * 用于检测图片中的文字区域
 */
export const TEXT_DETECTORS = [
  { value: 'ctd', label: 'CTD (Comic Text Detector)' },
  { value: 'yolo', label: 'YOLO' },
  { value: 'yolov5', label: 'YOLOv5' },
  { value: 'default', label: 'Default (DBNet)' }
] as const

/** 文字检测器值类型 */
export type TextDetectorValue = (typeof TEXT_DETECTORS)[number]['value']

/**
 * AI 视觉 OCR 服务商列表
 * 支持 AI 视觉 OCR 的云服务商
 */
export const AI_VISION_OCR_PROVIDERS = [
  { value: 'siliconflow', label: 'SiliconFlow' },
  { value: 'deepseek', label: 'DeepSeek' },
  { value: 'volcano', label: '火山引擎' },
  { value: 'gemini', label: 'Google Gemini' },
  { value: 'custom_openai_vision', label: '自定义 OpenAI 兼容服务' }
] as const

/** AI 视觉 OCR 服务商值类型 */
export type AiVisionOcrProviderValue = (typeof AI_VISION_OCR_PROVIDERS)[number]['value']

// ============================================================
// 取消思考方法常量
// ============================================================

/**
 * 取消思考方法列表
 * 用于高质量翻译和 AI 校对的低推理模式
 */
export const NO_THINKING_METHODS = [
  { value: 'gemini', label: 'Gemini风格 (reasoning_effort=low)' },
  { value: 'volcano', label: '火山引擎风格 (thinking=null)' }
] as const

/** 取消思考方法值类型 */
export type NoThinkingMethodValue = (typeof NO_THINKING_METHODS)[number]['value']

// ============================================================
// 网页导入常量
// ============================================================

/** 网页导入设置存储键 */
export const STORAGE_KEY_WEB_IMPORT_SETTINGS = 'webImportSettings'

/** 网页导入默认提取提示词 */
export const DEFAULT_WEB_IMPORT_EXTRACTION_PROMPT = `你是一个专业的漫画数据提取助手。请针对当前网页执行以下提取任务:

## 1. 交互行为
- 请模拟用户行为，缓慢向下滚动页面至底部，以触发所有采用"懒加载"技术的漫画图片。
- 在滚动过程中，请确保等待图片加载完成，识别并提取真实的漫画内容图片。

## 2. 提取逻辑
- **图片过滤**: 忽略所有加载占位图（如 loading.gif、spacer.gif）、广告图或图标，仅提取属于漫画正文的图片。
- **属性识别**: 优先提取 \`data-src\`、\`data-original\`、\`original\` 或 \`file\` 等包含真实高清原图地址的属性。如果这些属性不存在，再提取 \`src\` 属性。
- **元数据**: 提取漫画的名称（comic_title）和当前章节的名称（chapter_title）。

## 3. 数据结构
- 必须按图片在页面中显示的先后顺序提取，并为每张图片分配一个从 1 开始的 \`page_number\`（页码序号）。
- 最终结果以 JSON 格式输出，包含漫画名称、章节名以及包含序号和图片链接的列表。

## 4. 输出格式 (Valid JSON Only)
严格按照以下 JSON 格式输出，不要包含 Markdown 代码块标记（如 \\\`\\\`\\\`json）：

{
  "comic_title": "漫画名称",
  "chapter_title": "第X话 章节标题",
  "pages": [
    {"page_number": 1, "image_url": "https://..."},
    {"page_number": 2, "image_url": "https://..."}
  ],
  "total_pages": 1
}`

/**
 * 网页导入 AI Agent 服务商列表
 */
export const WEB_IMPORT_AGENT_PROVIDERS = [
  { value: 'openai', label: 'OpenAI' },
  { value: 'siliconflow', label: 'SiliconFlow' },
  { value: 'deepseek', label: 'DeepSeek' },
  { value: 'volcano', label: '火山引擎' },
  { value: 'gemini', label: 'Google Gemini' },
  { value: 'custom_openai', label: '自定义 OpenAI 兼容服务' }
] as const
