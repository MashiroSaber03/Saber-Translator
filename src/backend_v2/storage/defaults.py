"""Factory-owned v2 settings and prompt defaults."""

from __future__ import annotations


DEFAULT_FONT_ID = "00000000-0000-0000-0000-000000000010"

DEFAULT_TEXT_STYLE: dict[str, object] = {
    "fontSize": 26,
    "autoFontSize": True,
    "fontFamily": DEFAULT_FONT_ID,
    "layoutDirection": "auto",
    "textColor": "#000000",
    "fillColor": "#FFFFFF",
    "inpaintMethod": "solid",
    "useAutoTextColor": False,
    "strokeEnabled": True,
    "strokeColor": "#FFFFFF",
    "strokeWidth": 3,
    "lineSpacing": 1.0,
    "textAlign": "start",
}

DEFAULT_WORKFLOW_PREFERENCES: dict[str, object] = {
    "rememberWorkflowModeEnabled": False,
    "lastWorkflowMode": "translate-current",
}

DEFAULT_WEB_IMPORT_EXTRACTION_PROMPT = """你是一个专业的漫画数据提取助手。请针对当前网页执行以下提取任务:

## 1. 交互行为
- 请模拟用户行为，缓慢向下滚动页面至底部，以触发所有采用"懒加载"技术的漫画图片。
- 在滚动过程中，请确保等待图片加载完成，识别并提取真实的漫画内容图片。

## 2. 提取逻辑
- **图片过滤**: 忽略所有加载占位图（如 loading.gif、spacer.gif）、广告图或图标，仅提取属于漫画正文的图片。
- **属性识别**: 优先提取 `data-src`、`data-original`、`original` 或 `file` 等包含真实高清原图地址的属性。如果这些属性不存在，再提取 `src` 属性。
- **元数据**: 提取漫画的名称（comic_title）和当前章节的名称（chapter_title）。

## 3. 数据结构
- 必须按图片在页面中显示的先后顺序提取，并为每张图片分配一个从 1 开始的 `page_number`（页码序号）。
- 最终结果以 JSON 格式输出，包含漫画名称、章节名以及包含序号和图片链接的列表。

## 4. 输出格式 (Valid JSON Only)
严格按照以下 JSON 格式输出，不要包含 Markdown 代码块标记（如 ```json）：

{
  "comic_title": "漫画名称",
  "chapter_title": "第X话 章节标题",
  "pages": [
    {"page_number": 1, "image_url": "https://..."},
    {"page_number": 2, "image_url": "https://..."}
  ],
  "total_pages": 1
}"""

DEFAULT_WEB_IMPORT_SETTINGS: dict[str, object] = {
    "firecrawl": {},
    "agent": {
        "provider": "openai",
        "customBaseUrl": "",
        "modelName": "gpt-4o-mini",
        "useStream": False,
        "forceJsonOutput": True,
        "maxRetries": 3,
        "timeout": 120,
    },
    "extraction": {
        "prompt": DEFAULT_WEB_IMPORT_EXTRACTION_PROMPT,
        "maxIterations": 10,
    },
    "download": {
        "concurrency": 3,
        "timeout": 30,
        "retries": 3,
        "delay": 100,
        "useReferer": True,
    },
    "imagePreprocess": {
        "enabled": False,
        "autoRotate": True,
        "compression": {
            "enabled": False,
            "quality": 85,
            "maxWidth": 0,
            "maxHeight": 0,
        },
        "formatConvert": {
            "enabled": False,
            "targetFormat": "original",
        },
    },
    "advanced": {"bypassProxy": False},
    "ui": {
        "showAgentLogs": True,
        "autoImport": False,
    },
}

DEFAULT_INSIGHT_SETTINGS: dict[str, object] = {
    "analysis": {
        "batch": {
            "pagesPerBatch": 5,
            "contextBatchCount": 3,
            "architecturePreset": "standard",
            "customLayers": [],
        }
    },
    "vlm": {"provider": "gemini"},
    "chat": {"provider": "gemini", "useSameAsVlm": False},
    "embedding": {"provider": "openai"},
    "reranker": {"provider": "jina"},
    "imageGen": {"provider": "gpt2api"},
}

FACTORY_PROMPTS: dict[str, str] = {
    "translate": (
        "将输入内容准确、自然地翻译为简体中文，仅输出译文；"
        "保持角色语气、上下文与专有名词一致。"
    ),
    "textbox": (
        "将输入内容翻译为简体中文，并简要说明关键语法、语气与译法。"
    ),
    "ai_vision_ocr": (
        "按阅读顺序识别漫画图片中的文字；不要臆造不可见内容。"
    ),
    "hq_translate": (
        "结合连续漫画页面、原文和既有术语，将内容翻译为自然的简体中文；"
        "保持跨页上下文和角色口吻一致。"
    ),
    "proofreading": (
        "结合原图、原文和既有译文进行校对，只修正准确性、流畅性、"
        "术语一致性与角色语气问题。"
    ),
    "batch_analysis": (
        "分析漫画页面，输出符合给定 schema 的摘要、关键事件、连续性与警告。"
    ),
    "segment_summary": (
        "汇总给定页面段落，保留事件因果、角色状态变化与未解决线索。"
    ),
    "chapter_summary": (
        "汇总章节内容，保持事件顺序，并指出关键转折与连续性信息。"
    ),
    "book_overview": (
        "基于已发布分析生成全书概览，不添加输入中没有依据的事实。"
    ),
    "group_summary": (
        "将下级分析结果归并为结构化摘要，保留来源覆盖范围和警告。"
    ),
    "qa_response": (
        "只依据检索到的漫画分析上下文回答问题；证据不足时明确说明。"
    ),
    "question_decompose": (
        "把用户问题拆成便于检索的独立子问题，并保留原始意图。"
    ),
    "analysis_system": (
        "你是漫画分析助手。所有结论必须来自当前分析资料，区分事实与推断。"
    ),
}


def default_translation_settings() -> dict[str, object]:
    """Return the complete browser schema-v3 factory document.

    The v2 database is the first source of truth, so a fresh database must
    contain a document the strict frontend parser can hydrate without falling
    back to a second browser-owned settings source.
    """

    def openai_options(
        *,
        use_stream: bool,
        rpm_limit: int,
        transport_retries: int,
        business_retries: int,
    ) -> dict[str, object]:
        return {
            "request": {"forceJsonOutput": False},
            "execution": {
                "useStream": use_stream,
                "rpmLimit": rpm_limit,
                "transportRetries": transport_retries,
                "businessRetries": business_retries,
            },
        }

    return {
        "settingsSchemaVersion": 3,
        "ocrEngine": "manga_ocr",
        "sourceLanguage": "japanese",
        "textDetector": "default",
        "minTextBlockAreaPercent": 0.05,
        "enableAuxYoloDetection": False,
        "auxYoloConfThreshold": 0.4,
        "auxYoloOverlapThreshold": 0.1,
        "enableSaberYoloRefine": True,
        "saberYoloRefineOverlapThreshold": 50,
        "baiduOcr": {
            "version": "standard",
            "sourceLanguage": "JAP",
        },
        "paddleOcrVl": {"sourceLanguage": "japanese"},
        "aiVisionOcr": {
            "provider": "gemini",
            "modelName": "",
            "prompt": FACTORY_PROMPTS["ai_vision_ocr"],
            "promptMode": "normal",
            "customBaseUrl": "",
            "openaiOptions": openai_options(
                use_stream=False,
                rpm_limit=0,
                transport_retries=1,
                business_retries=3,
            ),
            "minImageSize": 32,
        },
        "hybridOcr": {
            "enabled": False,
            "secondaryEngine": "48px_ocr",
            "confidenceThreshold": 0.2,
        },
        "translation": {
            "provider": "siliconflow",
            "modelName": "",
            "customBaseUrl": "",
            "openaiOptions": openai_options(
                use_stream=True,
                rpm_limit=0,
                transport_retries=1,
                business_retries=3,
            ),
            "translationMode": "batch",
            "batchNormalPrompt": FACTORY_PROMPTS["translate"],
            "batchJsonPrompt": FACTORY_PROMPTS["translate"],
            "singleNormalPrompt": FACTORY_PROMPTS["translate"],
            "singleJsonPrompt": FACTORY_PROMPTS["translate"],
        },
        "targetLanguage": "zh",
        "translatePrompt": FACTORY_PROMPTS["translate"],
        "useTextboxPrompt": False,
        "textboxPrompt": "",
        "hqTranslation": {
            "provider": "siliconflow",
            "modelName": "",
            "customBaseUrl": "",
            "openaiOptions": openai_options(
                use_stream=True,
                rpm_limit=7,
                transport_retries=3,
                business_retries=3,
            ),
            "batchSize": 3,
            "prompt": FACTORY_PROMPTS["hq_translate"],
        },
        "pluginAgent": {
            "provider": "siliconflow",
            "modelName": "",
            "customBaseUrl": "",
            "openaiOptions": openai_options(
                use_stream=True,
                rpm_limit=0,
                transport_retries=10,
                business_retries=10,
            ),
        },
        "proofreading": {
            "enabled": False,
            "rounds": [],
            "maxRetries": 2,
        },
        "boxExpand": {"ratio": 0, "top": 0, "bottom": 0, "left": 0, "right": 0},
        "preciseMask": {"dilateSize": 10, "boxExpandRatio": 20},
        "showDetectionDebug": False,
        "parallel": {"enabled": False, "deepLearningLockSize": 1},
        "removeTextWithOcr": False,
        "enableVerboseLogs": False,
        "lamaDisableResize": False,
    }
