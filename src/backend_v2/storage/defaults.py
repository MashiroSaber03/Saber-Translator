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
        "textStyle": dict(DEFAULT_TEXT_STYLE),
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
            "apiKey": "",
            "secretKey": "",
            "version": "standard",
            "sourceLanguage": "JAP",
        },
        "paddleOcrVl": {"sourceLanguage": "japanese"},
        "aiVisionOcr": {
            "provider": "gemini",
            "apiKey": "",
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
            "apiKey": "",
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
            "apiKey": "",
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
            "apiKey": "",
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
