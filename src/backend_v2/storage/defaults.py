"""Factory-owned v2 settings and prompt defaults."""

from __future__ import annotations

from src.shared.constants import (
    BATCH_TRANSLATE_JSON_SYSTEM_TEMPLATE,
    BATCH_TRANSLATE_SYSTEM_TEMPLATE,
    DEFAULT_AI_VISION_OCR_PROMPT,
    DEFAULT_HQ_TRANSLATE_PROMPT,
    DEFAULT_PROMPT,
    DEFAULT_PROOFREADING_PROMPT,
    DEFAULT_TRANSLATE_JSON_PROMPT,
    DEFAULT_WEB_IMPORT_EXTRACTION_PROMPT,
)


DEFAULT_FONT_ID = "00000000-0000-0000-0000-000000000010"
TRANSLATION_SETTINGS_SCHEMA_VERSION = 8
TEXT_STYLE_DEFAULTS_SCHEMA_VERSION = 2

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
    "inlineAlign": "start",
    "blockAlign": "start",
}

DEFAULT_WORKFLOW_PREFERENCES: dict[str, object] = {
    "rememberWorkflowModeEnabled": False,
    "lastWorkflowMode": "translate-current",
}

DEFAULT_EXPORT_PREFERENCES: dict[str, object] = {
    "preserveOriginalFilenames": False,
}

DEFAULT_CUSTOM_AI_PROFILES: dict[str, object] = {
    "profiles": [],
}

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
    "translate": BATCH_TRANSLATE_SYSTEM_TEMPLATE,
    "textbox": (
        "将输入内容翻译为简体中文，并简要说明关键语法、语气与译法。"
    ),
    "ai_vision_ocr": DEFAULT_AI_VISION_OCR_PROMPT,
    "hq_translate": DEFAULT_HQ_TRANSLATE_PROMPT,
    "proofreading": DEFAULT_PROOFREADING_PROMPT,
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
    """Return the complete current browser settings document.

    The v2 database is the first source of truth, so a fresh database must
    contain a document the strict frontend parser can hydrate directly.
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
        "settingsSchemaVersion": TRANSLATION_SETTINGS_SCHEMA_VERSION,
        "ocrEngine": "manga_ocr",
        "textDetector": "default",
        "minTextBlockAreaPercent": 0.05,
        "enableAuxYoloDetection": False,
        "auxYoloConfThreshold": 0.4,
        "auxYoloOverlapThreshold": 0.1,
        "enableSaberYoloRefine": True,
        "saberYoloRefineOverlapThreshold": 50.0,
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
            "batchNormalPrompt": BATCH_TRANSLATE_SYSTEM_TEMPLATE,
            "batchJsonPrompt": BATCH_TRANSLATE_JSON_SYSTEM_TEMPLATE,
            "singleNormalPrompt": DEFAULT_PROMPT,
            "singleJsonPrompt": DEFAULT_TRANSLATE_JSON_PROMPT,
        },
        "targetLanguage": "zh",
        "translatePrompt": BATCH_TRANSLATE_SYSTEM_TEMPLATE,
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
                transport_retries=1,
                business_retries=0,
            ),
        },
        "proofreading": {
            "enabled": False,
            "rounds": [],
        },
        "boxExpand": {
            "ratio": 0.0,
            "top": 0.0,
            "bottom": 0.0,
            "left": 0.0,
            "right": 0.0,
        },
        "preciseMask": {"dilateSize": 10, "boxExpandRatio": 20.0},
        "showDetectionDebug": False,
        "parallel": {"enabled": False, "deepLearningLockSize": 1},
        "removeTextWithOcr": False,
        "compressVisionImages": True,
        "lamaDisableResize": False,
    }
