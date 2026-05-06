"""
插件 v2 契约定义。

围绕 7 个原子翻译步骤外加一对 pipeline 全局生命周期钩子工作。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


PLUGIN_STEPS: Tuple[str, ...] = (
    "detect",
    "ocr",
    "color",
    "translate",
    "ai_translate",
    "inpaint",
    "render",
    "pipeline",
)

PLUGIN_MODES: Tuple[str, ...] = (
    "standard",
    "hq",
    "proofread",
    "remove_text",
)

FAILURE_POLICIES: Tuple[str, ...] = (
    "continue",
    "fail",
)

STEP_HOOK_METHODS: Dict[str, Dict[str, str]] = {
    "detect": {
        "before": "before_detect",
        "after": "after_detect",
    },
    "ocr": {
        "before": "before_ocr",
        "after": "after_ocr",
    },
    "color": {
        "before": "before_color",
        "after": "after_color",
    },
    "translate": {
        "before": "before_translate",
        "after": "after_translate",
    },
    "ai_translate": {
        "before": "before_ai_translate",
        "after": "after_ai_translate",
    },
    "inpaint": {
        "before": "before_inpaint",
        "after": "after_inpaint",
    },
    "render": {
        "before": "before_render",
        "after": "after_render",
    },
    "pipeline": {
        "before": "before_pipeline",
        "after": "after_pipeline",
    },
}

HOOK_METHOD_TO_STEP_PHASE: Dict[str, Tuple[str, str]] = {
    method_name: (step, phase)
    for step, phase_mapping in STEP_HOOK_METHODS.items()
    for phase, method_name in phase_mapping.items()
}


def normalize_plugin_mode(mode: str | None) -> str:
    if not mode:
        return "standard"

    normalized = str(mode).strip().lower()
    alias_mapping = {
        "remove-text": "remove_text",
        "removetext": "remove_text",
        "ai-translate": "ai_translate",
        "aitranslate": "ai_translate",
    }
    return alias_mapping.get(normalized, normalized.replace("-", "_"))


def normalize_plugin_step(step: str) -> str:
    normalized = str(step or "").strip().lower()
    alias_mapping = {
        "aitranslate": "ai_translate",
        "ai-translate": "ai_translate",
    }
    return alias_mapping.get(normalized, normalized.replace("-", "_"))


@dataclass
class PluginContext:
    step: str
    mode: str
    route: str
    scope: str
    metadata: Dict[str, Any] = field(default_factory=dict)
