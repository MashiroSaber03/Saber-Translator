"""Strict loader for factory prompt text shared by the frontend and backend."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


PROMPT_FACTORY_DEFAULTS_PATH = Path(__file__).with_name(
    "prompt_defaults_factory.json"
)

_REQUIRED_KEYS = frozenset(
    {
        "singleNormal",
        "singleJson",
        "batchNormal",
        "batchJson",
        "aiVisionOcrNormal",
        "aiVisionOcrJson",
        "hqTranslation",
        "proofreading",
        "autoGlossary",
        "webImportExtraction",
    }
)


@lru_cache(maxsize=1)
def load_prompt_factory_defaults() -> dict[str, str]:
    try:
        raw = json.loads(PROMPT_FACTORY_DEFAULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("prompt_defaults_factory.json 无法读取") from exc
    if not isinstance(raw, dict) or set(raw) != _REQUIRED_KEYS:
        raise RuntimeError("prompt_defaults_factory.json 字段无效")
    if any(not isinstance(value, str) or not value for value in raw.values()):
        raise RuntimeError("prompt_defaults_factory.json 提示词必须是非空字符串")
    return dict(raw)


def get_prompt_factory_defaults() -> dict[str, str]:
    return dict(load_prompt_factory_defaults())
