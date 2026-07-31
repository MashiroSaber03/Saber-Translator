"""seed complete backend-owned Web Import and Insight settings

Revision ID: 0016
Revises: 0015
Create Date: 2026-07-31
"""

from __future__ import annotations

import json
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0016"
down_revision: Union[str, Sequence[str], None] = "0015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_WEB_IMPORT_PROMPT = """你是一个专业的漫画数据提取助手。请针对当前网页执行以下提取任务:

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

_DEFAULTS = {
    "web_import": {
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
            "prompt": _WEB_IMPORT_PROMPT,
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
    },
    "insight": {
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
    },
}


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _merge_known_fields(default: object, current: object) -> object:
    """Fill missing fields while discarding the old browser-only settings keys."""

    if not isinstance(default, dict):
        return current
    current_object = current if isinstance(current, dict) else {}
    return {
        key: (
            _merge_known_fields(default_value, current_object[key])
            if key in current_object
            else default_value
        )
        for key, default_value in default.items()
    }


def upgrade() -> None:
    connection = op.get_bind()
    for domain, payload in _DEFAULTS.items():
        row = connection.execute(
            sa.text(
                "SELECT payload_json FROM app_settings WHERE domain=:domain"
            ),
            {"domain": domain},
        ).scalar_one_or_none()
        if row is None:
            connection.execute(
                sa.text(
                    "INSERT INTO app_settings "
                    "(domain, revision, payload_json, schema_version) "
                    "VALUES (:domain, 1, :payload, 1)"
                ),
                {"domain": domain, "payload": _json(payload)},
            )
            continue
        try:
            current = json.loads(str(row))
        except (TypeError, ValueError):
            current = {}
        normalized = _merge_known_fields(payload, current)
        connection.execute(
            sa.text(
                "UPDATE app_settings SET payload_json=:payload "
                "WHERE domain=:domain"
            ),
            {"domain": domain, "payload": _json(normalized)},
        )


def downgrade() -> None:
    connection = op.get_bind()
    for domain, payload in _DEFAULTS.items():
        connection.execute(
            sa.text(
                "UPDATE app_settings SET payload_json='{}' "
                "WHERE domain=:domain AND payload_json=:payload"
            ),
            {"domain": domain, "payload": _json(payload)},
        )
