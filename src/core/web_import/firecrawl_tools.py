"""Current Firecrawl scrape tool used by the webpage-import Agent."""

from __future__ import annotations

import logging
import math
from typing import Any
from urllib.parse import urlparse

import httpx

from src.shared.memory_errors import is_memory_allocation_error


logger = logging.getLogger("WebImport.FirecrawlTools")
FIRECRAWL_API_BASE = "https://api.firecrawl.dev/v2"
_FORMATS = {"markdown", "html", "screenshot", "links"}

FIRECRAWL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "firecrawl_scrape",
            "description": (
                "抓取当前漫画网页的 HTML、Markdown、链接或截图；"
                "需要动态加载时可执行等待、点击或滚动动作。"
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "要抓取的 HTTP(S) 网页 URL",
                    },
                    "formats": {
                        "type": "array",
                        "items": {"type": "string", "enum": sorted(_FORMATS)},
                        "default": ["markdown", "html"],
                    },
                    "wait_for": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                        "description": "等待页面加载的毫秒数",
                    },
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["wait", "click", "scroll", "screenshot"],
                                },
                                "milliseconds": {"type": "integer", "minimum": 0},
                                "selector": {"type": "string"},
                                "direction": {
                                    "type": "string",
                                    "enum": ["up", "down"],
                                },
                                "amount": {"type": "integer", "minimum": 1},
                            },
                            "required": ["type"],
                        },
                    },
                },
                "required": ["url"],
            },
        },
    }
]


def execute_firecrawl_tool_sync(
    tool_name: str,
    tool_args: dict[str, Any],
    api_key: str,
    timeout: float,
    bypass_proxy: bool,
) -> dict[str, Any]:
    """Execute one scrape call and return a canonical tool result."""

    try:
        if tool_name != "firecrawl_scrape":
            raise ValueError(f"未知的 Firecrawl 工具: {tool_name}")
        payload = _scrape_payload(tool_args)
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Firecrawl API Key 不能为空")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("Firecrawl 超时时间必须是正有限数")
        if not isinstance(bypass_proxy, bool):
            raise ValueError("Firecrawl 代理开关必须是布尔值")
    except (TypeError, ValueError) as exc:
        return {"error": str(exc)}

    try:
        with httpx.Client(
            timeout=float(timeout),
            trust_env=not bypass_proxy,
        ) as client:
            response = client.post(
                f"{FIRECRAWL_API_BASE}/scrape",
                headers={
                    "Authorization": f"Bearer {api_key.strip()}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        logger.warning("Firecrawl 请求失败: HTTP %s", status)
        return {"error": f"Firecrawl HTTP {status}"}
    except Exception as exc:
        if is_memory_allocation_error(exc):
            raise
        logger.warning("Firecrawl 请求失败: %s", type(exc).__name__)
        return {"error": f"Firecrawl 请求失败: {type(exc).__name__}"}

    if (
        not isinstance(result, dict)
        or result.get("success") is not True
        or not isinstance(result.get("data"), dict)
    ):
        return {"error": "Firecrawl 返回格式无效"}
    return {"success": True, "data": result["data"]}


def _scrape_payload(tool_args: object) -> dict[str, Any]:
    if not isinstance(tool_args, dict):
        raise TypeError("Firecrawl 工具参数必须是对象")
    allowed = {"url", "formats", "wait_for", "actions"}
    if set(tool_args) - allowed:
        raise ValueError("Firecrawl 工具参数包含未知字段")

    url = tool_args.get("url")
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Firecrawl url 必须是非空字符串")
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Firecrawl url 必须是 HTTP(S) URL")

    formats = tool_args.get("formats", ["markdown", "html"])
    if (
        not isinstance(formats, list)
        or not formats
        or any(not isinstance(value, str) or value not in _FORMATS for value in formats)
        or len(set(formats)) != len(formats)
    ):
        raise ValueError("Firecrawl formats 字段无效")

    wait_for = tool_args.get("wait_for", 0)
    if isinstance(wait_for, bool) or not isinstance(wait_for, int) or wait_for < 0:
        raise ValueError("Firecrawl wait_for 必须是非负整数")

    payload: dict[str, Any] = {
        "url": url.strip(),
        "formats": formats,
    }
    if wait_for:
        payload["waitFor"] = wait_for
    if "actions" in tool_args:
        payload["actions"] = _actions(tool_args["actions"])
    return payload


def _actions(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("Firecrawl actions 必须是数组")
    result: list[dict[str, Any]] = []
    fields = {
        "wait": {"type", "milliseconds"},
        "click": {"type", "selector"},
        "scroll": {"type", "direction", "amount"},
        "screenshot": {"type"},
    }
    for action in value:
        if not isinstance(action, dict) or not isinstance(action.get("type"), str):
            raise ValueError("Firecrawl action 必须是带 type 的对象")
        action_type = action["type"]
        expected = fields.get(action_type)
        if expected is None or set(action) != expected:
            raise ValueError(f"Firecrawl {action_type} action 字段无效")
        if action_type == "wait":
            milliseconds = action["milliseconds"]
            if (
                isinstance(milliseconds, bool)
                or not isinstance(milliseconds, int)
                or milliseconds < 0
            ):
                raise ValueError("Firecrawl wait action 毫秒数无效")
        elif action_type == "click":
            if not isinstance(action["selector"], str) or not action["selector"].strip():
                raise ValueError("Firecrawl click action 选择器无效")
        elif action_type == "scroll":
            amount = action["amount"]
            if action["direction"] not in {"up", "down"}:
                raise ValueError("Firecrawl scroll action 方向无效")
            if isinstance(amount, bool) or not isinstance(amount, int) or amount < 1:
                raise ValueError("Firecrawl scroll action 距离无效")
        result.append(dict(action))
    return result
