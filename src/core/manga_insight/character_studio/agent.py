"""
Card Agent helpers.
"""

from __future__ import annotations

import json
from typing import Any, Dict


def build_agent_context(document: Dict[str, Any], preview_session: Dict[str, Any], compressed_context: str) -> str:
    identity = document.get("identity", {})
    core = document.get("coreMessages", {})
    preview_messages = preview_session.get("messages", [])[-6:]
    variables = preview_session.get("variables", {})
    runtime_log = preview_session.get("log", [])[-8:]

    parts = [
        "【事实层】",
        f"目标角色名: {identity.get('name', '')}",
        "压缩摘要（唯一外部事实来源）:",
        compressed_context,
        "",
        "【当前角色卡】",
        "以下内容仅作一致性、保留内容与 patch 建议参考，不可当作新的事实来源。",
        json.dumps({
            "identity": identity,
            "coreMessages": core,
            "lorebook": document.get("lorebook", {}),
            "regexScripts": document.get("regexScripts", []),
            "stateTasks": document.get("stateTasks", []),
        }, ensure_ascii=False, indent=2),
        "",
        "【运行时线索】",
        f"当前预览变量: {variables}",
    ]
    if preview_messages:
        parts.append("最近对话:")
        for msg in preview_messages:
            parts.append(f"- {msg.get('role')}: {str(msg.get('content', ''))[:200]}")
    if runtime_log:
        parts.append("最近命中/执行日志:")
        for item in runtime_log:
            parts.append(f"- {json.dumps(item, ensure_ascii=False)}")
    parts.append(
        "请优先返回简洁建议；如需修改角色卡，请输出 ```json:patch 代码块，"
        "格式形如 {\"set\": {\"identity.description\": \"...\"}, \"greeting_add\": \"...\"}。"
    )
    return "\n".join(parts)
