"""
Card Agent helpers.
"""

from __future__ import annotations

from typing import Any, Dict, List


def build_agent_context(document: Dict[str, Any], preview_session: Dict[str, Any]) -> str:
    identity = document.get("identity", {})
    core = document.get("coreMessages", {})
    lorebook = document.get("lorebook", {}).get("entries", []) or []
    regex_scripts = document.get("regexScripts", []) or []
    state_tasks = document.get("stateTasks", []) or []
    preview_messages = preview_session.get("messages", [])[-6:]
    variables = preview_session.get("variables", {})

    parts = [
        f"角色: {identity.get('name', '')}",
        f"描述: {identity.get('description', '')[:300]}",
        f"人格: {identity.get('personality', '')[:200]}",
        f"场景: {identity.get('scenario', '')[:200]}",
        f"主问候: {core.get('first_message', '')[:240]}",
        f"备用问候数: {len(core.get('alternate_greetings', []) or [])}",
        f"世界书条目数: {len(lorebook)}",
        f"正则脚本数: {len(regex_scripts)}",
        f"状态任务数: {len(state_tasks)}",
        f"当前预览变量: {variables}",
    ]
    if preview_messages:
        parts.append("最近对话:")
        for msg in preview_messages:
            parts.append(f"- {msg.get('role')}: {str(msg.get('content', ''))[:200]}")
    parts.append(
        "请优先返回简洁建议；如需修改角色卡，请输出 ```json:patch 代码块，"
        "格式形如 {\"set\": {\"identity.description\": \"...\"}, \"greeting_add\": \"...\"}。"
    )
    return "\n".join(parts)
