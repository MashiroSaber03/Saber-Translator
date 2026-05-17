"""
Card Agent helpers.
"""

from __future__ import annotations

import json
from typing import Any, Dict


def build_agent_context(document: Dict[str, Any], session_state: Dict[str, Any], compressed_context: str) -> str:
    identity = document.get("identity", {})
    core = document.get("coreMessages", {})
    recent_messages = session_state.get("messages", [])[-6:]
    variables = session_state.get("variables", {})
    runtime_log = session_state.get("log", [])[-8:]

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
        f"当前会话变量: {variables}",
    ]
    if recent_messages:
        parts.append("最近对话:")
        for msg in recent_messages:
            parts.append(f"- {msg.get('role')}: {str(msg.get('content', ''))[:200]}")
    if runtime_log:
        parts.append("最近命中/执行日志:")
        for item in runtime_log:
            parts.append(f"- {json.dumps(item, ensure_ascii=False)}")
    parts.append(
        "请优先返回简洁建议；如需修改角色卡，请输出 ```json:patch 代码块。"
        "支持的 patch 字段如下："
        "1) set: 用于直接设置已有字段路径，例如 identity.description、identity.name、coreMessages.first_message、coreMessages.system_prompt、lorebook.name。不要用 set 修改 lorebook.entries、regexScripts、stateTasks 或它们的数组项。"
        "2) greeting_add: 追加一条或多条 alternate_greetings，可使用单个字符串或字符串数组。"
        "3) worldbook_add: 追加世界书条目，对象字段建议为 comment、keys、content，可选 secondary_keys、position、priority、depth。keys 必须是字符串数组；position 只能使用 before_char、at_depth、after_char。"
        "4) worldbook_update: 更新已有世界书条目，必须使用 {id, changes} 结构；id 必须来自当前角色卡已有条目的稳定 id。changes 仅允许 comment、keys、secondary_keys、content、enabled、constant、selective、priority、position、depth、probability、prevent_recursion、use_regex、match_persona_description、match_character_description、match_character_personality、match_character_depth_prompt、match_scenario。position 只能使用 before_char、at_depth、after_char。"
        "5) worldbook_delete: 删除已有世界书条目，必须使用 {id} 结构；id 必须来自当前角色卡已有条目的稳定 id。"
        "6) regex_add: 追加正则脚本，对象字段必须使用 scriptName、findRegex、replaceString，可选 placement、markdownOnly、promptOnly、runOnEdit、disabled。placement 只能使用 1、2 或 [1,2]；1 表示用户输入/提示文本侧，2 表示角色回复/显示文本侧。不要使用 name、regex、replacement、condition。"
        "7) regex_update: 更新已有正则脚本，必须使用 {id, changes} 结构；id 必须来自当前角色卡已有条目的稳定 id。changes 仅允许 scriptName、findRegex、replaceString、placement、markdownOnly、promptOnly、runOnEdit、disabled。placement 只能使用 1、2 或 [1,2]。"
        "8) regex_delete: 删除已有正则脚本，必须使用 {id} 结构；id 必须来自当前角色卡已有条目的稳定 id。"
        "9) task_add: 追加状态任务，对象字段必须使用 name、triggerTiming、commands，可选 interval、disabled。triggerTiming 只能是 initialization、message_received、message_sent。commands 应为可执行字符串，推荐 <<taskjs>> ... <</taskjs>>。"
        "10) task_update: 更新已有状态任务，必须使用 {id, changes} 结构；id 必须来自当前角色卡已有条目的稳定 id。changes 仅允许 name、triggerTiming、interval、commands、disabled。triggerTiming 只能是 initialization、message_received、message_sent。"
        "11) task_delete: 删除已有状态任务，必须使用 {id} 结构；id 必须来自当前角色卡已有条目的稳定 id。"
        "如需一次修改多条同类条目，可以把 greeting_add、worldbook_add/worldbook_update/worldbook_delete、regex_add/regex_update/regex_delete、task_add/task_update/task_delete 写成数组。"
        "进行 update/delete 时，必须使用现有条目的稳定 id，不要使用数组索引。"
        "不要输出其他 patch 顶层字段。"
    )
    return "\n".join(parts)
