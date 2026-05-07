from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


_SKILL_PATH = Path(__file__).with_name("plugin_builder_skill.md")

PLUGIN_AGENT_OVERVIEW = [
    "插件只能作用在既定 hook 节点，不能修改项目其他模块。",
    "新建插件默认 `default_enabled = False`。",
    "修改 OCR 文本时应同步 `ocr_results[i][\"text\"]`。",
    "优先使用 canonical step 和 mode 名称，不要发明别名。",
]

PLUGIN_AGENT_PROMPT_EXAMPLES = [
    "做一个 OCR 后处理插件，把每个识别结果去掉首尾空格，并把省略号统一改成三个点。",
    "做一个普通翻译后处理插件，把译文中的某些敏感词替换成更自然的说法。",
    "做一个 before_translate 插件，给 prompt 动态追加角色口癖说明。",
    "做一个 HQ 翻译结果后处理插件，给所有 AI 译文末尾追加测试标记。",
    "做一个 render 插件，统一开启描边并强制文字颜色为深色。",
    "做一个 pipeline 生命周期插件，在任务开始和结束时记录日志与耗时。",
]


def get_plugin_builder_skill_markdown() -> str:
    return _SKILL_PATH.read_text(encoding="utf-8")


def get_plugin_agent_settings_payload(*, plugin_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "success": True,
        "overview": list(PLUGIN_AGENT_OVERVIEW),
        "prompt_examples": list(PLUGIN_AGENT_PROMPT_EXAMPLES),
        "plugins": plugin_records,
    }
