from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


_SKILL_PATH = Path(__file__).with_name("plugin_builder_skill.md")

PLUGIN_AGENT_OVERVIEW_SECTIONS = [
    {
        "title": "基础规则",
        "items": [
            "插件只能挂在固定钩子节点上：**before_\\*** 表示步骤执行前，**after_\\*** 表示步骤执行后。",
            "新建插件默认是“默认不启用”，对应代码字段是 **default_enabled = False**，生成后建议先手动启用测试。",
            "如果你修改了 OCR 原文列表 **original_texts**，最好同时更新 **ocr_results[i][\"text\"]**，也就是“第 i 个 OCR 结果里的文本字段”，避免界面显示不一致。",
        ],
    },
    {
        "title": "检测与识别类 Hook",
        "items": [
            "**before_detect** / **after_detect**：检测前 / 检测后，用来修改检测参数或检测框结果。",
            "**before_ocr** / **after_ocr**：文字识别前 / 文字识别后，用来清洗 OCR 原文或同步识别结果。",
            "**before_color** / **after_color**：颜色提取前 / 颜色提取后，用来调整自动识别出的文字色和背景色。",
        ],
    },
    {
        "title": "翻译与渲染类 Hook",
        "items": [
            "**before_translate** / **after_translate**：普通翻译前 / 普通翻译后，用来改提示词、源文或普通译文。",
            "**before_ai_translate** / **after_ai_translate**：高质量翻译 / AI 校对前后，用来改 AI 提示词、结构化请求或 AI 译文。",
            "**before_inpaint** / **after_inpaint**：修复前 / 修复后，用来改修复参数或修复结果。",
            "**before_render** / **after_render**：渲染前 / 渲染后，用来统一描边、颜色、字号和最终显示效果。",
        ],
    },
    {
        "title": "生命周期与命名",
        "items": [
            "**before_pipeline** / **after_pipeline**：整次任务开始 / 结束时触发一次，适合做日志、统计、缓存和耗时记录。",
            "步骤名请使用系统规定的英文名：**detect**=检测、**ocr**=文字识别、**color**=颜色提取、**translate**=普通翻译、**ai_translate**=高质量翻译 / AI 校对、**inpaint**=修复、**render**=渲染、**pipeline**=整次任务生命周期。",
            "模式名请使用系统规定的英文名：**standard**=普通翻译、**hq**=高质量翻译、**proofread**=AI 校对、**remove_text**=仅消除文字；不要自己发明别名。",
        ],
    },
]

PLUGIN_AGENT_OVERVIEW = [
    item
    for section in PLUGIN_AGENT_OVERVIEW_SECTIONS
    for item in section["items"]
]

PLUGIN_AGENT_PROMPT_EXAMPLES = [
    "做一个 OCR 后处理插件，把每个识别结果去掉首尾空格，并把省略号统一改成三个点。",
    "做一个普通翻译后处理插件，把译文中的某些敏感词替换成更自然的说法。",
    "做一个 `before_translate`（普通翻译前）插件，给翻译提示词动态追加角色口癖说明。",
    "做一个 `after_ai_translate`（高质量翻译 / AI 校对后）插件，给所有 AI 译文末尾追加测试标记。",
    "做一个 `before_render`（渲染前）插件，统一开启描边并强制文字颜色为深色。",
    "做一个 `before_pipeline` / `after_pipeline`（整次任务开始 / 结束）插件，在任务开始和结束时记录日志与耗时。",
]


def get_plugin_builder_skill_markdown() -> str:
    return _SKILL_PATH.read_text(encoding="utf-8")


def get_plugin_agent_settings_payload(*, plugin_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "success": True,
        "overview": list(PLUGIN_AGENT_OVERVIEW),
        "overview_sections": [
            {
                "title": section["title"],
                "items": list(section["items"]),
            }
            for section in PLUGIN_AGENT_OVERVIEW_SECTIONS
        ],
        "prompt_examples": list(PLUGIN_AGENT_PROMPT_EXAMPLES),
        "plugins": plugin_records,
    }
