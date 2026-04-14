"""
角色卡映射与模板构建。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .models import (
    CharacterBookEntry,
    CharacterBookSchema,
    HelperUiManifest,
    MvuVariable,
    RegexProfile,
)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def build_worldbook(
    character: Dict[str, Any],
    timeline_data: Dict[str, Any],
    overview_text: str,
) -> CharacterBookSchema:
    """构建角色内嵌世界书。"""
    name = _as_text(character.get("name")) or "未知角色"
    aliases = [a for a in character.get("aliases", []) if _as_text(a)]
    relationships = character.get("relationships", []) or []
    key_moments = character.get("key_moments", []) or []
    plot_threads = timeline_data.get("plot_threads", []) or []

    entries: List[CharacterBookEntry] = []
    uid = 1

    # 角色基底
    profile_content = []
    if character.get("description"):
        profile_content.append(f"角色简介：{character.get('description')}")
    if character.get("arc"):
        profile_content.append(f"角色成长线：{character.get('arc')}")
    if overview_text:
        profile_content.append(f"故事语境：{_truncate(overview_text, 220)}")
    if profile_content:
        entries.append(
            CharacterBookEntry(
                uid=uid,
                key=[name] + aliases,
                comment=f"{name}角色基底",
                content="\n".join(profile_content),
                insertion_order=10,
                position="before_char",
            )
        )
        uid += 1

    # 关系条目
    for rel in relationships[:10]:
        other = _as_text(rel.get("character"))
        relation = _as_text(rel.get("relation"))
        if not other or not relation:
            continue
        entries.append(
            CharacterBookEntry(
                uid=uid,
                key=[other, name],
                keysecondary=aliases[:3],
                comment=f"{name}关系：{other}",
                content=f"{name} 与 {other} 的关系：{relation}",
                insertion_order=20,
                position="before_char",
            )
        )
        uid += 1

    # 关键时刻
    for idx, moment in enumerate(key_moments[:12], start=1):
        page = moment.get("page", 0)
        event = _as_text(moment.get("event"))
        significance = _as_text(moment.get("significance"))
        if not event:
            continue
        content = f"第{page}页关键时刻：{event}"
        if significance:
            content += f"\n意义：{significance}"
        entries.append(
            CharacterBookEntry(
                uid=uid,
                key=[name, f"{name}关键时刻{idx}"],
                comment=f"{name}关键时刻#{idx}",
                content=content,
                insertion_order=30,
                position="before_char",
            )
        )
        uid += 1

    # 线索条目（筛选相关线索）
    for thread in plot_threads[:15]:
        desc = _as_text(thread.get("description"))
        tname = _as_text(thread.get("name"))
        if not tname or not desc:
            continue

        is_related = name in desc
        if not is_related:
            for alias in aliases:
                if alias and alias in desc:
                    is_related = True
                    break
        if not is_related:
            continue

        status = _as_text(thread.get("status")) or "进行中"
        content = f"线索：{tname}\n状态：{status}\n说明：{_truncate(desc, 240)}"
        entries.append(
            CharacterBookEntry(
                uid=uid,
                key=[tname, name],
                comment=f"{name}相关线索：{tname}",
                content=content,
                insertion_order=40,
                position="before_char",
            )
        )
        uid += 1

    return CharacterBookSchema(
        name=f"{name} - Manga Insight Lorebook",
        description=f"{name} 的角色内嵌世界书",
        scan_depth=3,
        token_budget=900,
        recursive_scanning=True,
        extensions={"source": "manga_insight"},
        entries=entries,
    )


def build_regex_profiles(character_name: str) -> List[RegexProfile]:
    """构建角色作用域正则模板。"""
    escaped_name = re.escape(character_name or "角色")
    return [
        RegexProfile(
            id="name_normalization",
            name="角色称谓标准化",
            scope="character",
            source="ai_output",
            pattern=rf"(?:我|咱|本小姐)({escaped_name})",
            replacement=rf"\1",
            flags="gi",
            order=10,
            notes="统一称谓，减少不稳定自称结构。",
        ),
        RegexProfile(
            id="punctuation_cleanup",
            name="标点清洗",
            scope="character",
            source="ai_output",
            pattern=r"[ \t]{2,}",
            replacement=" ",
            flags="g",
            order=20,
            notes="收敛多余空白。",
        ),
        RegexProfile(
            id="quote_balance",
            name="对话引号修正",
            scope="character",
            source="ai_output",
            pattern=r"“([^”\n]+)$",
            replacement="“$1”",
            flags="gm",
            order=30,
            notes="修补缺失闭合引号的行。",
        ),
    ]


def build_mvu_template(character_name: str) -> List[MvuVariable]:
    """构建 MVU 模板。"""
    return [
        MvuVariable(
            name="relationship_score",
            type="number",
            scope="character",
            default=0,
            value=0,
            validator={"min": -100, "max": 100},
            description=f"{character_name} 与 {{user}} 的关系值。",
        ),
        MvuVariable(
            name="trust_score",
            type="number",
            scope="chat",
            default=20,
            value=20,
            validator={"min": 0, "max": 100},
            description="信任值，影响语气亲密度。",
        ),
        MvuVariable(
            name="emotion_state",
            type="string",
            scope="message",
            default="平静",
            value="平静",
            validator={"enum": ["平静", "警惕", "愤怒", "喜悦", "悲伤"]},
            description="当前情绪态。",
        ),
        MvuVariable(
            name="plot_stage",
            type="string",
            scope="script",
            default="起始",
            value="起始",
            validator={"enum": ["起始", "推进", "冲突", "转折", "收束"]},
            description="剧情阶段。",
        ),
        MvuVariable(
            name="form_state",
            type="string",
            scope="extension",
            default="default",
            value="default",
            validator={"pattern": r"^[a-zA-Z0-9_-]{1,32}$"},
            description="角色形态状态。",
        ),
    ]


def build_helper_ui_manifest(character_name: str) -> HelperUiManifest:
    """构建 Tavern Helper UI 模板。"""
    return HelperUiManifest(
        layout="split-dashboard",
        theme="manga-insight-light",
        panels=[
            {"id": "profile", "title": f"{character_name} 档案", "type": "markdown"},
            {"id": "status", "title": "状态监控", "type": "variables"},
            {"id": "lorebook", "title": "世界书命中", "type": "lorebook"},
        ],
        widgets=[
            {"id": "trust_meter", "type": "progress", "bind": "trust_score"},
            {"id": "relation_badge", "type": "badge", "bind": "relationship_score"},
            {"id": "emotion_chip", "type": "chip", "bind": "emotion_state"},
        ],
        actions=[
            {"id": "calm_down", "label": "平复情绪", "set": {"emotion_state": "平静"}},
            {"id": "advance_plot", "label": "推进剧情", "set": {"plot_stage": "推进"}},
        ],
        events=[
            {"on": "message.received", "action": "syncEmotionFromTone"},
            {"on": "lorebook.hit", "action": "highlightPanel", "target": "lorebook"},
        ],
        bindings=[
            {"widget": "trust_meter", "var": "trust_score"},
            {"widget": "relation_badge", "var": "relationship_score"},
            {"widget": "emotion_chip", "var": "emotion_state"},
        ],
    )


def build_import_manifest() -> Dict[str, Any]:
    """构建扩展安装清单。"""
    return {
        "version": "1.0.0",
        "requires": {
            "sillytavern": ">=1.12.0",
            "tavern_helper_capabilities": [
                "variables",
                "character_card_extensions",
                "ui_manifest",
                "regex_profiles",
            ],
        },
        "activate_steps": [
            "导入角色卡 PNG",
            "确认扩展区 data.extensions.saber_tavern 已存在",
            "若安装 Tavern Helper，启用对应扩展读取器",
        ],
        "fallback_behavior": "若扩展区未被识别，角色卡核心字段仍可正常使用。",
    }


def build_mes_example(dialogues: List[Dict[str, Any]], character_name: str) -> str:
    """构建 mes_example。"""
    examples: List[str] = []
    for item in dialogues[:4]:
        text = _as_text(item.get("text"))
        if not text:
            continue
        examples.append(f"<START>\n{{{{user}}}}: 现在情况怎么样？\n{{{{char}}}}: {text}")
    if examples:
        return "\n\n".join(examples)

    return (
        "<START>\n{{user}}: 我们下一步做什么？\n"
        f"{{{{char}}}}: 先稳住局面，再找关键线索。"
    )


def build_first_message(dialogues: List[Dict[str, Any]], character_name: str) -> str:
    for item in dialogues:
        text = _as_text(item.get("text"))
        if text:
            return text
    return f"我是{character_name}。先说重点，你希望我从哪里开始？"


def build_card_template(
    character: Dict[str, Any],
    overview_text: str,
    dialogues: List[Dict[str, Any]],
    timeline_data: Dict[str, Any],
) -> Dict[str, Any]:
    """构建 V2 角色卡模板。"""
    name = _as_text(character.get("name")) or "未知角色"
    aliases = [a for a in character.get("aliases", []) if _as_text(a)]
    description = _as_text(character.get("description")) or "暂无角色描述。"
    arc = _as_text(character.get("arc"))
    first_appearance = character.get("first_appearance", 0)

    profile_lines = [description]
    if arc:
        profile_lines.append(f"角色成长线：{arc}")
    if first_appearance:
        profile_lines.append(f"首次登场：第{first_appearance}页")

    worldbook = build_worldbook(character, timeline_data, overview_text).to_dict()
    regex_profiles = [r.to_dict() for r in build_regex_profiles(name)]
    mvu_vars = [v.to_dict() for v in build_mvu_template(name)]
    ui_manifest = build_helper_ui_manifest(name).to_dict()

    data = {
        "name": name,
        "description": "\n".join(profile_lines),
        "personality": arc or "外在克制，内在有明确目标感。",
        "scenario": _truncate(
            overview_text or "故事处于持续推进阶段，角色正面临新的选择与冲突。",
            520,
        ),
        "first_mes": build_first_message(dialogues, name),
        "mes_example": build_mes_example(dialogues, name),
        "creator_notes": "由 Saber Translator 漫画分析自动生成，可手动二次编辑。",
        "system_prompt": "保持角色设定一致，优先依据角色经历与关系进行回应。",
        "post_history_instructions": "保持叙事连续，避免脱离世界观。",
        "alternate_greetings": [
            f"你来得正好，我是{name}，我们直接进入正题。",
            f"……又见面了。我是{name}，这次你打算怎么推进？",
        ],
        "tags": list(dict.fromkeys(["manga", "insight", "auto-generated", name] + aliases[:4])),
        "creator": "Saber Translator",
        "character_version": "1.0.0",
        "character_book": worldbook,
        "extensions": {
            "saber_tavern": {
                "regex_profiles": regex_profiles,
                "mvu": {
                    "version": "1.0.0",
                    "variables": mvu_vars,
                },
                "ui_manifest": ui_manifest,
                "import_manifest": build_import_manifest(),
                "source": {
                    "generated_by": "manga_insight_character_cards",
                    "character": name,
                },
            }
        },
    }

    return {
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": data,
    }


def apply_generated_fields(card: Dict[str, Any], generated: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 LLM 生成字段合并到模板卡中，仅覆盖允许字段。
    """
    data = card.setdefault("data", {})
    allowed = [
        "description",
        "personality",
        "scenario",
        "first_mes",
        "mes_example",
        "creator_notes",
        "system_prompt",
        "post_history_instructions",
        "alternate_greetings",
        "tags",
    ]
    for key in allowed:
        if key in generated and generated[key] not in (None, ""):
            data[key] = generated[key]

    if isinstance(data.get("alternate_greetings"), str):
        data["alternate_greetings"] = [data["alternate_greetings"]]
    if not isinstance(data.get("tags"), list):
        data["tags"] = [str(data.get("tags", ""))] if data.get("tags") else []
    return card
