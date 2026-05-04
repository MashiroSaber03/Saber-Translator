"""
术语表 / 禁翻表共享约束层

参考 AiNiee 的思路：
- 术语表：命中当前文本后注入 prompt，并在翻译完成后做检查告警
- 禁翻表：命中当前文本后注入 prompt，同时执行译前占位保护与译后还原
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, Iterable, List, Tuple


CHINESE_TARGET_CODES = {
    "zh",
    "zh-cn",
    "zh-tw",
    "chinese",
    "chinese_simplified",
    "chinese_traditional",
}


def normalize_glossary_settings(payload: Any) -> Dict[str, Any]:
    return {
        "enabled": bool(_value(payload, "enabled", default=False)),
        "entries": [
            {
                "source": str(_value(entry, "source", "src", default="") or "").strip(),
                "target": str(_value(entry, "target", "dst", "recommendedTranslation", default="") or "").strip(),
                "note": str(_value(entry, "note", "info", default="") or "").strip(),
                "matchMode": _normalize_match_mode(_value(entry, "matchMode", "match_mode", default="text")),
            }
            for entry in _iter_entries(payload)
            if str(_value(entry, "source", "src", default="") or "").strip()
        ],
    }


def normalize_non_translate_settings(payload: Any) -> Dict[str, Any]:
    return {
        "enabled": bool(_value(payload, "enabled", default=False)),
        "entries": [
            {
                "pattern": str(_value(entry, "pattern", "marker", "markers", default="") or "").strip(),
                "note": str(_value(entry, "note", "info", default="") or "").strip(),
                "matchMode": _normalize_match_mode(_value(entry, "matchMode", "match_mode", default="text")),
            }
            for entry in _iter_entries(payload)
            if str(_value(entry, "pattern", "marker", "markers", default="") or "").strip()
        ],
    }


def append_prompt_sections(base_prompt: str | None, *sections: str) -> str:
    cleaned_sections = [section.strip() for section in sections if section and section.strip()]
    base = (base_prompt or "").strip()
    if not cleaned_sections:
        return base
    if not base:
        return "\n\n".join(cleaned_sections)
    return "\n\n".join([base, *cleaned_sections])


def build_glossary_prompt(settings: Dict[str, Any], texts: Iterable[str], target_language: str = "zh") -> str:
    matches = _match_glossary_entries(settings, texts)
    if not matches:
        return ""

    if _is_chinese_target(target_language):
        lines = ["###术语表", "原文|译文|备注"]
    else:
        lines = ["###Glossary", "Original Text|Translation|Remarks"]

    for match in matches:
        lines.append(
            f"{match['source']}|{match['target']}|{match['note'] or ' '}"
        )
    return "\n".join(lines)


def build_non_translate_prompt(settings: Dict[str, Any], texts: Iterable[str], target_language: str = "zh") -> str:
    matches = _match_non_translate_entries(settings, texts)
    if not matches:
        return ""

    if _is_chinese_target(target_language):
        lines = ["###禁翻表，以下特殊标记符无需翻译", "特殊标记符|备注"]
    else:
        lines = ["###Non-Translation List", "Special marker|Remarks"]

    for match in matches:
        lines.append(f"{match['pattern']}|{match['note'] or ' '}")
    return "\n".join(lines)


def build_non_translate_guard_prompt(
    mappings: Iterable[List[Dict[str, str]]],
    target_language: str = "zh",
) -> str:
    placeholders: List[str] = []
    seen = set()

    for replacements in mappings:
        for replacement in replacements or []:
            placeholder = str(replacement.get("placeholder", "") or "").strip()
            if not placeholder or placeholder in seen:
                continue
            seen.add(placeholder)
            placeholders.append(placeholder)

    if not placeholders:
        return ""

    if _is_chinese_target(target_language):
        lines = [
            "###占位符保护规则",
            "以下占位符代表原文中的禁翻内容。你必须逐字原样保留这些占位符，不得翻译、解释、删除、改写，也不要在占位符内部插入空格或其他字符：",
            *placeholders,
        ]
    else:
        lines = [
            "###Placeholder Protection Rules",
            "The following placeholders represent protected non-translatable content. You must preserve them exactly as-is. Do not translate, explain, delete, rewrite, or insert spaces inside them:",
            *placeholders,
        ]

    return "\n".join(lines)


def protect_texts_with_non_translate(texts: List[str], settings: Dict[str, Any]) -> Tuple[List[str], List[List[Dict[str, str]]]]:
    protected_texts: List[str] = []
    mappings: List[List[Dict[str, str]]] = []
    for text in texts:
        protected_text, replacements = protect_text_with_non_translate(text, settings)
        protected_texts.append(protected_text)
        mappings.append(replacements)
    return protected_texts, mappings


def protect_text_with_non_translate(text: str, settings: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]]]:
    if not text or not settings.get("enabled"):
        return text, []

    matches = _select_non_overlapping_matches(text, settings)
    if not matches:
        return text, []

    chunks: List[str] = []
    replacements: List[Dict[str, str]] = []
    cursor = 0
    for index, match in enumerate(matches, start=1):
        placeholder = f"__SABER_NTL_{index:04d}__"
        chunks.append(text[cursor:match["start"]])
        chunks.append(placeholder)
        replacements.append({
            "placeholder": placeholder,
            "original": match["text"],
        })
        cursor = match["end"]
    chunks.append(text[cursor:])
    return "".join(chunks), replacements


def restore_texts_with_non_translate(texts: List[str], mappings: List[List[Dict[str, str]]]) -> List[str]:
    restored = []
    for index, text in enumerate(texts):
        restored.append(restore_text_with_non_translate(text, mappings[index] if index < len(mappings) else []))
    return restored


def restore_text_with_non_translate(text: str, replacements: List[Dict[str, str]]) -> str:
    restored = text
    for replacement in replacements:
        placeholder = replacement.get("placeholder", "")
        original = replacement.get("original", "")
        if placeholder:
            restored = restored.replace(placeholder, original)
    return restored


def protect_hq_json_data(
    json_data: List[Dict[str, Any]],
    settings: Dict[str, Any],
    fields: Tuple[str, ...] = ("original", "translated"),
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[int, int], List[Dict[str, str]]]]:
    protected = copy.deepcopy(json_data)
    bubble_mappings: Dict[Tuple[int, int], List[Dict[str, str]]] = {}

    for image_index, image_item in enumerate(protected):
        actual_image_index = int(image_item.get("imageIndex", image_index))
        bubbles = image_item.get("bubbles", [])
        if not isinstance(bubbles, list):
            continue
        for bubble_index, bubble in enumerate(bubbles):
            if not isinstance(bubble, dict):
                continue
            all_replacements: List[Dict[str, str]] = []
            actual_bubble_index = int(bubble.get("bubbleIndex", bubble_index))
            for field_name in fields:
                raw_value = bubble.get(field_name)
                if isinstance(raw_value, str) and raw_value.strip():
                    protected_text, replacements = protect_text_with_non_translate(raw_value, settings)
                    bubble[field_name] = protected_text
                    all_replacements.extend(replacements)
            bubble_mappings[(actual_image_index, actual_bubble_index)] = all_replacements

    return protected, bubble_mappings


def restore_hq_result_data(
    result_data: List[Dict[str, Any]],
    bubble_mappings: Dict[Tuple[int, int], List[Dict[str, str]]],
) -> List[Dict[str, Any]]:
    restored = copy.deepcopy(result_data)
    for image_index, image_item in enumerate(restored):
        actual_image_index = int(image_item.get("imageIndex", image_index))
        bubbles = image_item.get("bubbles", [])
        if not isinstance(bubbles, list):
            continue
        for bubble_index, bubble in enumerate(bubbles):
            if not isinstance(bubble, dict):
                continue
            actual_bubble_index = int(bubble.get("bubbleIndex", bubble_index))
            replacements = bubble_mappings.get((actual_image_index, actual_bubble_index), [])
            translated = bubble.get("translated")
            if isinstance(translated, str) and replacements:
                bubble["translated"] = restore_text_with_non_translate(translated, replacements)
    return restored


def collect_glossary_warnings(
    settings: Dict[str, Any],
    source_text: str,
    translated_text: str,
    *,
    image_index: int | None = None,
    bubble_index: int | None = None,
) -> List[Dict[str, Any]]:
    if not source_text or not translated_text or not settings.get("enabled"):
        return []

    warnings: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()

    for match in _match_glossary_entries(settings, [source_text]):
        expected_target = match.get("target", "").strip()
        matched_source = match.get("source", "").strip()
        if not expected_target or not matched_source:
            continue
        dedupe_key = (matched_source.casefold(), expected_target.casefold())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        if expected_target.casefold() not in translated_text.casefold():
            warnings.append({
                "imageIndex": image_index,
                "bubbleIndex": bubble_index,
                "source": matched_source,
                "expectedTarget": expected_target,
                "actualTranslation": translated_text,
            })
    return warnings


def validate_constraint_entries(entries: List[Dict[str, Any]], *, pattern_field: str) -> List[str]:
    errors: List[str] = []
    for index, entry in enumerate(entries or []):
        match_mode = _normalize_match_mode(entry.get("matchMode", entry.get("match_mode", "text")))
        pattern = str(entry.get(pattern_field, "") or "").strip()
        if match_mode == "regex" and pattern:
            try:
                re.compile(pattern)
            except re.error as exc:
                errors.append(f"第 {index + 1} 行正则无效: {exc}")
    return errors


def _iter_entries(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        entries = payload.get("entries")
        if isinstance(entries, list):
            return [entry for entry in entries if isinstance(entry, dict)]
    return []


def _value(payload: Any, *keys: str, default: Any = None) -> Any:
    if not isinstance(payload, dict):
        return default
    for key in keys:
        if key in payload and payload.get(key) is not None:
            return payload.get(key)
    return default


def _normalize_match_mode(value: Any) -> str:
    return "regex" if str(value or "").strip().lower() == "regex" else "text"


def _is_chinese_target(target_language: str | None) -> bool:
    normalized = str(target_language or "").strip().lower()
    return normalized in CHINESE_TARGET_CODES


def _build_matcher(raw_pattern: str, match_mode: str) -> re.Pattern[str] | None:
    try:
        if match_mode == "regex":
            return re.compile(raw_pattern, re.IGNORECASE)
        return re.compile(re.escape(raw_pattern), re.IGNORECASE)
    except re.error:
        return None


def _match_glossary_entries(settings: Dict[str, Any], texts: Iterable[str]) -> List[Dict[str, str]]:
    if not settings.get("enabled"):
        return []

    full_text = "\n".join(text for text in texts if text)
    if not full_text:
        return []

    matches: List[Dict[str, str]] = []
    seen: set[Tuple[str, str, str]] = set()
    for entry in settings.get("entries", []):
        source = str(entry.get("source", "") or "").strip()
        target = str(entry.get("target", "") or "").strip()
        if not source or not target:
            continue
        matcher = _build_matcher(source, entry.get("matchMode", "text"))
        if matcher is None:
            continue
        for match in matcher.finditer(full_text):
            matched_text = match.group(0)
            if not matched_text:
                continue
            dedupe_key = (matched_text.casefold(), target.casefold(), str(entry.get("note", "") or ""))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            matches.append({
                "source": matched_text,
                "target": target,
                "note": str(entry.get("note", "") or "").strip(),
            })
    return matches


def _match_non_translate_entries(settings: Dict[str, Any], texts: Iterable[str]) -> List[Dict[str, str]]:
    if not settings.get("enabled"):
        return []

    full_text = "\n".join(text for text in texts if text)
    if not full_text:
        return []

    matches: List[Dict[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for entry in settings.get("entries", []):
        pattern = str(entry.get("pattern", "") or "").strip()
        if not pattern:
            continue
        matcher = _build_matcher(pattern, entry.get("matchMode", "text"))
        if matcher is None:
            continue
        for match in matcher.finditer(full_text):
            matched_text = match.group(0)
            if not matched_text:
                continue
            dedupe_key = (matched_text, str(entry.get("note", "") or ""))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            matches.append({
                "pattern": matched_text,
                "note": str(entry.get("note", "") or "").strip(),
            })
    return matches


def _select_non_overlapping_matches(text: str, settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    all_matches: List[Dict[str, Any]] = []
    for entry in settings.get("entries", []):
        raw_pattern = str(entry.get("pattern", "") or "").strip()
        if not raw_pattern:
            continue
        matcher = _build_matcher(raw_pattern, entry.get("matchMode", "text"))
        if matcher is None:
            continue
        for match in matcher.finditer(text):
            matched_text = match.group(0)
            if not matched_text:
                continue
            all_matches.append({
                "start": match.start(),
                "end": match.end(),
                "text": matched_text,
            })

    if not all_matches:
        return []

    all_matches.sort(key=lambda item: (item["start"], -(item["end"] - item["start"])))
    selected: List[Dict[str, Any]] = []
    cursor = -1
    for match in all_matches:
        if match["start"] < cursor:
            continue
        selected.append(match)
        cursor = match["end"]
    return selected
