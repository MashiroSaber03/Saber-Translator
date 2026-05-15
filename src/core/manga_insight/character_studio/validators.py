"""
Validation helpers for Character Studio documents and exports.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .adapters import build_export_bundle, ensure_document_shape


def validate_document(document: Dict[str, Any]) -> Dict[str, List[str]]:
    doc = ensure_document_shape(document)
    errors: List[str] = []
    warnings: List[str] = []

    if not doc["identity"]["name"].strip():
        errors.append("identity.name 不能为空")
    if not doc["coreMessages"]["first_message"].strip():
        warnings.append("coreMessages.first_message 为空")
    if not isinstance(doc["coreMessages"]["alternate_greetings"], list):
        errors.append("coreMessages.alternate_greetings 必须为数组")

    regex_errors = _validate_regex_scripts(doc.get("regexScripts", []))
    errors.extend(regex_errors)

    task_errors = _validate_state_tasks(doc.get("stateTasks", []))
    errors.extend(task_errors)

    lorebook_errors = _validate_lorebook_entries(doc.get("lorebook", {}).get("entries", []))
    errors.extend(lorebook_errors)

    return {
        "errors": errors,
        "warnings": warnings,
    }


def _validate_regex_scripts(scripts: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    for idx, script in enumerate(scripts or []):
        if not isinstance(script, dict):
            errors.append(f"regexScripts[{idx}] 必须为对象")
            continue
        pattern = script.get("findRegex", "")
        if not str(pattern).strip():
            errors.append(f"regexScripts[{idx}].findRegex 不能为空")
            continue
        try:
            re.compile(str(pattern))
        except re.error as exc:
            errors.append(f"regexScripts[{idx}] 正则非法: {exc}")
    return errors


def _validate_state_tasks(tasks: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    for idx, task in enumerate(tasks or []):
        if not isinstance(task, dict):
            errors.append(f"stateTasks[{idx}] 必须为对象")
            continue
        if not str(task.get("name", "")).strip():
            errors.append(f"stateTasks[{idx}].name 不能为空")
        commands = str(task.get("commands", "") or "")
        if not commands.strip():
            errors.append(f"stateTasks[{idx}].commands 不能为空")
        if "<<taskjs>>" in commands and "<</taskjs>>" not in commands:
            errors.append(f"stateTasks[{idx}] 缺少 <</taskjs>> 结束标记")
    return errors


def _validate_lorebook_entries(entries: List[Dict[str, Any]], path: str = "lorebook.entries") -> List[str]:
    errors: List[str] = []
    for idx, entry in enumerate(entries or []):
        current_path = f"{path}[{idx}]"
        if not isinstance(entry, dict):
            errors.append(f"{current_path} 必须为对象")
            continue
        keys = entry.get("keys", []) or []
        if not isinstance(keys, list) or len(keys) == 0:
            errors.append(f"{current_path}.keys 必须为非空数组")
        children = entry.get("children", []) or []
        if children:
            errors.extend(_validate_lorebook_entries(children, path=f"{current_path}.children"))
    return errors


def build_diagnostics_report(document: Dict[str, Any]) -> Dict[str, Any]:
    bundle = build_export_bundle(document)
    validation = validate_document(document)
    export_warnings: List[str] = []
    if not bundle["v3"]["data"]["character_book"]["entries"]:
        export_warnings.append("character_book 为空，导出将不包含世界书条目")
    if not bundle["v3"]["data"]["alternate_greetings"]:
        export_warnings.append("alternate_greetings 为空")
    return {
        "valid": len(validation["errors"]) == 0,
        "errors": validation["errors"],
        "warnings": validation["warnings"] + export_warnings,
        "checks": {
            "document": len(validation["errors"]) == 0,
            "v3_export": bundle["v3"]["spec"] == "chara_card_v3",
            "v2_export": bundle["v2"]["spec"] == "chara_card_v2",
            "regex_scripts": len(document.get("regexScripts", [])) >= 0,
            "state_tasks": len(document.get("stateTasks", [])) >= 0,
        },
    }
