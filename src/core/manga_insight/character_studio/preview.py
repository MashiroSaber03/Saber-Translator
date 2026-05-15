"""
Lightweight preview runtime for Character Studio.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Tuple


def initialize_preview_session(document: Dict[str, Any]) -> Dict[str, Any]:
    session = {
        "doc_id": document.get("id", ""),
        "messages": [],
        "variables": {},
        "log": [],
    }
    for task in document.get("stateTasks", []) or []:
        if task.get("disabled"):
            continue
        if task.get("triggerTiming") != "initialization":
            continue
        _apply_task_commands(task.get("commands", ""), session)
        session["log"].append({
            "type": "task",
            "name": task.get("name", ""),
            "event": "initialization",
        })
    return session


def _apply_task_commands(commands: str, session: Dict[str, Any]) -> None:
    if not isinstance(commands, str):
        return
    for line in commands.splitlines():
        line = line.strip()
        match = re.search(r"/setvar\s+key=([A-Za-z0-9_\-\.]+)\s+([^'\")]+)", line)
        if match:
            session["variables"][match.group(1)] = match.group(2).strip().strip("'\"")


def apply_regex_scripts(text: str, scripts: List[Dict[str, Any]], placement: int) -> Tuple[str, str, List[Dict[str, Any]]]:
    visible_text = text
    prompt_text = text
    hits: List[Dict[str, Any]] = []
    for script in scripts or []:
        if script.get("disabled"):
            continue
        placement_values = script.get("placement", [2])
        if isinstance(placement_values, int):
            placement_values = [placement_values]
        if placement not in placement_values:
            continue
        pattern = str(script.get("findRegex", "") or "")
        if not pattern:
            continue
        replacement = str(script.get("replaceString", "") or "")
        try:
            regex = re.compile(pattern)
        except re.error:
            continue
        if regex.search(visible_text) or regex.search(prompt_text):
            hits.append({
                "type": "regex",
                "scriptName": script.get("scriptName", ""),
                "pattern": pattern,
            })
        if script.get("promptOnly"):
            prompt_text = regex.sub(replacement, prompt_text)
        elif script.get("markdownOnly"):
            visible_text = regex.sub(replacement, visible_text)
        else:
            visible_text = regex.sub(replacement, visible_text)
            prompt_text = regex.sub(replacement, prompt_text)
    return visible_text, prompt_text, hits


def match_lorebook(entries: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []
    for entry in _flatten(entries):
        if not entry.get("enabled", True):
            continue
        keys = entry.get("keys", []) or []
        if entry.get("constant", False):
            matched.append(entry)
            continue
        for key in keys:
            key_text = str(key or "").strip()
            if key_text and key_text.lower() in text.lower():
                matched.append(entry)
                break
    return matched


def _flatten(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for entry in entries or []:
        result.append(entry)
        result.extend(_flatten(entry.get("children", []) or []))
    return result
