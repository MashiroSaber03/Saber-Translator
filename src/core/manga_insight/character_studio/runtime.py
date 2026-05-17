"""
Shared runtime helpers for Character Studio chat and card assistant flows.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Tuple


def initialize_runtime_session(document: Dict[str, Any]) -> Dict[str, Any]:
    session = {
        "doc_id": document.get("id", ""),
        "messages": [],
        "variables": {},
        "log": [],
        "_runtime": {
            "event_counts": {
                "message_received": 0,
                "message_sent": 0,
            },
            "matched_lorebook_ids": [],
        },
    }
    session["log"].extend(run_state_tasks(session, document.get("stateTasks", []), event="initialization"))
    return session


def _apply_task_commands(commands: str, session: Dict[str, Any]) -> None:
    if not isinstance(commands, str):
        return
    for line in commands.splitlines():
        line = line.strip()
        match = re.search(r"/setvar\s+key=([A-Za-z0-9_\-\.]+)\s+([^'\")]+)", line)
        if match:
            session["variables"][match.group(1)] = match.group(2).strip().strip("'\"")


def run_state_tasks(session: Dict[str, Any], tasks: List[Dict[str, Any]], *, event: str) -> List[Dict[str, Any]]:
    runtime = session.setdefault("_runtime", {})
    event_counts = runtime.setdefault("event_counts", {})
    if event != "initialization":
        event_counts[event] = int(event_counts.get(event, 0) or 0) + 1
    current_count = int(event_counts.get(event, 0) or 0)

    logs: List[Dict[str, Any]] = []
    for task in tasks or []:
        if task.get("disabled"):
            continue
        if task.get("triggerTiming") != event:
            continue
        interval = int(task.get("interval", 0) or 0)
        if event != "initialization" and interval > 1 and current_count % interval != 0:
            continue
        _apply_task_commands(task.get("commands", ""), session)
        logs.append({
            "type": "task",
            "name": task.get("name", ""),
            "event": event,
            "interval": interval,
        })
    return logs


def apply_regex_scripts(
    text: str,
    scripts: List[Dict[str, Any]],
    placement: int,
    *,
    respect_run_on_edit: bool = False,
) -> Tuple[str, str, List[Dict[str, Any]]]:
    visible_text = text
    prompt_text = text
    hits: List[Dict[str, Any]] = []
    for script in scripts or []:
        if script.get("disabled"):
            continue
        if respect_run_on_edit and not bool(script.get("runOnEdit", True)):
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


def match_lorebook(entries: List[Dict[str, Any]], text: str, *, session: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []
    runtime = (session or {}).setdefault("_runtime", {}) if session is not None else {}
    matched_ids = set(runtime.setdefault("matched_lorebook_ids", [])) if session is not None else set()
    for entry in _flatten(entries):
        if not entry.get("enabled", True):
            continue
        entry_id = str(entry.get("id") or entry.get("comment") or "")
        if entry.get("prevent_recursion", False) and entry_id and entry_id in matched_ids:
            continue
        if entry.get("constant", False):
            matched.append(entry)
            if entry.get("prevent_recursion", False) and entry_id:
                matched_ids.add(entry_id)
            continue
        primary_match = _matches_keys(text, entry.get("keys", []) or [], bool(entry.get("use_regex", False)))
        secondary_match = _matches_keys(text, entry.get("secondary_keys", []) or [], bool(entry.get("use_regex", False)))
        selective = bool(entry.get("selective", True))
        if entry.get("secondary_keys"):
            hit = (primary_match and secondary_match) if selective else (primary_match or secondary_match)
        else:
            hit = primary_match
        if not hit or not _passes_probability(entry, text):
            continue
        matched.append(entry)
        if entry.get("prevent_recursion", False) and entry_id:
            matched_ids.add(entry_id)
    if session is not None:
        runtime["matched_lorebook_ids"] = list(matched_ids)
    return matched


def sort_lorebook_hits(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    position_order = {
        "before_char": 0,
        "at_depth": 1,
        "after_char": 2,
    }
    return sorted(
        entries or [],
        key=lambda entry: (
            position_order.get(str(entry.get("position", "before_char")), 1),
            -int(entry.get("priority", 100) or 100),
            -int(entry.get("depth", 4) or 4),
            str(entry.get("comment", "")),
        ),
    )


def _flatten(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for entry in entries or []:
        result.append(entry)
        result.extend(_flatten(entry.get("children", []) or []))
    return result


def _matches_keys(text: str, keys: List[Any], use_regex: bool) -> bool:
    haystack = text or ""
    for key in keys or []:
        key_text = str(key or "").strip()
        if not key_text:
            continue
        if use_regex:
            try:
                if re.search(key_text, haystack, re.IGNORECASE):
                    return True
            except re.error:
                continue
        elif key_text.lower() in haystack.lower():
            return True
    return False


def _passes_probability(entry: Dict[str, Any], text: str) -> bool:
    raw_probability = entry.get("probability", 100)
    if raw_probability in (None, ""):
        probability = 100
    else:
        probability = int(raw_probability)
    if probability <= 0:
        return False
    if probability >= 100:
        return True
    token = f"{entry.get('id', '')}|{text}".encode("utf-8")
    score = int(hashlib.sha1(token).hexdigest()[:8], 16) % 100 + 1
    return score <= probability
