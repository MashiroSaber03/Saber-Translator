"""Canonical v2 schema and validation for book translation constraints."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
import re
from typing import Any

from src.shared.prompt_defaults import get_prompt_factory_defaults


TRANSLATION_CONSTRAINTS_SCHEMA_VERSION = 2
DEFAULT_AUTO_GLOSSARY_PROMPT = get_prompt_factory_defaults()["autoGlossary"]

_MATCH_MODES = frozenset({"text", "regex"})
_MAX_TERM_LENGTH = 2_000
_MAX_NOTE_LENGTH = 10_000
_MAX_PROMPT_LENGTH = 100_000


def empty_translation_constraints() -> dict[str, Any]:
    """Return a detached canonical empty document payload."""

    return {
        "glossary": {
            "enabled": False,
            "autoExtractEnabled": False,
            "autoExtractPrompt": DEFAULT_AUTO_GLOSSARY_PROMPT,
            "entries": [],
        },
        "nonTranslate": {
            "enabled": False,
            "entries": [],
        },
    }


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return dict(value)


def _exact_fields(
    value: Mapping[str, Any],
    *,
    field: str,
    expected: set[str],
) -> None:
    actual = set(value)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise ValueError(f"{field} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{field} contains unknown fields: {', '.join(sorted(unknown))}")


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _string(
    value: object,
    *,
    field: str,
    maximum: int,
    allow_empty: bool,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    normalized = value.strip()
    if not allow_empty and not normalized:
        raise ValueError(f"{field} must not be empty")
    if len(normalized) > maximum:
        raise ValueError(f"{field} exceeds {maximum} characters")
    return normalized


def _match_mode(value: object, *, field: str) -> str:
    if not isinstance(value, str) or value not in _MATCH_MODES:
        raise ValueError(f"{field} must be text or regex")
    return value


def _compile_regex(value: str, *, field: str) -> None:
    try:
        re.compile(value)
    except re.error as exc:
        raise ValueError(f"{field} contains an invalid regular expression: {exc}") from exc


def _entries(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return value


def _glossary_entries(value: object) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    seen: dict[tuple[str, str], dict[str, str]] = {}
    for index, raw in enumerate(_entries(value, field="glossary.entries")):
        field = f"glossary.entries[{index}]"
        entry = _mapping(raw, field=field)
        _exact_fields(
            entry,
            field=field,
            expected={"source", "target", "note", "matchMode"},
        )
        source = _string(
            entry["source"],
            field=f"{field}.source",
            maximum=_MAX_TERM_LENGTH,
            allow_empty=False,
        )
        target = _string(
            entry["target"],
            field=f"{field}.target",
            maximum=_MAX_TERM_LENGTH,
            allow_empty=False,
        )
        note = _string(
            entry["note"],
            field=f"{field}.note",
            maximum=_MAX_NOTE_LENGTH,
            allow_empty=True,
        )
        match_mode = _match_mode(
            entry["matchMode"],
            field=f"{field}.matchMode",
        )
        if match_mode == "regex":
            _compile_regex(source, field=f"{field}.source")
        candidate = {
            "source": source,
            "target": target,
            "note": note,
            "matchMode": match_mode,
        }
        key = (match_mode, source)
        existing = seen.get(key)
        if existing is not None:
            raise ValueError(
                f"{field} duplicates an earlier glossary entry for {source}"
            )
        seen[key] = candidate
        normalized.append(candidate)
    return normalized


def _non_translate_entries(value: object) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    seen: dict[tuple[str, str], dict[str, str]] = {}
    for index, raw in enumerate(_entries(value, field="nonTranslate.entries")):
        field = f"nonTranslate.entries[{index}]"
        entry = _mapping(raw, field=field)
        _exact_fields(
            entry,
            field=field,
            expected={"pattern", "note", "matchMode"},
        )
        pattern = _string(
            entry["pattern"],
            field=f"{field}.pattern",
            maximum=_MAX_TERM_LENGTH,
            allow_empty=False,
        )
        note = _string(
            entry["note"],
            field=f"{field}.note",
            maximum=_MAX_NOTE_LENGTH,
            allow_empty=True,
        )
        match_mode = _match_mode(
            entry["matchMode"],
            field=f"{field}.matchMode",
        )
        if match_mode == "regex":
            _compile_regex(pattern, field=f"{field}.pattern")
        candidate = {
            "pattern": pattern,
            "note": note,
            "matchMode": match_mode,
        }
        key = (match_mode, pattern)
        existing = seen.get(key)
        if existing is not None:
            raise ValueError(
                f"{field} duplicates an earlier non-translate entry for {pattern}"
            )
        seen[key] = candidate
        normalized.append(candidate)
    return normalized


def validate_translation_constraints(payload: object) -> dict[str, Any]:
    """Validate and normalize the one canonical persisted v2 payload."""

    document = _mapping(payload, field="translation constraints")
    _exact_fields(
        document,
        field="translation constraints",
        expected={"glossary", "nonTranslate"},
    )
    glossary = _mapping(document["glossary"], field="glossary")
    _exact_fields(
        glossary,
        field="glossary",
        expected={"enabled", "autoExtractEnabled", "autoExtractPrompt", "entries"},
    )
    non_translate = _mapping(document["nonTranslate"], field="nonTranslate")
    _exact_fields(
        non_translate,
        field="nonTranslate",
        expected={"enabled", "entries"},
    )
    prompt = _string(
        glossary["autoExtractPrompt"],
        field="glossary.autoExtractPrompt",
        maximum=_MAX_PROMPT_LENGTH,
        allow_empty=False,
    )
    return {
        "glossary": {
            "enabled": _boolean(glossary["enabled"], field="glossary.enabled"),
            "autoExtractEnabled": _boolean(
                glossary["autoExtractEnabled"],
                field="glossary.autoExtractEnabled",
            ),
            "autoExtractPrompt": prompt,
            "entries": _glossary_entries(glossary["entries"]),
        },
        "nonTranslate": {
            "enabled": _boolean(
                non_translate["enabled"],
                field="nonTranslate.enabled",
            ),
            "entries": _non_translate_entries(non_translate["entries"]),
        },
    }


def with_glossary_delta(
    payload: Mapping[str, Any],
    delta: list[Mapping[str, Any]],
) -> tuple[dict[str, Any], int]:
    """Append non-conflicting text terms and return the normalized payload/count."""

    result = validate_translation_constraints(deepcopy(dict(payload)))
    glossary = result["glossary"]
    existing = {
        (entry["matchMode"], entry["source"])
        for entry in glossary["entries"]
    }
    added = 0
    for entry in _glossary_entries(list(delta)):
        key = (entry["matchMode"], entry["source"])
        if key in existing:
            continue
        glossary["entries"].append(entry)
        existing.add(key)
        added += 1
    return validate_translation_constraints(result), added
