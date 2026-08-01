"""Strict boundary normalization for persisted Insight page analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


ALLOWED_IMPORTANCE = frozenset({"high", "medium", "normal"})
MAX_SUMMARY_CHARS = 20_000
MAX_CONTINUITY_CHARS = 10_000
MAX_EVENTS = 100
MAX_WARNINGS = 100


class InvalidPageAnalysis(ValueError):
    pass


def normalize_page_analysis(
    raw: Mapping[str, Any],
    *,
    page_id: str,
    source_asset_id: str,
    source_checksum: str,
    page_number: int,
) -> dict[str, Any]:
    """Return only the canonical persisted page-analysis fields."""

    page = _extract_page(raw, page_number)
    summary = _bounded_text(
        page.get("page_summary"),
        field="page_summary",
        maximum=MAX_SUMMARY_CHARS,
        required=True,
    )
    continuity = _bounded_text(
        page.get("continuity_notes", ""),
        field="continuity_notes",
        maximum=MAX_CONTINUITY_CHARS,
        required=False,
    )
    raw_events = page.get("key_events", [])
    if raw_events is None:
        raw_events = []
    if (
        not isinstance(raw_events, Sequence)
        or isinstance(raw_events, (str, bytes, bytearray))
        or len(raw_events) > MAX_EVENTS
    ):
        raise InvalidPageAnalysis("key_events must be an array of at most 100 items")
    events: list[dict[str, str]] = []
    for index, event in enumerate(raw_events):
        if not isinstance(event, Mapping):
            raise InvalidPageAnalysis(f"key_events[{index}] must be an object")
        event_summary = _bounded_text(
            event.get("summary"),
            field=f"key_events[{index}].summary",
            maximum=2_000,
            required=True,
        )
        importance = str(event.get("importance", "normal")).strip().lower()
        if importance not in ALLOWED_IMPORTANCE:
            importance = "normal"
        normalized = {
            "summary": event_summary,
            "importance": importance,
        }
        event_type = _bounded_text(
            event.get("event_type", ""),
            field=f"key_events[{index}].event_type",
            maximum=100,
            required=False,
        )
        if event_type:
            normalized["event_type"] = event_type
        events.append(normalized)

    raw_warnings = page.get("warnings", [])
    if raw_warnings is None:
        raw_warnings = []
    if (
        not isinstance(raw_warnings, Sequence)
        or isinstance(raw_warnings, (str, bytes, bytearray))
        or len(raw_warnings) > MAX_WARNINGS
    ):
        raise InvalidPageAnalysis("warnings must be an array of at most 100 items")
    warnings: list[dict[str, str]] = []
    for index, warning in enumerate(raw_warnings):
        if not isinstance(warning, Mapping):
            raise InvalidPageAnalysis(f"warnings[{index}] must be an object")
        code = _bounded_text(
            warning.get("code", "MODEL_WARNING"),
            field=f"warnings[{index}].code",
            maximum=100,
            required=True,
        ).upper()
        message = _bounded_text(
            warning.get("message"),
            field=f"warnings[{index}].message",
            maximum=2_000,
            required=True,
        )
        warnings.append({"code": code, "message": message})

    return {
        "schema_version": 2,
        "page_id": page_id,
        "source_asset_id": source_asset_id,
        "source_checksum": source_checksum,
        "page_number_snapshot": page_number,
        "page_summary": summary,
        "key_events": events,
        "continuity_notes": continuity,
        "warnings": warnings,
    }


def _extract_page(
    raw: Mapping[str, Any],
    page_number: int,
) -> Mapping[str, Any]:
    pages = raw.get("pages")
    if pages is None:
        raise InvalidPageAnalysis("model result must contain pages")
    if (
        not isinstance(pages, Sequence)
        or isinstance(pages, (str, bytes, bytearray))
    ):
        raise InvalidPageAnalysis("pages must be an array")
    if len(pages) == 1:
        only_page = pages[0]
        if not isinstance(only_page, Mapping):
            raise InvalidPageAnalysis("pages[0] must be an object")
        return only_page
    matches = []
    for page in pages:
        if not isinstance(page, Mapping):
            continue
        try:
            candidate = int(page.get("page_number", -1))
        except (TypeError, ValueError):
            continue
        if candidate == page_number:
            matches.append(page)
    if len(matches) != 1:
        raise InvalidPageAnalysis(
            "model result must contain exactly one matching page"
        )
    return matches[0]


def _bounded_text(
    value: object,
    *,
    field: str,
    maximum: int,
    required: bool,
) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise InvalidPageAnalysis(f"{field} must be a string")
    normalized = value.strip()
    if required and not normalized:
        raise InvalidPageAnalysis(f"{field} is required")
    if len(normalized) > maximum:
        raise InvalidPageAnalysis(f"{field} exceeds {maximum} characters")
    return normalized
