"""Strict boundary normalization for persisted Insight page analysis."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any


ALLOWED_IMPORTANCE = frozenset({"high", "medium", "normal"})
PAGE_FIELDS = frozenset(
    {
        "page_number",
        "page_summary",
        "key_events",
        "continuity_notes",
        "warnings",
    }
)
EVENT_FIELDS = frozenset({"summary", "importance", "event_type"})
WARNING_FIELDS = frozenset({"code", "message"})
PERSISTED_FIELDS = frozenset(
    {
        "schema_version",
        "page_id",
        "source_asset_id",
        "source_checksum",
        "page_number_snapshot",
        "page_summary",
        "key_events",
        "continuity_notes",
        "warnings",
    }
)


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

    _require_identity(
        page_id=page_id,
        source_asset_id=source_asset_id,
        source_checksum=source_checksum,
        page_number=page_number,
    )
    page = _extract_page(raw, page_number)
    if set(page) != PAGE_FIELDS:
        raise InvalidPageAnalysis(
            "page analysis must contain exactly the current fields"
        )
    summary = _required_text(page["page_summary"], "page_summary")
    continuity = _text(page["continuity_notes"], "continuity_notes")
    raw_events = page["key_events"]
    if not isinstance(raw_events, list):
        raise InvalidPageAnalysis("key_events must be an array")
    events: list[dict[str, str]] = []
    for index, event in enumerate(raw_events):
        if not isinstance(event, Mapping):
            raise InvalidPageAnalysis(f"key_events[{index}] must be an object")
        if set(event) not in (
            {"summary", "importance"},
            EVENT_FIELDS,
        ):
            raise InvalidPageAnalysis(
                f"key_events[{index}] fields are invalid"
            )
        event_summary = _required_text(
            event["summary"],
            f"key_events[{index}].summary",
        )
        importance = event["importance"]
        if importance not in ALLOWED_IMPORTANCE:
            raise InvalidPageAnalysis(
                f"key_events[{index}].importance is invalid"
            )
        normalized = {
            "summary": event_summary,
            "importance": importance,
        }
        if "event_type" in event:
            normalized["event_type"] = _required_text(
                event["event_type"],
                f"key_events[{index}].event_type",
            )
        events.append(normalized)

    raw_warnings = page["warnings"]
    if not isinstance(raw_warnings, list):
        raise InvalidPageAnalysis("warnings must be an array")
    warnings: list[dict[str, str]] = []
    for index, warning in enumerate(raw_warnings):
        if not isinstance(warning, Mapping):
            raise InvalidPageAnalysis(f"warnings[{index}] must be an object")
        if set(warning) != WARNING_FIELDS:
            raise InvalidPageAnalysis(f"warnings[{index}] fields are invalid")
        code = _required_text(
            warning["code"],
            f"warnings[{index}].code",
        )
        message = _required_text(
            warning["message"],
            f"warnings[{index}].message",
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


def validate_persisted_page_analysis(raw: object) -> dict[str, Any]:
    """Validate the one current persisted page-analysis shape without fallback."""

    if not isinstance(raw, Mapping) or set(raw) != PERSISTED_FIELDS:
        raise InvalidPageAnalysis(
            "persisted page analysis must contain exactly the current fields"
        )
    if raw["schema_version"] != 2:
        raise InvalidPageAnalysis("persisted page analysis schema_version must be 2")
    normalized = normalize_page_analysis(
        {
            "pages": [
                {
                    "page_number": raw["page_number_snapshot"],
                    "page_summary": raw["page_summary"],
                    "key_events": raw["key_events"],
                    "continuity_notes": raw["continuity_notes"],
                    "warnings": raw["warnings"],
                }
            ]
        },
        page_id=raw["page_id"],
        source_asset_id=raw["source_asset_id"],
        source_checksum=raw["source_checksum"],
        page_number=raw["page_number_snapshot"],
    )
    if dict(raw) != normalized:
        raise InvalidPageAnalysis("persisted page analysis is not canonical")
    return normalized


def _extract_page(
    raw: Mapping[str, Any],
    page_number: int,
) -> Mapping[str, Any]:
    if set(raw) != {"pages"}:
        raise InvalidPageAnalysis("model result must contain only pages")
    pages = raw["pages"]
    if not isinstance(pages, list):
        raise InvalidPageAnalysis("pages must be an array")
    if len(pages) != 1 or not isinstance(pages[0], Mapping):
        raise InvalidPageAnalysis("pages must contain exactly one page object")
    page = pages[0]
    actual_page_number = page.get("page_number")
    if (
        isinstance(actual_page_number, bool)
        or not isinstance(actual_page_number, int)
        or actual_page_number != page_number
    ):
        raise InvalidPageAnalysis(
            f"page_number must be {page_number}, got {actual_page_number!r}"
        )
    return page


def _text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise InvalidPageAnalysis(f"{field} must be a string")
    return value.strip()


def _required_text(value: object, field: str) -> str:
    normalized = _text(value, field)
    if not normalized:
        raise InvalidPageAnalysis(f"{field} is required")
    return normalized


def _require_identity(
    *,
    page_id: object,
    source_asset_id: object,
    source_checksum: object,
    page_number: object,
) -> None:
    if not isinstance(page_id, str) or not page_id:
        raise InvalidPageAnalysis("page_id must be a non-empty string")
    if not isinstance(source_asset_id, str) or not source_asset_id:
        raise InvalidPageAnalysis("source_asset_id must be a non-empty string")
    if (
        not isinstance(source_checksum, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_checksum) is None
    ):
        raise InvalidPageAnalysis("source_checksum must be a lowercase SHA-256 digest")
    if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:
        raise InvalidPageAnalysis("page_number must be a positive integer")
