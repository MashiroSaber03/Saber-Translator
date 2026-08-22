"""Versioned Insight overview, timeline, compressed-context, and vector jobs."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any, Callable, Protocol
import uuid

from sqlalchemy import Engine, func, insert, select, update
from sqlalchemy.engine import Connection

from src.backend_v2.auth.ownership import effective_owner_id
from src.backend_v2.redaction import redact_sensitive_text
from src.backend_v2.serialization import canonical_json as _json
from src.backend_v2.insight.page_schema import (
    InvalidPageAnalysis,
    validate_persisted_page_analysis,
)
from src.backend_v2.insight.repository import (
    InsightConflict,
    InsightNotFound,
    OVERVIEW_TEMPLATES,
    contains_nonempty_text,
)
from src.backend_v2.insight.provider_runtime import (
    frozen_chat_config,
    frozen_embedding_config,
)
from src.backend_v2.timestamps import utcnow
from src.backend_v2.jobs.repository import (
    AttemptFence,
    AttemptFenced,
    JobConflict,
    JobItemSpec,
    JobQueueRepository,
    JobSpec,
)
from src.backend_v2.settings.resolver import SettingsResolver
from src.backend_v2.storage.platform_repositories import SettingsRepository
from src.backend_v2.storage.schema import (
    analysis_artifacts,
    analysis_heads,
    analysis_layer_result_pages,
    analysis_layer_results,
    analysis_page_results,
    analysis_run_targets,
    analysis_runs,
    assets,
    books,
    chapters,
    pages,
    page_assets,
    timeline_characters,
    timeline_events,
    timeline_versions,
    vector_generations,
    jobs,
)
from src.shared.memory_errors import is_memory_allocation_error


DERIVED_KINDS = frozenset(
    {"overview", "compressed_context", "timeline", "vector"}
)
FINAL_ANALYSIS_RUN_STATUSES = frozenset(
    {"completed", "completed_with_errors"}
)


def _json_object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise InsightConflict(f"stored {field} is missing; clear current Insight data")
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise InsightConflict(
            f"stored {field} is invalid; clear current Insight data"
        ) from exc
    if not isinstance(parsed, Mapping):
        raise InsightConflict(f"stored {field} must be an object")
    return dict(parsed)


def _required_mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InsightConflict(f"{field} must be an object")
    return dict(value)


def _required_integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InsightConflict(f"{field} must be an integer of at least {minimum}")
    return value


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise InsightConflict(f"{field} must be a non-empty string")
    return value


def _required_text(value: object, field: str) -> str:
    text = _required_string(value, field)
    if not text.strip():
        raise InsightConflict(f"{field} must not be blank")
    return text


def _required_boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise InsightConflict(f"{field} must be a boolean")
    return value


def _required_datetime(value: object, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise InsightConflict(f"{field} must be a timestamp")
    return value


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, field)


def validate_artifact_payload(
    *,
    kind: str,
    template: str,
    payload: object,
) -> dict[str, Any]:
    if kind == "overview":
        if template not in OVERVIEW_TEMPLATES:
            raise InsightConflict("Insight overview template is invalid")
    elif kind == "compressed_context":
        if template != "default":
            raise InsightConflict("Insight compressed context template is invalid")
    else:
        raise InsightConflict("Insight artifact kind is invalid")
    if not isinstance(payload, Mapping) or not contains_nonempty_text(payload):
        raise InsightConflict("Insight artifact payload must be a non-empty object")
    normalized = dict(payload)
    if kind == "overview":
        _required_text(normalized.get("title"), "overview title")
        _required_text(normalized.get("content"), "overview content")
    return normalized


def _required_sha256(value: object, field: str) -> str:
    text = _required_string(value, field)
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise InsightConflict(f"{field} must be a lowercase SHA-256 digest")
    return text


def _persisted_page_analysis(
    value: object,
    *,
    page_id: str,
    page_number: int,
    source_asset_id: str,
    source_checksum: str,
) -> dict[str, Any]:
    try:
        payload = validate_persisted_page_analysis(
            _json_object(value, "analysis page payload")
        )
    except InvalidPageAnalysis as exc:
        raise InsightConflict(
            "stored analysis page payload is invalid; clear current Insight data"
        ) from exc
    if (
        payload["page_id"] != page_id
        or payload["page_number_snapshot"] != page_number
        or payload["source_asset_id"] != source_asset_id
        or payload["source_checksum"] != source_checksum
    ):
        raise InsightConflict(
            "stored analysis page payload identity is invalid; "
            "clear current Insight data"
        )
    return payload


def _safe_timeline_error(error: object) -> str:
    message = redact_sensitive_text(error).strip()
    if not message:
        message = type(error).__name__
    return message[:1000]


def _validate_timeline_parts(
    *,
    content: object,
    events: object,
    characters: object,
    require_events: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(content, Mapping):
        raise ValueError("timeline response content must be an object")
    if not isinstance(events, list) or (require_events and not events):
        raise ValueError("timeline response must contain at least one event")
    if not isinstance(characters, list):
        raise ValueError("timeline response is missing characters")

    validated_events: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise ValueError(f"timeline event {index + 1} must be an object")
        summary = event.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(f"timeline event {index + 1} summary is invalid")
        page_ids = event.get("page_ids")
        if page_ids is not None and (
            not isinstance(page_ids, list)
            or any(not isinstance(value, str) or not value for value in page_ids)
        ):
            raise ValueError(f"timeline event {index + 1} page_ids are invalid")
        page_numbers = event.get("page_numbers")
        if page_numbers is not None and (
            not isinstance(page_numbers, list)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in page_numbers
            )
        ):
            raise ValueError(f"timeline event {index + 1} page_numbers are invalid")
        if not page_ids and not page_numbers:
            raise ValueError(
                f"timeline event {index + 1} must reference at least one page"
            )
        validated_events.append(dict(event))

    validated_characters: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, character in enumerate(characters):
        if not isinstance(character, Mapping):
            raise ValueError(f"timeline character {index + 1} must be an object")
        name = character.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"timeline character {index + 1} name is invalid")
        if name in names:
            raise ValueError(f"timeline character name is duplicated: {name}")
        names.add(name)
        description = character.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                f"timeline character {index + 1} description is invalid"
            )
        first_page = character.get("first_page")
        if (
            isinstance(first_page, bool)
            or not isinstance(first_page, int)
            or first_page < 1
        ):
            raise ValueError(f"timeline character {index + 1} first_page is invalid")
        related_pages = character.get("related_page_numbers")
        if related_pages is not None and (
            not isinstance(related_pages, list)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in related_pages
            )
        ):
            raise ValueError(
                f"timeline character {index + 1} related_page_numbers are invalid"
            )
        key_moments = character.get("key_moments")
        if not isinstance(key_moments, list):
            raise ValueError(
                f"timeline character {index + 1} key_moments are invalid"
            )
        for moment_index, moment in enumerate(key_moments):
            if not isinstance(moment, Mapping):
                raise ValueError(
                    f"timeline character {index + 1} key moment "
                    f"{moment_index + 1} must be an object"
                )
            page = moment.get("page")
            summary = moment.get("summary")
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError(
                    f"timeline character {index + 1} key moment "
                    f"{moment_index + 1} summary is invalid"
                )
            if page is not None and (
                isinstance(page, bool)
                or not isinstance(page, int)
                or page < 1
            ):
                raise ValueError(
                    f"timeline character {index + 1} key moment "
                    f"{moment_index + 1} page is invalid"
                )
        validated_characters.append(dict(character))
    return dict(content), validated_events, validated_characters


def _normalized_timeline_result(
    result: object,
    *,
    mode: str,
    fallback_reason: str | None,
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise ValueError("timeline response must be an object")
    content, events, characters = _validate_timeline_parts(
        content=result.get("content"),
        events=result.get("events"),
        characters=result.get("characters"),
        require_events=True,
    )
    content.update(
        {
            "requested_mode": "enhanced",
            "actual_mode": mode,
            "fallback_reason": fallback_reason,
            "degraded": mode != "enhanced",
        }
    )
    return {
        "mode": mode,
        "content": content,
        "events": events,
        "characters": characters,
    }


def _validate_timeline_content_collections(content: Mapping[str, Any]) -> None:
    plot_arcs = content.get("plot_arcs")
    if plot_arcs is not None:
        if not isinstance(plot_arcs, list):
            raise InsightConflict("timeline plot_arcs must be an array")
        arc_ids: set[str] = set()
        for index, arc in enumerate(plot_arcs, start=1):
            if not isinstance(arc, Mapping):
                raise InsightConflict(f"timeline plot arc {index} must be an object")
            arc_id = _required_text(arc.get("id"), f"timeline plot arc {index} id")
            if arc_id in arc_ids:
                raise InsightConflict(f"timeline plot arc id is duplicated: {arc_id}")
            arc_ids.add(arc_id)
            _required_text(arc.get("name"), f"timeline plot arc {index} name")
            _required_text(
                arc.get("description"),
                f"timeline plot arc {index} description",
            )
            page_range = arc.get("page_range")
            if not isinstance(page_range, Mapping):
                raise InsightConflict(
                    f"timeline plot arc {index} page_range must be an object"
                )
            start = _required_integer(
                page_range.get("start"),
                f"timeline plot arc {index} start page",
                minimum=1,
            )
            end = _required_integer(
                page_range.get("end"),
                f"timeline plot arc {index} end page",
                minimum=1,
            )
            if end < start:
                raise InsightConflict(
                    f"timeline plot arc {index} page range is reversed"
                )
            for field in ("mood",):
                if arc.get(field) is not None:
                    _required_text(
                        arc.get(field),
                        f"timeline plot arc {index} {field}",
                    )
            event_ids = arc.get("event_ids")
            if event_ids is not None and (
                not isinstance(event_ids, list)
                or any(not isinstance(value, str) or not value for value in event_ids)
                or len(set(event_ids)) != len(event_ids)
            ):
                raise InsightConflict(
                    f"timeline plot arc {index} event_ids are invalid"
                )

    plot_threads = content.get("plot_threads")
    if plot_threads is None:
        return
    if not isinstance(plot_threads, list):
        raise InsightConflict("timeline plot_threads must be an array")
    thread_ids: set[str] = set()
    for index, thread in enumerate(plot_threads, start=1):
        if not isinstance(thread, Mapping):
            raise InsightConflict(f"timeline plot thread {index} must be an object")
        thread_id = _required_text(
            thread.get("id"),
            f"timeline plot thread {index} id",
        )
        if thread_id in thread_ids:
            raise InsightConflict(f"timeline plot thread id is duplicated: {thread_id}")
        thread_ids.add(thread_id)
        for field in ("name", "type", "status"):
            _required_text(
                thread.get(field),
                f"timeline plot thread {index} {field}",
            )
        if thread.get("description") is not None:
            _required_text(
                thread.get("description"),
                f"timeline plot thread {index} description",
            )
        for field in ("introduced_at", "resolved_at"):
            if thread.get(field) is not None:
                _required_integer(
                    thread.get(field),
                    f"timeline plot thread {index} {field}",
                    minimum=1,
                )


def _validate_timeline_metadata(
    content: Mapping[str, Any],
    *,
    mode: str,
) -> None:
    story_summary = content.get("story_summary")
    if not isinstance(story_summary, str):
        raise InsightConflict("timeline story_summary must be a string")
    if mode != "simple" and not story_summary.strip():
        raise InsightConflict(
            "enhanced timeline story_summary must be a non-empty string"
        )
    if content.get("requested_mode") != "enhanced":
        raise InsightConflict("timeline requested_mode is invalid")
    if content.get("actual_mode") != mode:
        raise InsightConflict("timeline actual_mode does not match mode")
    degraded = _required_boolean(content.get("degraded"), "timeline degraded")
    if degraded != (mode != "enhanced"):
        raise InsightConflict("timeline degraded flag does not match mode")
    fallback_reason = content.get("fallback_reason")
    if mode == "enhanced":
        if fallback_reason is not None:
            raise InsightConflict(
                "enhanced timeline fallback_reason must be null"
            )
    elif not isinstance(fallback_reason, str) or not fallback_reason.strip():
        raise InsightConflict(
            "degraded timeline fallback_reason must be a non-empty string"
        )
    _validate_timeline_content_collections(content)


def validate_timeline_payload(
    *,
    mode: str,
    content: object,
    events: object,
    characters: object,
    require_events: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if mode not in {"enhanced", "compressed", "simple"}:
        raise InsightConflict("timeline mode is invalid")
    try:
        validated = _validate_timeline_parts(
            content=content,
            events=events,
            characters=characters,
            require_events=require_events,
        )
    except ValueError as exc:
        raise InsightConflict("timeline payload is invalid") from exc
    _validate_timeline_metadata(validated[0], mode=mode)
    return validated


def _canonical_timeline_references(
    *,
    frozen: AnalysisInputSnapshot,
    events: Sequence[Mapping[str, Any]],
    characters: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    page_ids_by_number: dict[int, str] = {}
    page_numbers_by_id: dict[str, int] = {}
    for index, page in enumerate(frozen.pages, start=1):
        page_id = _required_string(
            page.get("pageId"),
            f"timeline source page {index} pageId",
        )
        page_number = _required_integer(
            page.get("pageNumber"),
            f"timeline source page {index} pageNumber",
            minimum=1,
        )
        if page_id in page_numbers_by_id or page_number in page_ids_by_number:
            raise InsightConflict("timeline source pages are duplicated")
        page_numbers_by_id[page_id] = page_number
        page_ids_by_number[page_number] = page_id

    canonical_events: list[dict[str, Any]] = []
    for index, event in enumerate(events, start=1):
        page_ids = list(event.get("page_ids") or [])
        page_numbers = list(event.get("page_numbers") or [])
        if len(set(page_ids)) != len(page_ids) or len(set(page_numbers)) != len(
            page_numbers
        ):
            raise InsightConflict(f"timeline event {index} page references are duplicated")
        if page_ids:
            if any(page_id not in page_numbers_by_id for page_id in page_ids):
                raise InsightConflict(
                    f"timeline event {index} references an unknown page"
                )
            expected_numbers = [page_numbers_by_id[page_id] for page_id in page_ids]
            if page_numbers and page_numbers != expected_numbers:
                raise InsightConflict(
                    f"timeline event {index} page references do not match"
                )
            page_numbers = expected_numbers
        else:
            if any(page_number not in page_ids_by_number for page_number in page_numbers):
                raise InsightConflict(
                    f"timeline event {index} references an unknown page"
                )
            page_ids = [page_ids_by_number[page_number] for page_number in page_numbers]
        canonical_events.append(
            {
                **dict(event),
                "page_ids": page_ids,
                "page_numbers": page_numbers,
            }
        )

    canonical_characters: list[dict[str, Any]] = []
    valid_page_numbers = set(page_ids_by_number)
    for index, character in enumerate(characters, start=1):
        referenced = [character["first_page"]]
        referenced.extend(character.get("related_page_numbers") or [])
        referenced.extend(
            moment["page"]
            for moment in character["key_moments"]
            if moment.get("page") is not None
        )
        if any(page_number not in valid_page_numbers for page_number in referenced):
            raise InsightConflict(
                f"timeline character {index} references an unknown page"
            )
        canonical_characters.append(dict(character))
    return canonical_events, canonical_characters


def _timeline_thumbnail_page_numbers(
    *,
    content: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    characters: Sequence[Mapping[str, Any]],
) -> set[int]:
    numbers: set[int] = set()

    def add(value: object, field: str) -> None:
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise InsightConflict(
                f"stored timeline {field} is invalid; clear current Insight data"
            )
        numbers.add(value)

    for event in events:
        page_numbers = event.get("page_numbers", [])
        if not isinstance(page_numbers, list):
            raise InsightConflict(
                "stored timeline event page_numbers are invalid; "
                "clear current Insight data"
            )
        for value in page_numbers:
            add(value, "event page number")
    for character in characters:
        add(character.get("first_page"), "character first page")
        related_pages = character.get("related_page_numbers", [])
        if not isinstance(related_pages, list):
            raise InsightConflict(
                "stored timeline character page numbers are invalid; "
                "clear current Insight data"
            )
        for value in related_pages:
            add(value, "character related page")
        key_moments = character.get("key_moments", [])
        if not isinstance(key_moments, list):
            raise InsightConflict(
                "stored timeline character moments are invalid; "
                "clear current Insight data"
            )
        for moment in key_moments:
            if not isinstance(moment, Mapping):
                raise InsightConflict(
                    "stored timeline character moment is invalid; "
                    "clear current Insight data"
                )
            add(moment.get("page"), "character moment page")
    plot_arcs = content.get("plot_arcs", [])
    if not isinstance(plot_arcs, list):
        raise InsightConflict("stored timeline plot_arcs are invalid; clear current Insight data")
    for arc in plot_arcs:
        if not isinstance(arc, Mapping):
            raise InsightConflict("stored timeline plot arc is invalid; clear current Insight data")
        page_range = arc.get("page_range")
        if page_range is not None:
            if not isinstance(page_range, Mapping):
                raise InsightConflict(
                    "stored timeline plot arc page range is invalid; "
                    "clear current Insight data"
                )
            add(page_range.get("start"), "plot arc start page")
    plot_threads = content.get("plot_threads", [])
    if not isinstance(plot_threads, list):
        raise InsightConflict(
            "stored timeline plot_threads are invalid; clear current Insight data"
        )
    for thread in plot_threads:
        if not isinstance(thread, Mapping):
            raise InsightConflict(
                "stored timeline plot thread is invalid; clear current Insight data"
            )
        add(thread.get("introduced_at"), "plot thread introduced page")
        add(thread.get("resolved_at"), "plot thread resolved page")
    return numbers


@dataclass(frozen=True, slots=True)
class AnalysisInputSnapshot:
    book_id: str
    source_run_id: str | None
    source_run_status: str | None
    result_ids: tuple[str, ...]
    pages: tuple[dict[str, Any], ...]
    fingerprint: str


class DerivedAlgorithms(Protocol):
    def build_layer(
        self,
        inputs: Sequence[Mapping[str, Any]],
        *,
        layer: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def build_overview(
        self,
        pages: Sequence[Mapping[str, Any]],
        *,
        template: str,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def build_compressed_context(
        self,
        pages: Sequence[Mapping[str, Any]],
        *,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def build_timeline(
        self,
        pages: Sequence[Mapping[str, Any]],
        *,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def embed_documents(
        self,
        documents: Sequence[str],
        *,
        config: Mapping[str, Any],
    ) -> Sequence[Sequence[float]]: ...


class ProviderDerivedAlgorithms:
    """Current derived-analysis implementation for the Worker."""

    def build_layer(
        self,
        inputs: Sequence[Mapping[str, Any]],
        *,
        layer: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        prompt_type = _required_string(
            layer.get("promptType"),
            "Insight layer promptType",
        )
        layer_name = _required_string(layer.get("name"), "Insight layer name")
        prompt = (
            f"请生成“{layer_name}”层级摘要。"
            "只依据输入，保留关键事件、连续性和因果关系。输出 JSON。\n\n"
            + "\n\n".join(_json(dict(value)) for value in inputs)
        )
        result = self._chat_json(
            prompt,
            config=config,
            prompt_type=prompt_type,
        )
        if not isinstance(result, Mapping) or not contains_nonempty_text(result):
            raise ValueError("summary layer response must be a non-empty object")
        return dict(result)

    def build_overview(
        self,
        pages: Sequence[Mapping[str, Any]],
        *,
        template: str,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        prompt = (
            f"请根据以下逐页分析生成“{template}”漫画概览。"
            "只依据输入，不补写不存在的情节。输出 JSON，至少包含 title 与 content。\n\n"
            + _page_context(pages)
        )
        result = self._chat_json(prompt, config=config, prompt_type="book_overview")
        if not isinstance(result, Mapping):
            raise ValueError("overview response must be an object")
        title = result.get("title")
        content = result.get("content")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("overview response title must be a non-empty string")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("overview response content must be a non-empty string")
        return dict(result)

    def build_compressed_context(
        self,
        pages: Sequence[Mapping[str, Any]],
        *,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        prompt = (
            "把以下漫画逐页分析压缩成可供后续问答和剧情生成使用的上下文。"
            "保留事件顺序、因果、角色状态变化和未解决线索。输出 JSON。\n\n"
            + _page_context(pages)
        )
        result = self._chat_json(prompt, config=config, prompt_type="group_summary")
        if not isinstance(result, Mapping) or not contains_nonempty_text(result):
            raise ValueError(
                "compressed context response must be a non-empty object"
            )
        return dict(result)

    def build_timeline(
        self,
        pages: Sequence[Mapping[str, Any]],
        *,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        enhanced_prompt = (
            "根据以下漫画分析生成增强时间线。输出 JSON："
            '{"content":{"story_summary":"...","plot_arcs":'
            '[{"id":"...","name":"...","description":"...",'
            '"page_range":{"start":1,"end":2},"mood":"...",'
            '"event_ids":["..."]}],"plot_threads":[]},'
            '"events":[{"summary":"...","page_ids":["..."]}],'
            '"characters":[{"name":"...","aliases":[],"description":"...",'
            '"personality":"...","arc":"...","first_page":1,'
            '"key_moments":[{"summary":"...","page":1}],'
            '"related_page_numbers":[1]}]}。'
            "不要把推断写成事实。\n\n"
            + _page_context(pages)
        )
        enhanced_error: Exception | None = None
        try:
            result = self._chat_json(
                enhanced_prompt,
                config=config,
                prompt_type="book_overview",
            )
            return _normalized_timeline_result(
                result,
                mode="enhanced",
                fallback_reason=None,
            )
        except Exception as exc:
            if is_memory_allocation_error(exc):
                raise
            enhanced_error = exc

        compressed_payloads: list[dict[str, Any]] = []
        for index, page in enumerate(pages):
            analysis = _required_mapping(
                page.get("analysis"),
                f"timeline input {index + 1} analysis",
            )
            compressed_context = analysis.get("compressed_context")
            if compressed_context is None:
                continue
            compressed_payload = _required_mapping(
                compressed_context,
                f"timeline input {index + 1} compressed_context",
            )
            if compressed_payload:
                compressed_payloads.append(compressed_payload)
        compressed_error: Exception | None = None
        if compressed_payloads:
            compressed_prompt = (
                "根据以下压缩上下文生成漫画时间线。输出 JSON，必须包含 "
                "content、events 和 characters；事件使用 page_ids 或 page_numbers "
                "关联来源页面。不要补写上下文中不存在的事实。\n\n"
                + "\n\n".join(_json(value) for value in compressed_payloads)
            )
            try:
                result = self._chat_json(
                    compressed_prompt,
                    config=config,
                    prompt_type="book_overview",
                )
                return _normalized_timeline_result(
                    result,
                    mode="compressed",
                    fallback_reason=_safe_timeline_error(enhanced_error),
                )
            except Exception as exc:
                if is_memory_allocation_error(exc):
                    raise
                compressed_error = exc

        events = []
        story_summary = ""
        for page_index, page in enumerate(pages):
            payload = _required_mapping(
                page.get("analysis"),
                f"timeline input {page_index + 1} analysis",
            )
            compressed_context_value = payload.get("compressed_context")
            compressed_context = (
                _required_mapping(
                    compressed_context_value,
                    f"timeline input {page_index + 1} compressed_context",
                )
                if compressed_context_value is not None
                else {}
            )
            if compressed_context and not story_summary:
                for key in ("story_summary", "summary", "content"):
                    value = compressed_context.get(key)
                    if value is None:
                        continue
                    if not isinstance(value, str):
                        raise InsightConflict(
                            f"timeline input {page_index + 1} {key} must be a string"
                        )
                    if value:
                        story_summary = value
                        break
            raw_events = payload.get("key_events", [])
            if not isinstance(raw_events, list):
                raise InsightConflict(
                    f"timeline input {page_index + 1} key_events must be an array"
                )
            raw_page_ids = page.get("pageIds")
            if raw_page_ids is None:
                raw_page_ids = [page.get("pageId")]
            if not isinstance(raw_page_ids, list) or any(
                not isinstance(value, str) or not value for value in raw_page_ids
            ):
                raise InsightConflict(
                    f"timeline input {page_index + 1} pageIds are invalid"
                )
            raw_page_numbers = page.get("pageNumbers")
            if raw_page_numbers is None:
                raw_page_numbers = [page.get("pageNumber")]
            if not isinstance(raw_page_numbers, list) or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in raw_page_numbers
            ):
                raise InsightConflict(
                    f"timeline input {page_index + 1} pageNumbers are invalid"
                )
            for event_index, event in enumerate(raw_events):
                if not isinstance(event, Mapping):
                    raise InsightConflict(
                        f"timeline input {page_index + 1} event "
                        f"{event_index + 1} must be an object"
                    )
                summary = event.get("summary")
                importance = event.get("importance", "normal")
                if not isinstance(summary, str) or not summary.strip():
                    raise InsightConflict(
                        f"timeline input {page_index + 1} event "
                        f"{event_index + 1} summary is invalid"
                    )
                if not isinstance(importance, str):
                    raise InsightConflict(
                        f"timeline input {page_index + 1} event "
                        f"{event_index + 1} importance is invalid"
                    )
                events.append(
                    {
                        "summary": summary,
                        "importance": importance,
                        "page_ids": list(raw_page_ids),
                        "page_numbers": list(raw_page_numbers),
                    }
                )
        if not events:
            reasons = [f"enhanced: {_safe_timeline_error(enhanced_error)}"]
            if compressed_error is not None:
                reasons.append(
                    f"compressed: {_safe_timeline_error(compressed_error)}"
                )
            raise InsightConflict(
                "timeline generation failed in every mode; " + "; ".join(reasons)
            )
        fallback_reason = _safe_timeline_error(enhanced_error)
        if compressed_error is not None:
            fallback_reason += (
                f"; compressed: {_safe_timeline_error(compressed_error)}"
            )
        return {
            "mode": "simple",
            "content": {
                "story_summary": story_summary,
                "requested_mode": "enhanced",
                "actual_mode": "simple",
                "fallback_reason": fallback_reason,
                "degraded": True,
                "source": "page_key_events",
            },
            "events": events,
            "characters": [],
        }

    def embed_documents(
        self,
        documents: Sequence[str],
        *,
        config: Mapping[str, Any],
    ) -> Sequence[Sequence[float]]:
        from src.core.manga_insight.embedding_client import EmbeddingClient

        client = EmbeddingClient(frozen_embedding_config(config))

        async def execute() -> Sequence[Sequence[float]]:
            return await client.embed_batch(list(documents))

        return asyncio.run(execute())

    @staticmethod
    def _chat_json(
        prompt: str,
        *,
        config: Mapping[str, Any],
        prompt_type: str,
    ) -> object:
        from src.core.manga_insight.embedding_client import ChatClient

        chat_config = frozen_chat_config(config)
        prompts = _required_mapping(
            config.get("prompts"),
            "frozen Insight prompts",
        )
        system_section = _required_mapping(
            prompts.get("analysis_system"),
            "frozen Insight analysis_system prompt",
        )
        prompt_section = _required_mapping(
            prompts.get(prompt_type),
            f"frozen Insight {prompt_type} prompt",
        )
        system = _required_string(
            system_section.get("content"),
            "frozen Insight analysis_system prompt content",
        )
        configured = _required_string(
            prompt_section.get("content"),
            f"frozen Insight {prompt_type} prompt content",
        )
        client = ChatClient(chat_config)

        async def execute() -> object:
            return await client.generate_json(
                f"{configured}\n\n{prompt}".strip(),
                system=system,
            )

        return asyncio.run(execute())


def _validated_vector_batch(
    records: Sequence[Mapping[str, Any]],
    embeddings: Sequence[Sequence[float]],
    *,
    field: str,
    expected_dimension: int | None,
) -> tuple[
    list[str],
    list[str],
    list[dict[str, str | int | float | bool]],
    list[list[float]],
    int,
]:
    if not records or len(records) != len(embeddings):
        raise InsightConflict(f"{field} embedding result count mismatch")
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | int | float | bool]] = []
    vectors: list[list[float]] = []
    dimension = expected_dimension
    for index, (record, embedding) in enumerate(
        zip(records, embeddings, strict=True),
        start=1,
    ):
        if not isinstance(record, Mapping) or set(record) != {
            "id",
            "document",
            "metadata",
        }:
            raise InsightConflict(f"{field} vector record {index} fields are invalid")
        record_id = _required_string(
            record["id"],
            f"{field} vector record {index} id",
        )
        document = _required_string(
            record["document"],
            f"{field} vector record {index} document",
        )
        metadata = _required_mapping(
            record["metadata"],
            f"{field} vector record {index} metadata",
        )
        normalized_metadata: dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            if not isinstance(key, str) or not key:
                raise InsightConflict(
                    f"{field} vector record {index} metadata key is invalid"
                )
            if not isinstance(value, (str, int, float, bool)):
                raise InsightConflict(
                    f"{field} vector record {index} metadata value is invalid"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise InsightConflict(
                    f"{field} vector record {index} metadata value is invalid"
                )
            normalized_metadata[key] = value
        if (
            not isinstance(embedding, Sequence)
            or isinstance(embedding, (str, bytes, bytearray))
            or not embedding
            or any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                for value in embedding
            )
        ):
            raise InsightConflict(f"{field} embedding {index} is invalid")
        if dimension is None:
            dimension = len(embedding)
        elif len(embedding) != dimension:
            raise InsightConflict(f"{field} embedding dimensions do not match")
        ids.append(record_id)
        documents.append(document)
        metadatas.append(normalized_metadata)
        vectors.append([float(value) for value in embedding])
    if len(set(ids)) != len(ids):
        raise InsightConflict(f"{field} vector record ids are duplicated")
    if dimension is None:
        raise InsightConflict(f"{field} embedding dimension is missing")
    return ids, documents, metadatas, vectors, dimension


@dataclass(frozen=True, slots=True)
class VectorCollectionInspection:
    expected: tuple[str, ...]
    actual: tuple[str, ...]
    missing: tuple[str, ...]
    orphaned: tuple[str, ...]


class InsightVectorStore:
    """Generation-isolated Chroma collections owned exclusively by Worker."""

    def __init__(self, data_root: Path) -> None:
        self.path = data_root / "chroma"

    @staticmethod
    def names(book_id: str, generation: int) -> tuple[str, str]:
        prefix = hashlib.sha256(book_id.encode("utf-8")).hexdigest()[:20]
        return (
            f"b{prefix}_g{generation}_pages",
            f"b{prefix}_g{generation}_events",
        )

    def publish(
        self,
        *,
        book_id: str,
        generation: int,
        page_records: Sequence[Mapping[str, Any]],
        page_embeddings: Sequence[Sequence[float]],
        event_records: Sequence[Mapping[str, Any]],
        event_embeddings: Sequence[Sequence[float]],
    ) -> None:
        self.publish_batches(
            book_id=book_id,
            generation=generation,
            page_batches=((page_records, page_embeddings),) if page_records else (),
            event_batches=((event_records, event_embeddings),) if event_records else (),
            expected_page_count=len(page_records),
            expected_event_count=len(event_records),
        )

    def publish_batches(
        self,
        *,
        book_id: str,
        generation: int,
        page_batches: Iterable[
            tuple[Sequence[Mapping[str, Any]], Sequence[Sequence[float]]]
        ],
        event_batches: Iterable[
            tuple[Sequence[Mapping[str, Any]], Sequence[Sequence[float]]]
        ],
        resume: bool = False,
        initial_page_count: int = 0,
        initial_event_count: int = 0,
        expected_page_count: int | None = None,
        expected_event_count: int | None = None,
        on_batch: Callable[[str, int], bool] | None = None,
    ) -> dict[str, object]:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise InsightConflict("ChromaDB is not installed") from exc
        self.path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(self.path),
            settings=Settings(anonymized_telemetry=False),
        )
        page_name, event_name = self.names(book_id, generation)
        if resume:
            pages_collection = client.get_or_create_collection(
                page_name,
                metadata={"hnsw:space": "cosine"},
            )
            events_collection = client.get_or_create_collection(
                event_name,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            existing = {
                _required_string(
                    getattr(collection, "name", collection),
                    "Chroma collection name",
                )
                for collection in client.list_collections()
            }
            for name in (page_name, event_name):
                if name in existing:
                    client.delete_collection(name)
            pages_collection = client.create_collection(
                page_name,
                metadata={"hnsw:space": "cosine"},
            )
            events_collection = client.create_collection(
                event_name,
                metadata={"hnsw:space": "cosine"},
            )
        page_count = initial_page_count
        event_count = initial_event_count
        page_dimension: int | None = None
        event_dimension: int | None = None
        try:
            for page_records, page_embeddings in page_batches:
                (
                    page_ids,
                    page_documents,
                    page_metadatas,
                    page_vectors,
                    page_dimension,
                ) = _validated_vector_batch(
                    page_records,
                    page_embeddings,
                    field="page",
                    expected_dimension=page_dimension,
                )
                pages_collection.upsert(
                    ids=page_ids,
                    embeddings=page_vectors,
                    documents=page_documents,
                    metadatas=page_metadatas,
                )
                page_count += len(page_records)
                if on_batch is not None and not on_batch("pages", page_count):
                    return {
                        "completed": False,
                        "pageCount": page_count,
                        "eventCount": event_count,
                    }
            for event_records, event_embeddings in event_batches:
                (
                    event_ids,
                    event_documents,
                    event_metadatas,
                    event_vectors,
                    event_dimension,
                ) = _validated_vector_batch(
                    event_records,
                    event_embeddings,
                    field="event",
                    expected_dimension=event_dimension,
                )
                events_collection.upsert(
                    ids=event_ids,
                    embeddings=event_vectors,
                    documents=event_documents,
                    metadatas=event_metadatas,
                )
                event_count += len(event_records)
                if on_batch is not None and not on_batch("events", event_count):
                    return {
                        "completed": False,
                        "pageCount": page_count,
                        "eventCount": event_count,
                    }
            if (
                expected_page_count is not None
                and pages_collection.count() != expected_page_count
            ):
                raise InsightConflict("page vector coverage is incomplete")
            if (
                expected_event_count is not None
                and events_collection.count() != expected_event_count
            ):
                raise InsightConflict("event vector coverage is incomplete")
            return {
                "completed": True,
                "pageCount": page_count,
                "eventCount": event_count,
            }
        except AttemptFenced:
            # A newer attempt owns publication. Keep the last fenced checkpoint;
            # the replacement attempt will idempotently upsert from that offset.
            raise
        except Exception:
            for name in (page_name, event_name):
                try:
                    client.delete_collection(name)
                except Exception:
                    pass
            raise

    def expected_collection_names(self, engine: Engine) -> set[str]:
        expected: set[str] = set()
        with engine.connect() as connection:
            rows = connection.execute(
                select(
                    vector_generations.c.book_id,
                    vector_generations.c.generation,
                ).where(vector_generations.c.status != "failed")
            )
            for book_id, generation in rows:
                expected.update(
                    self.names(
                        _required_string(book_id, "vector generation bookId"),
                        _required_integer(
                            generation,
                            "vector generation number",
                            minimum=1,
                        ),
                    )
                )
        return expected

    def inspect_collections(self, engine: Engine) -> VectorCollectionInspection:
        expected = self.expected_collection_names(engine)
        if not self.path.exists() or not any(self.path.iterdir()):
            return VectorCollectionInspection(
                expected=tuple(sorted(expected)),
                actual=(),
                missing=tuple(sorted(expected)),
                orphaned=(),
            )
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise InsightConflict("ChromaDB is not installed") from exc
        client = chromadb.PersistentClient(
            path=str(self.path),
            settings=Settings(anonymized_telemetry=False),
        )
        actual = {
            _required_string(
                getattr(collection, "name", collection),
                "Chroma collection name",
            )
            for collection in client.list_collections()
        }
        managed = {
            name
            for name in actual
            if re.fullmatch(r"b[0-9a-f]{20}_g[1-9][0-9]*_(?:pages|events)", name)
        }
        return VectorCollectionInspection(
            expected=tuple(sorted(expected)),
            actual=tuple(sorted(actual)),
            missing=tuple(sorted(expected - actual)),
            orphaned=tuple(sorted(managed - expected)),
        )

    def collect_orphan_collections(self, engine: Engine) -> int:
        inspection = self.inspect_collections(engine)
        if not inspection.orphaned:
            return 0
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=str(self.path),
            settings=Settings(anonymized_telemetry=False),
        )
        deleted = 0
        for name in inspection.orphaned:
            client.delete_collection(name)
            deleted += 1
        return deleted


class InsightDerivedRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    @staticmethod
    def _assert_book_owned(connection: Connection, book_id: str) -> None:
        if connection.execute(
            select(books.c.id).where(
                books.c.id == book_id,
                books.c.kind == "library",
                books.c.owner_user_id == effective_owner_id(),
            )
        ).scalar_one_or_none() is None:
            raise InsightNotFound("book not found")

    def snapshot(
        self,
        *,
        book_id: str,
        frozen_inputs: Sequence[Mapping[str, Any]] | None = None,
    ) -> AnalysisInputSnapshot:
        with self.engine.connect() as connection:
            self._assert_book_owned(connection, book_id)
            return self._snapshot(
                connection,
                book_id=book_id,
                frozen_inputs=frozen_inputs,
            )

    def snapshot_in_transaction(
        self,
        connection: Connection,
        *,
        book_id: str,
        frozen_inputs: Sequence[Mapping[str, Any]] | None = None,
    ) -> AnalysisInputSnapshot:
        """Read the current analysis snapshot inside a caller-owned transaction."""

        self._assert_book_owned(connection, book_id)
        return self._snapshot(
            connection,
            book_id=book_id,
            frozen_inputs=frozen_inputs,
        )

    def snapshot_for_run(self, *, run_id: str) -> AnalysisInputSnapshot:
        """Read only successful staging results from one isolated full run."""

        source_pointer = page_assets.alias("run_snapshot_source")
        with self.engine.connect() as connection:
            run = connection.execute(
                select(
                    analysis_runs.c.book_id,
                    analysis_runs.c.status,
                ).where(
                    analysis_runs.c.id == run_id,
                    analysis_runs.c.owner_user_id == effective_owner_id(),
                )
            ).mappings().one_or_none()
            if run is None:
                raise InsightNotFound("analysis run not found")
            rows = list(
                connection.execute(
                    select(
                        analysis_page_results,
                        analysis_run_targets.c.chapter_id,
                        analysis_run_targets.c.ordinal.label("target_ordinal"),
                        source_pointer.c.asset_id.label(
                            "current_source_asset_id"
                        ),
                        assets.c.checksum.label("current_source_checksum"),
                    )
                    .join(
                        analysis_run_targets,
                        (analysis_run_targets.c.run_id == analysis_page_results.c.run_id)
                        & (
                            analysis_run_targets.c.page_id_snapshot
                            == analysis_page_results.c.page_id_snapshot
                        ),
                    )
                    .join(
                        source_pointer,
                        (source_pointer.c.page_id == analysis_run_targets.c.page_id)
                        & (source_pointer.c.role == "source"),
                    )
                    .join(assets, assets.c.id == source_pointer.c.asset_id)
                    .where(
                        analysis_page_results.c.run_id == run_id,
                        analysis_run_targets.c.status == "completed",
                    )
                    .order_by(analysis_run_targets.c.ordinal)
                ).mappings()
            )
        if not rows:
            raise InsightConflict("analysis run has no successful page results")
        run_status = _required_string(run["status"], "analysis run status")
        if run_status != "staging":
            raise InsightConflict("analysis run is not staging")
        page_payloads: list[dict[str, Any]] = []
        seen_page_ids: set[str] = set()
        previous_ordinal = 0
        for index, row in enumerate(rows, start=1):
            target_ordinal = _required_integer(
                row["target_ordinal"],
                f"analysis target {index} ordinal",
                minimum=1,
            )
            if target_ordinal <= previous_ordinal:
                raise InsightConflict(
                    "stored analysis target order is invalid; "
                    "clear current Insight data"
                )
            previous_ordinal = target_ordinal
            if _required_string(
                row["status"],
                "analysis page result status",
            ) != "staging" or _required_integer(
                row["schema_version"],
                "analysis page result schema version",
                minimum=1,
            ) != 2:
                raise InsightConflict(
                    "stored staging page analysis is invalid; "
                    "clear current Insight data"
                )
            page_id = _required_string(
                row["page_id_snapshot"],
                "analysis page result pageId",
            )
            if page_id in seen_page_ids:
                raise InsightConflict(
                    "stored analysis page results are duplicated; "
                    "clear current Insight data"
                )
            seen_page_ids.add(page_id)
            if _required_string(
                row["page_id"],
                "analysis page result current pageId",
            ) != page_id:
                raise InsightConflict(
                    "stored analysis page identity is invalid; "
                    "clear current Insight data"
                )
            page_number = _required_integer(
                row["page_number_snapshot"],
                "analysis page result pageNumber",
                minimum=1,
            )
            source_asset_id = _required_string(
                row["source_asset_id"],
                "analysis page result source asset id",
            )
            source_checksum = _required_sha256(
                row["source_checksum"],
                "analysis page result source checksum",
            )
            if (
                _required_string(
                    row["current_source_asset_id"],
                    "analysis page current source asset id",
                )
                != source_asset_id
                or _required_sha256(
                    row["current_source_checksum"],
                    "analysis page current source checksum",
                )
                != source_checksum
            ):
                raise InsightConflict(
                    "analysis page source changed before derived processing"
                )
            page_payloads.append(
                {
                    "resultId": _required_string(
                        row["id"],
                        "analysis page result id",
                    ),
                    "pageId": page_id,
                    "pageNumber": page_number,
                    "chapterId": _optional_string(
                        row["chapter_id"],
                        "analysis page result chapterId",
                    ),
                    "sourceChecksum": source_checksum,
                    "currentSourceChecksum": source_checksum,
                    "analysis": _persisted_page_analysis(
                        row["payload_json"],
                        page_id=page_id,
                        page_number=page_number,
                        source_asset_id=source_asset_id,
                        source_checksum=source_checksum,
                    ),
                }
            )
        pages_payload = tuple(page_payloads)
        fingerprint = _analysis_input_fingerprint(pages_payload)
        return AnalysisInputSnapshot(
            book_id=_required_string(run["book_id"], "analysis run bookId"),
            source_run_id=run_id,
            source_run_status=run_status,
            result_ids=tuple(
                _required_string(row["id"], "analysis page result id")
                for row in rows
            ),
            pages=pages_payload,
            fingerprint=fingerprint,
        )

    @staticmethod
    def _snapshot(
        connection: Connection,
        *,
        book_id: str,
        frozen_inputs: Sequence[Mapping[str, Any]] | None = None,
    ) -> AnalysisInputSnapshot:
        book_head = connection.execute(
                select(
                    analysis_heads.c.active_run_id,
                    analysis_heads.c.updated_at.label("head_updated_at"),
                    analysis_runs.c.status,
                )
                .join(
                    analysis_runs,
                    analysis_runs.c.id == analysis_heads.c.active_run_id,
                )
                .where(
                    analysis_heads.c.book_id == book_id,
                    analysis_heads.c.page_id.is_(None),
                )
            ).mappings().one_or_none()
        source_pointer = page_assets.alias("derived_current_source")
        numbered_pages = (
            select(
                pages.c.id.label("page_id"),
                pages.c.chapter_id,
                chapters.c.ordinal.label("chapter_ordinal"),
                pages.c.ordinal.label("page_ordinal"),
                func.row_number()
                .over(order_by=(chapters.c.ordinal, pages.c.ordinal))
                .label("current_page_number"),
            )
            .join(chapters, chapters.c.id == pages.c.chapter_id)
            .where(chapters.c.book_id == book_id)
            .subquery("derived_numbered_pages")
        )
        if frozen_inputs is None:
            active_run_id: str | None = None
            active_head_updated_at: datetime | None = None
            active_target_statuses: dict[str, str] = {}
            if book_head is not None:
                active_run_id = _required_string(
                    book_head["active_run_id"],
                    "active Insight run id",
                )
                active_head_updated_at = _required_datetime(
                    book_head["head_updated_at"],
                    "active Insight book head updatedAt",
                )
                active_run_status = _required_string(
                    book_head["status"],
                    "active Insight run status",
                )
                if active_run_status not in FINAL_ANALYSIS_RUN_STATUSES:
                    raise InsightConflict(
                        "active Insight run status is invalid; "
                        "clear current Insight data"
                    )
                for target in connection.execute(
                    select(
                        analysis_run_targets.c.page_id_snapshot,
                        analysis_run_targets.c.status,
                    ).where(
                        analysis_run_targets.c.run_id == active_run_id
                    )
                ).mappings():
                    target_page_id = _required_string(
                        target["page_id_snapshot"],
                        "active Insight target page id",
                    )
                    target_status = _required_string(
                        target["status"],
                        "active Insight target status",
                    )
                    if (
                        target_page_id in active_target_statuses
                        or target_status
                        not in {"completed", "failed", "conflict"}
                    ):
                        raise InsightConflict(
                            "active Insight run targets are invalid; "
                            "clear current Insight data"
                        )
                    active_target_statuses[target_page_id] = target_status
            rows = list(
                connection.execute(
                    select(
                        analysis_page_results,
                        numbered_pages.c.chapter_ordinal,
                        numbered_pages.c.page_ordinal,
                        numbered_pages.c.current_page_number,
                        analysis_heads.c.updated_at.label(
                            "page_head_updated_at"
                        ),
                        source_pointer.c.asset_id.label(
                            "current_source_asset_id"
                        ),
                        assets.c.checksum.label("current_source_checksum"),
                    )
                    .join(
                        analysis_heads,
                        analysis_heads.c.active_result_id
                        == analysis_page_results.c.id,
                    )
                    .join(
                        numbered_pages,
                        numbered_pages.c.page_id == analysis_heads.c.page_id,
                    )
                    .join(
                        source_pointer,
                        (source_pointer.c.page_id == numbered_pages.c.page_id)
                        & (source_pointer.c.role == "source"),
                    )
                    .join(assets, assets.c.id == source_pointer.c.asset_id)
                    .where(analysis_heads.c.book_id == book_id)
                    .order_by(
                        numbered_pages.c.chapter_ordinal,
                        numbered_pages.c.page_ordinal,
                    )
                ).mappings()
            )
            if active_run_id is not None:
                if active_head_updated_at is None:
                    raise InsightConflict(
                        "active Insight book head timestamp is missing; "
                        "clear current Insight data"
                    )
                current_rows: list[Mapping[str, Any]] = []
                for row in rows:
                    row_page_id = _required_string(
                        row["page_id_snapshot"],
                        "current analysis pageId",
                    )
                    row_run_id = _required_string(
                        row["run_id"],
                        "current analysis run id",
                    )
                    is_fallback = (
                        row_run_id != active_run_id
                        and active_target_statuses.get(row_page_id)
                        in {"failed", "conflict"}
                        and _required_datetime(
                            row["page_head_updated_at"],
                            "current analysis page head updatedAt",
                        )
                        <= active_head_updated_at
                    )
                    if not is_fallback:
                        current_rows.append(row)
                rows = current_rows
            if not rows and book_head is None:
                raise InsightNotFound("book has no published page analysis")
            current_page_ids = [
                _required_string(value, "current Insight page id")
                for value in connection.execute(
                    select(numbered_pages.c.page_id).order_by(
                        numbered_pages.c.current_page_number
                    )
                ).scalars()
            ]
            if len(set(current_page_ids)) != len(current_page_ids):
                raise InsightConflict(
                    "current Insight pages are duplicated; "
                    "clear current Insight data"
                )
            analyzed_page_ids = {
                _required_string(
                    row["page_id_snapshot"],
                    "current analysis pageId",
                )
                for row in rows
            }
            missing_page_ids = set(current_page_ids) - analyzed_page_ids
            if any(
                active_target_statuses.get(page_id)
                not in {"failed", "conflict"}
                for page_id in missing_page_ids
            ):
                raise InsightConflict(
                    "current book contains pages without published analysis"
                )
            ordered_inputs = []
            for row in rows:
                page_id = _required_string(
                    row["page_id_snapshot"],
                    "current analysis pageId",
                )
                _required_integer(
                    row["page_number_snapshot"],
                    "analysis page number snapshot",
                    minimum=1,
                )
                current_page_number = _required_integer(
                    row["current_page_number"],
                    "current page number",
                    minimum=1,
                )
                source_asset_id = _required_string(
                    row["source_asset_id"],
                    "current analysis source asset id",
                )
                source_checksum = _required_sha256(
                    row["source_checksum"],
                    "current analysis source checksum",
                )
                if (
                    _required_string(
                        row["page_id"],
                        "current analysis page id",
                    )
                    != page_id
                    or _required_string(
                        row["current_source_asset_id"],
                        "current page source asset id",
                    )
                    != source_asset_id
                    or _required_sha256(
                        row["current_source_checksum"],
                        "current page source checksum",
                    )
                    != source_checksum
                ):
                    raise InsightConflict(
                        "published page analysis is stale; reanalyze the page"
                    )
                ordered_inputs.append(
                    {
                        "resultId": _required_string(
                            row["id"],
                            "current analysis result id",
                        ),
                        "pageId": page_id,
                        "pageNumber": current_page_number,
                        "currentSourceChecksum": source_checksum,
                    }
                )
        else:
            ordered_inputs = []
            for index, value in enumerate(frozen_inputs):
                if not isinstance(value, Mapping):
                    raise InsightConflict(
                        "frozen analysis input must be an object"
                    )
                item = dict(value)
                if set(item) != {
                    "resultId",
                    "pageId",
                    "pageNumber",
                    "currentSourceChecksum",
                }:
                    raise InsightConflict("frozen analysis input fields are invalid")
                ordered_inputs.append(
                    {
                        "resultId": _required_string(
                            item["resultId"],
                            f"frozen analysis input {index + 1} resultId",
                        ),
                        "pageId": _required_string(
                            item["pageId"],
                            f"frozen analysis input {index + 1} pageId",
                        ),
                        "pageNumber": _required_integer(
                            item["pageNumber"],
                            f"frozen analysis input {index + 1} pageNumber",
                            minimum=1,
                        ),
                        "currentSourceChecksum": _required_sha256(
                            item["currentSourceChecksum"],
                            f"frozen analysis input {index + 1} currentSourceChecksum",
                        ),
                    }
                )
            result_ids = [value["resultId"] for value in ordered_inputs]
            if not result_ids or len(set(result_ids)) != len(result_ids):
                raise InsightConflict("frozen analysis inputs are invalid")
            frozen_page_ids = [value["pageId"] for value in ordered_inputs]
            if len(set(frozen_page_ids)) != len(frozen_page_ids):
                raise InsightConflict("frozen analysis inputs are invalid")
            current_rows = list(
                connection.execute(
                    select(
                        numbered_pages.c.page_id,
                        numbered_pages.c.current_page_number,
                        assets.c.checksum.label("current_source_checksum"),
                    )
                    .join(
                        source_pointer,
                        (source_pointer.c.page_id == numbered_pages.c.page_id)
                        & (source_pointer.c.role == "source"),
                    )
                    .join(assets, assets.c.id == source_pointer.c.asset_id)
                    .where(numbered_pages.c.page_id.in_(frozen_page_ids))
                ).mappings()
            )
            current_by_page: dict[str, Mapping[str, Any]] = {}
            for row in current_rows:
                current_page_id = _required_string(
                    row["page_id"],
                    "current frozen analysis page id",
                )
                if current_page_id in current_by_page:
                    raise InsightConflict(
                        "current frozen analysis pages are duplicated; "
                        "clear current Insight data"
                    )
                current_by_page[current_page_id] = row
            if set(current_by_page) != set(frozen_page_ids):
                raise InsightConflict(
                    "frozen analysis pages no longer match current content"
                )
            for frozen_input in ordered_inputs:
                current = current_by_page[frozen_input["pageId"]]
                if (
                    _required_integer(
                        current["current_page_number"],
                        "current frozen analysis page number",
                        minimum=1,
                    )
                    != frozen_input["pageNumber"]
                    or _required_sha256(
                        current["current_source_checksum"],
                        "current frozen analysis source checksum",
                    )
                    != frozen_input["currentSourceChecksum"]
                ):
                    raise InsightConflict(
                        "frozen analysis inputs no longer match current content"
                    )
            selected = list(
                connection.execute(
                    select(analysis_page_results).where(
                        analysis_page_results.c.id.in_(tuple(result_ids))
                    )
                ).mappings()
            )
            by_id: dict[str, Mapping[str, Any]] = {}
            for row in selected:
                result_id = _required_string(
                    row["id"],
                    "frozen analysis result id",
                )
                if result_id in by_id:
                    raise InsightConflict(
                        "frozen analysis results are duplicated; "
                        "clear current Insight data"
                    )
                by_id[result_id] = row
            if set(by_id) != set(result_ids):
                raise InsightNotFound("frozen analysis input no longer exists")
            rows = [by_id[value] for value in result_ids]
            for row, frozen_input in zip(rows, ordered_inputs):
                if _required_string(
                    row["page_id_snapshot"],
                    "frozen analysis result pageId",
                ) != frozen_input["pageId"]:
                    raise InsightConflict(
                        "frozen analysis input does not match its result"
                    )
        if not rows:
            raise InsightNotFound("book has no published page analysis")
        if len(rows) != len(ordered_inputs):
            raise InsightConflict("analysis input rows are incomplete")
        page_payloads: list[dict[str, Any]] = []
        seen_result_ids: set[str] = set()
        seen_page_ids: set[str] = set()
        seen_page_numbers: set[int] = set()
        for row, frozen_input in zip(rows, ordered_inputs):
            result_id = _required_string(
                row["id"],
                "analysis page result id",
            )
            page_id = frozen_input["pageId"]
            page_number = frozen_input["pageNumber"]
            source_asset_id = _required_string(
                row["source_asset_id"],
                "analysis page source asset id",
            )
            source_checksum = _required_sha256(
                row["source_checksum"],
                "analysis page source checksum",
            )
            analysis_page_number = _required_integer(
                row["page_number_snapshot"],
                "analysis page snapshot page number",
                minimum=1,
            )
            if (
                result_id in seen_result_ids
                or page_id in seen_page_ids
                or page_number in seen_page_numbers
            ):
                raise InsightConflict(
                    "analysis inputs are duplicated; clear current Insight data"
                )
            seen_result_ids.add(result_id)
            seen_page_ids.add(page_id)
            seen_page_numbers.add(page_number)
            if (
                _required_string(
                    row["status"],
                    "analysis page result status",
                )
                != "published"
                or _required_integer(
                    row["schema_version"],
                    "analysis page result schema version",
                    minimum=1,
                )
                != 2
                or _required_string(
                    row["page_id"],
                    "analysis page current page id",
                )
                != page_id
                or _required_string(
                    row["page_id_snapshot"],
                    "analysis page snapshot page id",
                )
                != page_id
                or source_checksum != frozen_input["currentSourceChecksum"]
            ):
                raise InsightConflict(
                    "published page analysis identity is invalid; "
                    "clear current Insight data"
                )
            page_payloads.append(
                {
                    "resultId": result_id,
                    "pageId": page_id,
                    "pageNumber": page_number,
                    "sourceChecksum": source_checksum,
                    "currentSourceChecksum": frozen_input[
                        "currentSourceChecksum"
                    ],
                    "analysis": _persisted_page_analysis(
                        row["payload_json"],
                        page_id=page_id,
                        page_number=analysis_page_number,
                        source_asset_id=source_asset_id,
                        source_checksum=source_checksum,
                    ),
                }
            )
        pages_payload = tuple(page_payloads)
        fingerprint = _analysis_input_fingerprint(pages_payload)
        source_run_id: str | None = None
        source_run_status: str | None = None
        if book_head is not None:
            active_run_id = _required_string(
                book_head["active_run_id"],
                "active Insight run id",
            )
            active_run_status = _required_string(
                book_head["status"],
                "active Insight run status",
            )
            if active_run_status not in FINAL_ANALYSIS_RUN_STATUSES:
                raise InsightConflict(
                    "active Insight run status is invalid; clear current Insight data"
                )
            selected_run_ids = {
                _required_string(row["run_id"], "analysis page result run id")
                for row in rows
            }
            if selected_run_ids == {active_run_id}:
                source_run_id = active_run_id
                source_run_status = active_run_status
        return AnalysisInputSnapshot(
            book_id=book_id,
            source_run_id=source_run_id,
            source_run_status=source_run_status,
            result_ids=tuple(
                _required_string(row["id"], "analysis page result id")
                for row in rows
            ),
            pages=pages_payload,
            fingerprint=fingerprint,
        )

    def publish_artifact(
        self,
        *,
        connection: Connection,
        frozen: AnalysisInputSnapshot,
        kind: str,
        template: str,
        payload: Mapping[str, Any],
        activate: bool = True,
    ) -> dict[str, Any]:
        canonical_payload = validate_artifact_payload(
            kind=kind,
            template=template,
            payload=payload,
        )
        status = "building"
        should_activate = False
        if activate:
            current = self._snapshot(connection, book_id=frozen.book_id)
            status = _publication_status(frozen, current)
            should_activate = status in {"ready", "degraded"}
        now = utcnow()
        revision = _required_integer(
            connection.execute(
                select(
                    func.coalesce(
                        func.max(analysis_artifacts.c.revision),
                        0,
                    )
                    + 1
                ).where(
                    analysis_artifacts.c.book_id == frozen.book_id,
                    analysis_artifacts.c.kind == kind,
                    analysis_artifacts.c.template == template,
                )
            ).scalar_one(),
            "Insight artifact revision",
            minimum=1,
        )
        if should_activate:
            connection.execute(
                update(analysis_artifacts)
                .where(
                    analysis_artifacts.c.book_id == frozen.book_id,
                    analysis_artifacts.c.kind == kind,
                    analysis_artifacts.c.template == template,
                    analysis_artifacts.c.is_active.is_(True),
                )
                .values(is_active=False, updated_at=now)
            )
        artifact_id = str(uuid.uuid4())
        connection.execute(
            insert(analysis_artifacts).values(
                id=artifact_id,
                book_id=frozen.book_id,
                run_id=frozen.source_run_id,
                kind=kind,
                template=template,
                status=status,
                revision=revision,
                is_active=should_activate,
                dependency_fingerprint=frozen.fingerprint,
                payload_json=_json(canonical_payload),
                asset_id=None,
                created_at=now,
                updated_at=now,
            )
        )
        return {
            "artifactId": artifact_id,
            "kind": kind,
            "template": template,
            "status": status,
            "revision": revision,
            "dependencyFingerprint": frozen.fingerprint,
        }

    def layer_units(
        self,
        *,
        run_id: str,
        layer_index: int,
        config: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        analysis_config = _required_mapping(
            config.get("analysis"),
            "frozen Insight analysis config",
        )
        raw_layers = analysis_config.get("layers")
        if (
            not isinstance(raw_layers, list)
            or layer_index < 0
            or layer_index >= len(raw_layers)
            or not isinstance(raw_layers[layer_index], Mapping)
        ):
            raise InsightConflict("frozen Insight layer definition is invalid")
        layer = dict(raw_layers[layer_index])
        if (
            layer.get("index") != layer_index
            or not isinstance(layer.get("name"), str)
            or not layer["name"].strip()
        ):
            raise InsightConflict("frozen Insight layer identity is invalid")
        units_per_group = _required_integer(
            layer.get("unitsPerGroup"),
            "frozen Insight layer unitsPerGroup",
        )
        align_to_chapter = layer.get("alignToChapter")
        if not isinstance(align_to_chapter, bool):
            raise InsightConflict(
                "frozen Insight layer alignToChapter must be boolean"
            )
        preserve_chapter_boundaries = any(
            isinstance(value, Mapping)
            and value.get("alignToChapter") is True
            for value in raw_layers[layer_index:]
        )
        if layer_index == 0:
            frozen = self.snapshot_for_run(run_id=run_id)
            source_units = [
                {
                    "content": _required_mapping(
                        page.get("analysis"),
                        "frozen Insight page analysis",
                    ),
                    "pages": [
                        {
                            "pageId": _required_string(
                                page.get("pageId"),
                                "frozen Insight pageId",
                            ),
                            "pageNumber": _required_integer(
                                page.get("pageNumber"),
                                "frozen Insight pageNumber",
                                minimum=1,
                            ),
                            "chapterId": page.get("chapterId"),
                        }
                    ],
                }
                for page in frozen.pages
            ]
            group_size = _required_integer(
                analysis_config.get("pagesPerBatch"),
                "frozen Insight pagesPerBatch",
                minimum=1,
            )
        else:
            with self.engine.connect() as connection:
                rows = list(
                    connection.execute(
                        select(analysis_layer_results)
                        .where(
                            analysis_layer_results.c.run_id == run_id,
                            analysis_layer_results.c.layer_index
                            == layer_index - 1,
                            analysis_layer_results.c.status == "staging",
                        )
                        .order_by(analysis_layer_results.c.unit_index)
                    ).mappings()
                )
                covered_by_result: dict[str, list[Mapping[str, Any]]] = {}
                if rows:
                    for page in connection.execute(
                        select(analysis_layer_result_pages)
                        .where(
                            analysis_layer_result_pages.c.layer_result_id.in_(
                                tuple(
                                    _required_string(
                                        row["id"],
                                        "analysis layer result id",
                                    )
                                    for row in rows
                                )
                            )
                        )
                        .order_by(
                            analysis_layer_result_pages.c.layer_result_id,
                            analysis_layer_result_pages.c.ordinal,
                        )
                    ).mappings():
                        covered_by_result.setdefault(
                            _required_string(
                                page["layer_result_id"],
                                "analysis layer result id",
                            ),
                            [],
                        ).append(page)
                source_units = []
                for row in rows:
                    row_id = _required_string(
                        row["id"],
                        "analysis layer result id",
                    )
                    covered = covered_by_result.get(row_id, ())
                    source_units.append(
                        {
                            "content": _json_object(
                                row["content_json"],
                                "analysis layer content",
                            ),
                            "pages": [
                                {
                                    "pageId": _required_string(
                                        page["page_id_snapshot"],
                                        "analysis layer pageId",
                                    ),
                                    "pageNumber": _required_integer(
                                        page["page_number_snapshot"],
                                        "analysis layer pageNumber",
                                        minimum=1,
                                    ),
                                    "chapterId": _optional_string(
                                        row["chapter_id"],
                                        "analysis layer chapterId",
                                    ),
                                }
                                for page in covered
                            ],
                        }
                    )
            if not source_units:
                raise InsightConflict(
                    f"Insight layer {layer_index - 1} has no staging units"
                )
            group_size = units_per_group or len(source_units)

        grouped: list[list[dict[str, Any]]] = []
        if preserve_chapter_boundaries:
            by_chapter: dict[str, list[dict[str, Any]]] = {}
            chapter_order: list[str] = []
            for source_index, unit in enumerate(source_units):
                chapters_in_unit = {
                    page.get("chapterId")
                    for page in unit["pages"]
                    if page.get("chapterId") is not None
                }
                if (
                    len(chapters_in_unit) != 1
                    or not all(
                        isinstance(chapter_id, str) and chapter_id
                        for chapter_id in chapters_in_unit
                    )
                ):
                    raise InsightConflict(
                        f"Insight source unit {source_index + 1} crosses chapter boundaries"
                    )
                chapter_key = next(iter(chapters_in_unit))
                if chapter_key not in by_chapter:
                    chapter_order.append(chapter_key)
                    by_chapter[chapter_key] = []
                by_chapter[chapter_key].append(unit)
            for chapter_key in chapter_order:
                chapter_units = by_chapter[chapter_key]
                size = group_size or len(chapter_units)
                grouped.extend(
                    chapter_units[offset : offset + size]
                    for offset in range(0, len(chapter_units), size)
                )
        else:
            size = group_size or len(source_units)
            grouped = [
                source_units[offset : offset + size]
                for offset in range(0, len(source_units), size)
            ]
        result: list[dict[str, Any]] = []
        for unit_index, group in enumerate(grouped):
            covered_pages: list[dict[str, Any]] = []
            seen: set[str] = set()
            for source in group:
                for page in source["pages"]:
                    page_id = _required_string(
                        page.get("pageId"),
                        "Insight source pageId",
                    )
                    if page_id not in seen:
                        seen.add(page_id)
                        covered_pages.append(dict(page))
            chapters_in_group = {
                page.get("chapterId")
                for page in covered_pages
                if page.get("chapterId") is not None
            }
            chapter_id = None
            if len(chapters_in_group) == 1:
                chapter_id = _required_string(
                    next(iter(chapters_in_group)),
                    "Insight source chapterId",
                )
            result.append(
                {
                    "unitIndex": unit_index,
                    "chapterId": chapter_id,
                    "pages": covered_pages,
                    "inputs": [dict(source["content"]) for source in group],
                    "layer": {
                        **layer,
                        "promptType": _layer_prompt_type(
                            layer_index=layer_index,
                            layer_count=len(raw_layers),
                            align_to_chapter=align_to_chapter,
                        ),
                    },
                }
            )
        return result

    def summary_inputs(
        self,
        frozen: AnalysisInputSnapshot,
    ) -> tuple[dict[str, Any], ...]:
        """Use the highest complete summary layer, with compact pages as fallback."""

        fallback_items: list[dict[str, Any]] = []
        for index, page in enumerate(frozen.pages, start=1):
            result_id = _required_string(
                page.get("resultId"),
                f"Insight summary input {index} resultId",
            )
            page_id = _required_string(
                page.get("pageId"),
                f"Insight summary input {index} pageId",
            )
            page_number = _required_integer(
                page.get("pageNumber"),
                f"Insight summary input {index} pageNumber",
                minimum=1,
            )
            analysis = _required_mapping(
                page.get("analysis"),
                f"Insight summary input {index} analysis",
            )
            fallback_items.append(
                {
                    "resultId": result_id,
                    "pageId": page_id,
                    "pageIds": [page_id],
                    "pageNumber": page_number,
                    "pageNumbers": [page_number],
                    "analysis": {
                        key: value
                        for key, value in analysis.items()
                        if key
                        in {
                            "page_summary",
                            "key_events",
                            "continuity_notes",
                            "warnings",
                        }
                    },
                }
            )
        fallback = tuple(fallback_items)
        if not frozen.source_run_id:
            return fallback
        with self.engine.connect() as connection:
            rows = list(
                connection.execute(
                    select(analysis_layer_results)
                    .where(
                        analysis_layer_results.c.run_id
                        == frozen.source_run_id,
                        analysis_layer_results.c.status.in_(
                            ("staging", "published")
                        ),
                    )
                    .order_by(
                        analysis_layer_results.c.layer_index.desc(),
                        analysis_layer_results.c.unit_index,
                    )
                ).mappings()
            )
            covered_rows = list(
                connection.execute(
                    select(analysis_layer_result_pages).where(
                        analysis_layer_result_pages.c.layer_result_id.in_(
                            tuple(
                                _required_string(
                                    row["id"],
                                    "analysis layer result id",
                                )
                                for row in rows
                            )
                        )
                    )
                ).mappings()
            ) if rows else []
        covered_by_result: dict[str, list[Mapping[str, Any]]] = {}
        for page in covered_rows:
            covered_by_result.setdefault(
                _required_string(
                    page["layer_result_id"],
                    "analysis layer result id",
                ),
                [],
            ).append(page)
        by_layer: dict[int, list[Mapping[str, Any]]] = {}
        for row in rows:
            by_layer.setdefault(
                _required_integer(
                    row["layer_index"],
                    "analysis layer index",
                ),
                [],
            ).append(row)
        expected = {
            _required_string(page.get("pageId"), "Insight summary pageId")
            for page in frozen.pages
        }
        for layer_index in sorted(by_layer, reverse=True):
            layer_rows = by_layer[layer_index]
            covered = {
                _required_string(
                    page["page_id_snapshot"],
                    "analysis layer pageId",
                )
                for row in layer_rows
                for page in covered_by_result.get(
                    _required_string(row["id"], "analysis layer result id"),
                    (),
                )
            }
            if covered != expected:
                continue
            inputs: list[dict[str, Any]] = []
            for row in layer_rows:
                pages_for_result = sorted(
                    covered_by_result.get(
                        _required_string(
                            row["id"],
                            "analysis layer result id",
                        ),
                        (),
                    ),
                    key=lambda page: _required_integer(
                        page["ordinal"],
                        "analysis layer page ordinal",
                        minimum=1,
                    ),
                )
                page_ids = [
                    _required_string(
                        page["page_id_snapshot"],
                        "analysis layer pageId",
                    )
                    for page in pages_for_result
                ]
                page_numbers = [
                    _required_integer(
                        page["page_number_snapshot"],
                        "analysis layer pageNumber",
                        minimum=1,
                    )
                    for page in pages_for_result
                ]
                if not page_ids:
                    raise InsightConflict(
                        "analysis layer has no covered pages; clear current Insight data"
                    )
                inputs.append(
                    {
                        "resultId": _required_string(
                            row["id"],
                            "analysis layer result id",
                        ),
                        "pageId": page_ids[0],
                        "pageIds": page_ids,
                        "pageNumber": page_numbers[0],
                        "pageNumbers": page_numbers,
                        "analysis": _json_object(
                            row["content_json"],
                            "analysis layer content",
                        ),
                    }
                )
            return tuple(inputs)
        return fallback

    def compressed_context_input(
        self,
        frozen: AnalysisInputSnapshot,
    ) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            statement = select(analysis_artifacts).where(
                analysis_artifacts.c.book_id == frozen.book_id,
                analysis_artifacts.c.kind == "compressed_context",
                analysis_artifacts.c.template == "default",
            )
            if frozen.source_run_id:
                staged = connection.execute(
                    statement.where(
                        analysis_artifacts.c.run_id == frozen.source_run_id,
                        analysis_artifacts.c.status == "building",
                        analysis_artifacts.c.dependency_fingerprint
                        == frozen.fingerprint,
                    ).order_by(analysis_artifacts.c.revision.desc())
                ).mappings().first()
            else:
                staged = None
            row = staged or connection.execute(
                statement.where(
                    analysis_artifacts.c.is_active.is_(True),
                    analysis_artifacts.c.status.in_(("ready", "degraded")),
                    analysis_artifacts.c.dependency_fingerprint
                    == frozen.fingerprint,
                )
                .order_by(analysis_artifacts.c.revision.desc())
            ).mappings().first()
        if row is None:
            return None
        return {
            "resultId": _required_string(
                row["id"],
                "compressed context artifact id",
            ),
            "pageId": _required_string(
                frozen.pages[0].get("pageId"),
                "compressed context first pageId",
            ),
            "pageIds": [
                _required_string(
                    page.get("pageId"),
                    "compressed context pageId",
                )
                for page in frozen.pages
            ],
            "pageNumber": _required_integer(
                frozen.pages[0].get("pageNumber"),
                "compressed context first pageNumber",
                minimum=1,
            ),
            "pageNumbers": [
                _required_integer(
                    page.get("pageNumber"),
                    "compressed context pageNumber",
                    minimum=1,
                )
                for page in frozen.pages
            ],
            "analysis": {
                "compressed_context": _json_object(
                    row["payload_json"],
                    "compressed context payload",
                ),
            },
        }

    @staticmethod
    def publish_layer(
        connection: Connection,
        *,
        run_id: str,
        layer_index: int,
        layer_name: str,
        units: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not units:
            raise InsightConflict("Insight layer has no publishable units")
        now = utcnow()
        for unit_index, unit in enumerate(units, start=1):
            raw_pages = unit.get("pages")
            if not isinstance(raw_pages, list) or not raw_pages or any(
                not isinstance(page, Mapping) for page in raw_pages
            ):
                raise InsightConflict("Insight layer covered pages are invalid")
            pages_covered = list(raw_pages)
            content = _required_mapping(
                unit.get("content"),
                "Insight layer content",
            )
            if not contains_nonempty_text(content):
                raise InsightConflict("Insight layer content must not be empty")
            validated_pages = []
            for page_index, page in enumerate(pages_covered, start=1):
                validated_pages.append(
                    {
                        **dict(page),
                        "pageId": _required_string(
                            page.get("pageId"),
                            f"Insight layer unit {unit_index} page {page_index} pageId",
                        ),
                        "pageNumber": _required_integer(
                            page.get("pageNumber"),
                            f"Insight layer unit {unit_index} page {page_index} pageNumber",
                            minimum=1,
                        ),
                    }
                )
            pages_covered = validated_pages
            fingerprint = hashlib.sha256(
                _json(
                    {
                        "pages": pages_covered,
                        "content": content,
                    }
                ).encode("utf-8")
            ).hexdigest()
            result_id = str(uuid.uuid4())
            page_numbers = [page["pageNumber"] for page in pages_covered]
            connection.execute(
                insert(analysis_layer_results).values(
                    id=result_id,
                    run_id=run_id,
                    layer_index=layer_index,
                    layer_name=layer_name,
                    unit_index=_required_integer(
                        unit.get("unitIndex"),
                        "Insight layer unitIndex",
                    ),
                    chapter_id=_optional_string(
                        unit.get("chapterId"),
                        "Insight layer chapterId",
                    ),
                    page_range_snapshot_json=_json(
                        {
                            "start": min(page_numbers),
                            "end": max(page_numbers),
                        }
                    ),
                    content_json=_json(content),
                    input_fingerprint=fingerprint,
                    status="staging",
                    created_at=now,
                    updated_at=now,
                )
            )
            connection.execute(
                insert(analysis_layer_result_pages),
                [
                    {
                        "layer_result_id": result_id,
                        "ordinal": ordinal,
                        "page_id": page["pageId"],
                        "page_id_snapshot": page["pageId"],
                        "page_number_snapshot": page["pageNumber"],
                    }
                    for ordinal, page in enumerate(
                        pages_covered,
                        start=1,
                    )
                ],
            )
        return {
            "runId": run_id,
            "layerIndex": layer_index,
            "unitCount": len(units),
        }

    def publish_timeline(
        self,
        *,
        connection: Connection,
        frozen: AnalysisInputSnapshot,
        result: Mapping[str, Any],
        activate: bool = True,
    ) -> dict[str, Any]:
        status = "building"
        if activate:
            current = self._snapshot(connection, book_id=frozen.book_id)
            status = _publication_status(frozen, current)
        mode = result.get("mode")
        if not isinstance(mode, str):
            raise InsightConflict("timeline mode is missing")
        if mode not in {"enhanced", "compressed", "simple"}:
            raise InsightConflict("timeline mode is invalid")
        try:
            content, raw_events, raw_characters = _validate_timeline_parts(
                content=result.get("content"),
                events=result.get("events"),
                characters=result.get("characters"),
                require_events=True,
            )
        except ValueError as exc:
            raise InsightConflict(str(exc)) from exc
        _validate_timeline_metadata(content, mode=mode)
        if any("eventId" in event for event in raw_events):
            raise InsightConflict("timeline event must not define eventId")
        if any("characterId" in character for character in raw_characters):
            raise InsightConflict("timeline character must not define characterId")
        raw_events, raw_characters = _canonical_timeline_references(
            frozen=frozen,
            events=raw_events,
            characters=raw_characters,
        )
        timeline_id = str(uuid.uuid4())
        now = utcnow()
        should_activate = activate and status in {"ready", "degraded"}
        if should_activate:
            connection.execute(
                update(timeline_versions)
                .where(
                    timeline_versions.c.book_id == frozen.book_id,
                    timeline_versions.c.is_active.is_(True),
                )
                .values(is_active=False, updated_at=now)
            )
        connection.execute(
            insert(timeline_versions).values(
                id=timeline_id,
                book_id=frozen.book_id,
                run_id=frozen.source_run_id,
                mode=mode,
                status=status,
                content_json=_json(content),
                dependency_fingerprint=frozen.fingerprint,
                is_active=should_activate,
                created_at=now,
                updated_at=now,
            )
        )
        if raw_events:
            connection.execute(
                insert(timeline_events),
                [
                    {
                        "id": str(uuid.uuid4()),
                        "timeline_version_id": timeline_id,
                        "ordinal": index,
                        "payload_json": _json(event),
                    }
                    for index, event in enumerate(raw_events, start=1)
                ],
            )
        character_rows = []
        for character in raw_characters:
            name = character["name"]
            character_rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "timeline_version_id": timeline_id,
                    "name": name,
                    "payload_json": _json(character),
                }
            )
        if character_rows:
            connection.execute(insert(timeline_characters), character_rows)
        return {
            "timelineVersionId": timeline_id,
            "mode": mode,
            "status": status,
            "eventCount": len(raw_events),
            "characterCount": len(character_rows),
        }

    def next_vector_generation(self, book_id: str) -> int:
        with self.engine.connect() as connection:
            return _required_integer(
                connection.execute(
                    select(
                        func.coalesce(
                            func.max(vector_generations.c.generation),
                            0,
                        )
                        + 1
                    ).where(vector_generations.c.book_id == book_id)
                ).scalar_one(),
                "next vector generation",
                minimum=1,
            )

    def checkpoint_vector_generation(
        self,
        *,
        connection: Connection,
        frozen: AnalysisInputSnapshot,
        generation: int,
        page_count: int,
        event_count: int,
    ) -> dict[str, Any]:
        generation = _required_integer(
            generation,
            "vector generation",
            minimum=1,
        )
        page_count = _required_integer(page_count, "vector page count")
        event_count = _required_integer(event_count, "vector event count")
        row = connection.execute(
            select(vector_generations).where(
                vector_generations.c.book_id == frozen.book_id,
                vector_generations.c.generation == generation,
            )
        ).mappings().one_or_none()
        now = utcnow()
        if row is None:
            generation_id = str(uuid.uuid4())
            connection.execute(
                insert(vector_generations).values(
                    id=generation_id,
                    book_id=frozen.book_id,
                    run_id=frozen.source_run_id,
                    generation=generation,
                    status="building",
                    dependency_fingerprint=frozen.fingerprint,
                    page_count=page_count,
                    event_count=event_count,
                    is_active=False,
                    created_at=now,
                    updated_at=now,
                )
            )
        else:
            if (
                row["run_id"] != frozen.source_run_id
                or row["dependency_fingerprint"] != frozen.fingerprint
                or _required_boolean(
                    row["is_active"],
                    "vector generation active flag",
                )
                or row["status"] != "building"
                or page_count
                < _required_integer(row["page_count"], "stored vector page count")
                or event_count
                < _required_integer(row["event_count"], "stored vector event count")
            ):
                raise InsightConflict("vector generation checkpoint conflicts")
            generation_id = _required_string(row["id"], "vector generation id")
            connection.execute(
                update(vector_generations)
                .where(vector_generations.c.id == generation_id)
                .values(
                    page_count=page_count,
                    event_count=event_count,
                    updated_at=now,
                )
            )
        return {
            "vectorGenerationId": generation_id,
            "generation": generation,
            "status": "building",
            "pageCount": page_count,
            "eventCount": event_count,
        }

    def fail_vector_generation(self, *, book_id: str, generation: int) -> None:
        book_id = _required_string(book_id, "vector bookId")
        generation = _required_integer(
            generation,
            "vector generation",
            minimum=1,
        )
        with self.engine.begin() as connection:
            connection.execute(
                update(vector_generations)
                .where(
                    vector_generations.c.book_id == book_id,
                    vector_generations.c.generation == generation,
                    vector_generations.c.status == "building",
                    vector_generations.c.is_active.is_(False),
                )
                .values(status="failed", updated_at=utcnow())
            )

    def publish_vector_generation(
        self,
        *,
        connection: Connection,
        frozen: AnalysisInputSnapshot,
        generation: int,
        page_count: int,
        event_count: int,
        activate: bool = True,
    ) -> dict[str, Any]:
        generation = _required_integer(
            generation,
            "vector generation",
            minimum=1,
        )
        page_count = _required_integer(page_count, "vector page count")
        event_count = _required_integer(event_count, "vector event count")
        status = "building"
        if activate:
            current = self._snapshot(connection, book_id=frozen.book_id)
            status = _publication_status(frozen, current)
        now = utcnow()
        should_activate = activate and status in {"ready", "degraded"}
        if should_activate:
            connection.execute(
                update(vector_generations)
                .where(
                    vector_generations.c.book_id == frozen.book_id,
                    vector_generations.c.is_active.is_(True),
                )
                .values(is_active=False, updated_at=now)
            )
        existing = connection.execute(
            select(vector_generations).where(
                vector_generations.c.book_id == frozen.book_id,
                vector_generations.c.generation == generation,
            )
        ).mappings().one_or_none()
        if existing is None:
            generation_id = str(uuid.uuid4())
            connection.execute(
                insert(vector_generations).values(
                    id=generation_id,
                    book_id=frozen.book_id,
                    run_id=frozen.source_run_id,
                    generation=generation,
                    status=status,
                    dependency_fingerprint=frozen.fingerprint,
                    page_count=page_count,
                    event_count=event_count,
                    is_active=should_activate,
                    created_at=now,
                    updated_at=now,
                )
            )
        else:
            if (
                existing["run_id"] != frozen.source_run_id
                or existing["dependency_fingerprint"] != frozen.fingerprint
                or _required_boolean(
                    existing["is_active"],
                    "vector generation active flag",
                )
                or existing["status"] != "building"
            ):
                raise InsightConflict("vector generation publication conflicts")
            generation_id = _required_string(
                existing["id"],
                "vector generation id",
            )
            connection.execute(
                update(vector_generations)
                .where(vector_generations.c.id == generation_id)
                .values(
                    status=status,
                    page_count=page_count,
                    event_count=event_count,
                    is_active=should_activate,
                    updated_at=now,
                )
            )
        return {
            "vectorGenerationId": generation_id,
            "generation": generation,
            "status": status,
            "pageCount": page_count,
            "eventCount": event_count,
        }

    def get_artifact(
        self,
        *,
        book_id: str,
        kind: str,
        template: str,
    ) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            self._assert_book_owned(connection, book_id)
            row = connection.execute(
                select(analysis_artifacts).where(
                    analysis_artifacts.c.book_id == book_id,
                    analysis_artifacts.c.kind == kind,
                    analysis_artifacts.c.template == template,
                    analysis_artifacts.c.is_active.is_(True),
                )
            ).mappings().one_or_none()
        if row is None:
            return None
        artifact_kind = _required_string(row["kind"], "Insight artifact kind")
        artifact_template = _required_string(
            row["template"],
            "Insight artifact template",
        )
        if artifact_kind not in {"overview", "compressed_context"}:
            raise InsightConflict(
                "active Insight artifact kind is invalid; clear current Insight data"
            )
        if (
            artifact_kind == "overview"
            and artifact_template not in OVERVIEW_TEMPLATES
        ) or (
            artifact_kind == "compressed_context"
            and artifact_template != "default"
        ):
            raise InsightConflict(
                "active Insight artifact template is invalid; clear current Insight data"
            )
        status = _required_string(row["status"], "Insight artifact status")
        if status not in {"ready", "degraded", "stale"}:
            raise InsightConflict(
                "active Insight artifact status is invalid; clear current Insight data"
            )
        try:
            payload = validate_artifact_payload(
                kind=artifact_kind,
                template=artifact_template,
                payload=_json_object(
                    row["payload_json"],
                    "analysis artifact payload",
                ),
            )
        except InsightConflict as exc:
            raise InsightConflict(
                "active Insight artifact payload is invalid; "
                "clear current Insight data"
            ) from exc
        return {
            "artifactId": _required_string(row["id"], "Insight artifact id"),
            "bookId": _required_string(row["book_id"], "Insight artifact bookId"),
            "runId": _optional_string(row["run_id"], "Insight artifact runId"),
            "kind": artifact_kind,
            "template": artifact_template,
            "status": status,
            "revision": _required_integer(
                row["revision"],
                "Insight artifact revision",
                minimum=1,
            ),
            "dependencyFingerprint": _required_sha256(
                row["dependency_fingerprint"],
                "Insight artifact dependency fingerprint",
            ),
            "payload": payload,
        }

    def get_timeline(
        self,
        *,
        book_id: str,
        event_after: int = 0,
        event_limit: int = 100,
        character_after: str | None = None,
        character_limit: int = 100,
    ) -> dict[str, Any] | None:
        if isinstance(event_after, bool) or not isinstance(event_after, int) or event_after < 0:
            raise ValueError("event cursor must be nonnegative")
        if isinstance(event_limit, bool) or not isinstance(event_limit, int) or not 1 <= event_limit <= 200:
            raise ValueError("event limit must be between 1 and 200")
        if (
            isinstance(character_limit, bool)
            or not isinstance(character_limit, int)
            or not 1 <= character_limit <= 200
        ):
            raise ValueError("character limit must be between 1 and 200")
        if character_after is not None:
            _required_string(character_after, "character cursor")
        with self.engine.connect() as connection:
            self._assert_book_owned(connection, book_id)
            row = connection.execute(
                select(timeline_versions).where(
                    timeline_versions.c.book_id == book_id,
                    timeline_versions.c.is_active.is_(True),
                )
            ).mappings().one_or_none()
            if row is None:
                return None
            timeline_id = _required_string(row["id"], "stored timeline id")
            mode = _required_string(row["mode"], "stored timeline mode")
            if mode not in {"enhanced", "compressed", "simple"}:
                raise InsightConflict(
                    "stored timeline mode is invalid; clear current Insight data"
                )
            status = _required_string(row["status"], "stored timeline status")
            if status not in {"ready", "degraded", "stale"}:
                raise InsightConflict(
                    "stored timeline status is invalid; clear current Insight data"
                )
            event_rows = list(
                connection.execute(
                    select(
                        timeline_events.c.id,
                        timeline_events.c.ordinal,
                        timeline_events.c.payload_json,
                    )
                    .where(
                        timeline_events.c.timeline_version_id == timeline_id,
                        timeline_events.c.ordinal > event_after,
                    )
                    .order_by(timeline_events.c.ordinal)
                    .limit(event_limit + 1)
                )
            )
            character_statement = select(
                timeline_characters.c.id,
                timeline_characters.c.name,
                timeline_characters.c.payload_json,
            ).where(
                timeline_characters.c.timeline_version_id == timeline_id
            )
            if character_after:
                character_statement = character_statement.where(
                    timeline_characters.c.name > character_after
                )
            character_rows = list(
                connection.execute(
                    character_statement.order_by(
                        timeline_characters.c.name
                    ).limit(character_limit + 1)
                )
            )
            event_count = _required_integer(
                connection.execute(
                    select(func.count(timeline_events.c.id)).where(
                        timeline_events.c.timeline_version_id == timeline_id
                    )
                ).scalar_one(),
                "stored timeline event count",
            )
            character_count = _required_integer(
                connection.execute(
                    select(func.count(timeline_characters.c.id)).where(
                        timeline_characters.c.timeline_version_id == timeline_id
                    )
                ).scalar_one(),
                "stored timeline character count",
            )
            page_count = _required_integer(
                connection.execute(
                    select(func.count(pages.c.id))
                    .join(chapters, chapters.c.id == pages.c.chapter_id)
                    .where(chapters.c.book_id == book_id)
                ).scalar_one(),
                "stored timeline page count",
            )
            has_more_events = len(event_rows) > event_limit
            selected_events = event_rows[:event_limit]
            has_more_characters = len(character_rows) > character_limit
            selected_characters = character_rows[:character_limit]
            content_payload = _json_object(
                row["content_json"],
                "timeline content",
            )
            event_payloads = []
            for event_id, _ordinal, value in selected_events:
                event_payload = _json_object(value, "timeline event payload")
                if "eventId" in event_payload:
                    raise InsightConflict(
                        "stored timeline event contains a reserved id; "
                        "clear current Insight data"
                    )
                event_payloads.append(
                    {
                        **event_payload,
                        "eventId": _required_string(
                            event_id,
                            "stored timeline event id",
                        ),
                    }
                )
            character_payloads = []
            for character_id, _name, value in selected_characters:
                character_payload = _json_object(
                    value,
                    "timeline character payload",
                )
                if "characterId" in character_payload:
                    raise InsightConflict(
                        "stored timeline character contains a reserved id; "
                        "clear current Insight data"
                    )
                character_payloads.append(
                    {
                        **character_payload,
                        "characterId": _required_string(
                            character_id,
                            "stored timeline character id",
                        ),
                    }
                )
            try:
                _validate_timeline_parts(
                    content=content_payload,
                    events=event_payloads,
                    characters=character_payloads,
                    require_events=False,
                )
                _validate_timeline_metadata(
                    content_payload,
                    mode=mode,
                )
            except ValueError as exc:
                raise InsightConflict(
                    "stored timeline payload is invalid; clear current Insight data"
                ) from exc
            referenced_page_ids: set[str] = set()
            for payload in event_payloads:
                page_ids = payload.get("page_ids")
                page_numbers = payload.get("page_numbers")
                if not isinstance(page_ids, list) or not page_ids or any(
                    not isinstance(page_id, str) or not page_id
                    for page_id in page_ids
                ):
                    raise InsightConflict(
                        "stored timeline event page_ids are invalid; clear current Insight data"
                    )
                if (
                    not isinstance(page_numbers, list)
                    or not page_numbers
                    or len(page_numbers) != len(page_ids)
                ):
                    raise InsightConflict(
                        "stored timeline event page references are incomplete; "
                        "clear current Insight data"
                    )
                referenced_page_ids.update(page_ids)
            referenced_page_numbers = _timeline_thumbnail_page_numbers(
                content=content_payload,
                events=event_payloads,
                characters=character_payloads,
            )
            page_thumbnails: dict[str, str] = {}
            if referenced_page_ids or referenced_page_numbers:
                thumbnail_pointer = page_assets.alias(
                    "timeline_thumbnail_pointer"
                )
                numbered_pages = (
                    select(
                        pages.c.id.label("page_id"),
                        func.row_number()
                        .over(order_by=(chapters.c.ordinal, pages.c.ordinal))
                        .label("page_number"),
                        thumbnail_pointer.c.asset_id.label(
                            "thumbnail_asset_id"
                        ),
                    )
                    .join(chapters, chapters.c.id == pages.c.chapter_id)
                    .outerjoin(
                        thumbnail_pointer,
                        (
                            thumbnail_pointer.c.page_id == pages.c.id
                        )
                        & (
                            thumbnail_pointer.c.role == "thumbnail_source"
                        ),
                    )
                    .where(chapters.c.book_id == book_id)
                    .subquery()
                )
                page_filter = (
                    numbered_pages.c.page_id.in_(referenced_page_ids)
                    if referenced_page_ids
                    else numbered_pages.c.page_number.in_(
                        referenced_page_numbers
                    )
                )
                if referenced_page_ids and referenced_page_numbers:
                    page_filter = page_filter | numbered_pages.c.page_number.in_(
                        referenced_page_numbers
                    )
                page_rows = list(
                    connection.execute(
                        select(
                            numbered_pages.c.page_id,
                            numbered_pages.c.page_number,
                            numbered_pages.c.thumbnail_asset_id,
                        ).where(page_filter)
                    ).mappings()
                )
                page_numbers_by_id: dict[str, int] = {}
                page_thumbnails = {}
                for page in page_rows:
                    page_id = _required_string(
                        page["page_id"],
                        "stored timeline page id",
                    )
                    page_number = _required_integer(
                        page["page_number"],
                        "stored timeline page number",
                        minimum=1,
                    )
                    if page_id in page_numbers_by_id:
                        raise InsightConflict(
                            "stored timeline page reference is duplicated; "
                            "clear current Insight data"
                        )
                    page_numbers_by_id[page_id] = page_number
                    thumbnail_asset_id = page["thumbnail_asset_id"]
                    if thumbnail_asset_id is not None:
                        page_thumbnails[f"{page_number}"] = (
                            "/api/v2/assets/"
                            + _required_string(
                                thumbnail_asset_id,
                                "stored timeline thumbnail asset id",
                            )
                        )
                for payload in event_payloads:
                    if [
                        page_numbers_by_id.get(page_id)
                        for page_id in payload["page_ids"]
                    ] != payload["page_numbers"]:
                        raise InsightConflict(
                            "stored timeline page references are stale or invalid; "
                            "clear current Insight data"
                        )
        return {
            "timelineVersionId": timeline_id,
            "bookId": _required_string(row["book_id"], "stored timeline bookId"),
            "runId": _optional_string(row["run_id"], "stored timeline runId"),
            "mode": mode,
            "status": status,
            "content": content_payload,
            "events": event_payloads,
            "characters": character_payloads,
            "eventPage": {
                "totalCount": event_count,
                "nextCursor": (
                    _required_integer(
                        selected_events[-1][1],
                        "stored timeline event cursor",
                        minimum=0,
                    )
                    if has_more_events and selected_events
                    else None
                )
            },
            "characterPage": {
                "totalCount": character_count,
                "nextCursor": (
                    _required_string(
                        selected_characters[-1][1],
                        "stored timeline character cursor",
                    )
                    if has_more_characters and selected_characters
                    else None
                )
            },
            "pageCount": page_count,
            "pageThumbnails": page_thumbnails,
            "dependencyFingerprint": _required_sha256(
                row["dependency_fingerprint"],
                "stored timeline dependency fingerprint",
            ),
        }

    def get_timeline_status(
        self,
        *,
        book_id: str,
    ) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            self._assert_book_owned(connection, book_id)
            row = connection.execute(
                select(
                    timeline_versions.c.id,
                    timeline_versions.c.mode,
                    timeline_versions.c.status,
                ).where(
                    timeline_versions.c.book_id == book_id,
                    timeline_versions.c.is_active.is_(True),
                )
            ).mappings().one_or_none()
        if row is None:
            return None
        mode = _required_string(row["mode"], "stored timeline mode")
        if mode not in {"enhanced", "compressed", "simple"}:
            raise InsightConflict(
                "stored timeline mode is invalid; clear current Insight data"
            )
        status = _required_string(row["status"], "stored timeline status")
        if status not in {"ready", "degraded", "stale"}:
            raise InsightConflict(
                "stored timeline status is invalid; clear current Insight data"
            )
        return {
            "timelineVersionId": _required_string(
                row["id"], "stored timeline id"
            ),
            "mode": mode,
            "status": status,
        }

    def list_timeline_characters(
        self,
        *,
        book_id: str,
    ) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            self._assert_book_owned(connection, book_id)
            timeline = connection.execute(
                select(
                    timeline_versions.c.id,
                    timeline_versions.c.mode,
                    timeline_versions.c.status,
                ).where(
                    timeline_versions.c.book_id == book_id,
                    timeline_versions.c.is_active.is_(True),
                )
            ).mappings().one_or_none()
            if timeline is None:
                return None
            rows = connection.execute(
                select(
                    timeline_characters.c.id,
                    timeline_characters.c.payload_json,
                )
                .where(
                    timeline_characters.c.timeline_version_id
                    == timeline["id"]
                )
                .order_by(timeline_characters.c.name)
            ).mappings()
            characters = []
            for row in rows:
                payload = _json_object(
                    row["payload_json"],
                    "timeline character payload",
                )
                if "characterId" in payload:
                    raise InsightConflict(
                        "stored timeline character contains a reserved id; "
                        "clear current Insight data"
                    )
                characters.append(
                    {
                        **payload,
                        "characterId": _required_string(
                            row["id"],
                            "stored timeline character id",
                        ),
                    }
                )
        mode = _required_string(timeline["mode"], "stored timeline mode")
        if mode not in {"enhanced", "compressed", "simple"}:
            raise InsightConflict(
                "stored timeline mode is invalid; clear current Insight data"
            )
        status = _required_string(timeline["status"], "stored timeline status")
        if status not in {"ready", "degraded", "stale"}:
            raise InsightConflict(
                "stored timeline status is invalid; clear current Insight data"
            )
        try:
            _validate_timeline_parts(
                content={},
                events=[],
                characters=characters,
                require_events=False,
            )
        except ValueError as exc:
            raise InsightConflict(
                "stored timeline character payload is invalid; "
                "clear current Insight data"
            ) from exc
        return {
            "timelineVersionId": _required_string(
                timeline["id"], "stored timeline id"
            ),
            "mode": mode,
            "status": status,
            "characters": characters,
        }

    def get_timeline_character(
        self,
        *,
        book_id: str,
        character_id: str,
    ) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            self._assert_book_owned(connection, book_id)
            row = connection.execute(
                select(
                    timeline_versions.c.id.label("timeline_version_id"),
                    timeline_versions.c.mode,
                    timeline_versions.c.status,
                    timeline_characters.c.id.label("character_id"),
                    timeline_characters.c.payload_json,
                )
                .join(
                    timeline_characters,
                    timeline_characters.c.timeline_version_id
                    == timeline_versions.c.id,
                )
                .where(
                    timeline_versions.c.book_id == book_id,
                    timeline_versions.c.is_active.is_(True),
                    timeline_characters.c.id == character_id,
                )
            ).mappings().one_or_none()
        if row is None:
            return None
        mode = _required_string(row["mode"], "stored timeline mode")
        if mode not in {"enhanced", "compressed", "simple"}:
            raise InsightConflict(
                "stored timeline mode is invalid; clear current Insight data"
            )
        status = _required_string(row["status"], "stored timeline status")
        if status not in {"ready", "degraded", "stale"}:
            raise InsightConflict(
                "stored timeline status is invalid; clear current Insight data"
            )
        payload = _json_object(
            row["payload_json"],
            "timeline character payload",
        )
        if "characterId" in payload:
            raise InsightConflict(
                "stored timeline character contains a reserved id; "
                "clear current Insight data"
            )
        character = {
            **payload,
            "characterId": _required_string(
                row["character_id"],
                "stored timeline character id",
            ),
        }
        try:
            _validate_timeline_parts(
                content={},
                events=[],
                characters=[character],
                require_events=False,
            )
        except ValueError as exc:
            raise InsightConflict(
                "stored timeline character payload is invalid; "
                "clear current Insight data"
            ) from exc
        return {
            "timelineVersionId": _required_string(
                row["timeline_version_id"],
                "stored timeline id",
            ),
            "mode": mode,
            "status": status,
            "character": character,
        }

    def qa_status(
        self,
        *,
        book_id: str,
        mode: str = "exact",
    ) -> dict[str, Any]:
        if mode not in {"exact", "global"}:
            raise ValueError("mode must be exact or global")
        with self.engine.connect() as connection:
            self._assert_book_owned(connection, book_id)
            try:
                current = self._snapshot(connection, book_id=book_id)
            except InsightNotFound:
                return {
                    "available": False,
                    "reason": "analysis_missing",
                    "repairAction": "analyze",
                }
        if mode == "global":
            with self.engine.connect() as connection:
                rows = list(
                    connection.execute(
                        select(
                            analysis_artifacts.c.kind,
                            analysis_artifacts.c.template,
                            analysis_artifacts.c.status,
                            analysis_artifacts.c.dependency_fingerprint,
                        ).where(
                            analysis_artifacts.c.book_id == book_id,
                            analysis_artifacts.c.is_active.is_(True),
                        )
                    ).mappings()
                )
            artifacts: dict[tuple[str, str], Mapping[str, Any]] = {}
            for row in rows:
                key = (
                    _required_string(row["kind"], "Insight artifact kind"),
                    _required_string(row["template"], "Insight artifact template"),
                )
                if key in artifacts:
                    raise InsightConflict(
                        "multiple active Insight artifacts conflict; clear current Insight data"
                    )
                artifacts[key] = row
            required = (
                (
                    ("overview", "story_summary"),
                    "global_summary_missing",
                    "global_summary_stale",
                    "overview_rebuild",
                ),
                (
                    ("compressed_context", "default"),
                    "compressed_context_missing",
                    "compressed_context_stale",
                    "compressed_context_rebuild",
                ),
            )
            for key, missing_reason, stale_reason, repair_action in required:
                row = artifacts.get(key)
                if row is None:
                    return {
                        "available": False,
                        "reason": missing_reason,
                        "repairAction": repair_action,
                    }
                status = _required_string(
                    row["status"],
                    "Insight artifact status",
                )
                if status not in {"ready", "degraded", "stale"}:
                    raise InsightConflict(
                        "active Insight artifact status is invalid; "
                        "clear current Insight data"
                    )
                dependency_fingerprint = _required_sha256(
                    row["dependency_fingerprint"],
                    "Insight artifact dependency fingerprint",
                )
                if (
                    status not in {"ready", "degraded"}
                    or dependency_fingerprint != current.fingerprint
                ):
                    return {
                        "available": False,
                        "reason": stale_reason,
                        "repairAction": repair_action,
                    }
            return {
                "available": True,
                "reason": None,
            }
        with self.engine.connect() as connection:
            vector = connection.execute(
                select(vector_generations).where(
                    vector_generations.c.book_id == book_id,
                    vector_generations.c.is_active.is_(True),
                )
            ).mappings().one_or_none()
        if vector is None:
            return {
                "available": False,
                "reason": "vector_missing",
                "repairAction": "vector_rebuild",
            }
        vector_status = _required_string(vector["status"], "vector status")
        if vector_status not in {"ready", "degraded", "stale"}:
            raise InsightConflict(
                "active vector status is invalid; clear current Insight data"
            )
        dependency_fingerprint = _required_sha256(
            vector["dependency_fingerprint"],
            "vector dependency fingerprint",
        )
        if (
            dependency_fingerprint != current.fingerprint
            or vector_status == "stale"
        ):
            return {
                "available": False,
                "reason": "vector_stale",
                "repairAction": "vector_rebuild",
            }
        return {
            "available": True,
            "reason": None,
            "generation": _required_integer(
                vector["generation"],
                "vector generation",
                minimum=1,
            ),
            "coverage": {
                "pages": _required_integer(
                    vector["page_count"],
                    "vector page count",
                ),
                "events": _required_integer(
                    vector["event_count"],
                    "vector event count",
                ),
            },
        }


class InsightDerivedCommandService:
    def __init__(self, engine: Engine) -> None:
        self.jobs = JobQueueRepository(engine)
        self.settings = SettingsResolver(engine)
        self.repository = InsightDerivedRepository(engine)

    def create_job(
        self,
        *,
        book_id: str,
        kind: str,
        template: str,
        idempotency_key: str,
    ) -> dict[str, object]:
        book_id = _required_string(book_id, "Insight bookId")
        if kind not in DERIVED_KINDS:
            raise ValueError("unsupported Insight derived kind")
        if kind == "overview":
            if template not in OVERVIEW_TEMPLATES:
                raise ValueError("unsupported Insight overview template")
        elif template != "default":
            raise ValueError(f"{kind} template must be default")
        frozen = self.repository.snapshot(book_id=book_id)
        config = self.settings.resolve_insight(
            book_id=book_id,
            scope="full",
        )
        config.update(
            {
                "bookId": book_id,
                "derivedKind": kind,
                "template": template,
                "sourceRunId": frozen.source_run_id,
                "sourceRunStatus": frozen.source_run_status,
                "analysisInputs": [
                    {
                        "resultId": page["resultId"],
                        "pageId": page["pageId"],
                        "pageNumber": page["pageNumber"],
                        "currentSourceChecksum": page[
                            "currentSourceChecksum"
                        ],
                    }
                    for page in frozen.pages
                ],
                "analysisInputFingerprint": frozen.fingerprint,
            }
        )
        step = {
            "overview": "insight_build_overview",
            "compressed_context": "insight_build_compressed_context",
            "timeline": "insight_build_timeline",
            "vector": "insight_build_vectors",
        }[kind]
        job_kind = "vector_rebuild" if kind == "vector" else "derived_rebuild"
        return self.jobs.create_batch(
            kind=job_kind,
            display_name=f"Insight · {kind}",
            specs=(
                JobSpec(
                    kind=job_kind,
                    book_id=book_id,
                    analysis_run_id=frozen.source_run_id,
                    config=config,
                    items=(
                        JobItemSpec(
                            page_id=None,
                            step_kinds=(step,),
                        ),
                    ),
                    target_display={
                        "bookId": book_id,
                        "kind": kind,
                        "template": template,
                    },
                ),
            ),
            idempotency_scope=f"insight-derived:{book_id}:{kind}:{template}",
            idempotency_key=idempotency_key,
            idempotency_payload={
                "bookId": book_id,
                "kind": kind,
                "template": template,
                "fingerprint": frozen.fingerprint,
            },
        )


class InsightDerivedWorkerService:
    def __init__(
        self,
        *,
        data_root: Path,
        engine: Engine,
        jobs: JobQueueRepository,
        algorithms: DerivedAlgorithms | None = None,
        vector_store: InsightVectorStore | None = None,
    ) -> None:
        self.engine = engine
        self.jobs = jobs
        self.repository = InsightDerivedRepository(engine)
        self.credentials = SettingsRepository(engine)
        self.algorithms = algorithms or ProviderDerivedAlgorithms()
        self.vector_store = vector_store or InsightVectorStore(data_root)

    def handle(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        raw_config = _required_mapping(
            step.get("config"),
            "frozen Insight job config",
        )
        kind = _required_string(step.get("stepKind"), "Insight step kind")
        step_id = _required_string(step.get("stepId"), "Insight stepId")
        layer_match = re.fullmatch(
            r"insight_build_layer_(0|[1-9][0-9]*)",
            kind,
        )
        if kind in {"insight_build_vectors", "insight_stage_vectors"}:
            credential_sections = ("embedding",)
        elif (
            layer_match is not None
            or kind
            in {
                "insight_build_overview",
                "insight_stage_overview_no_spoiler",
                "insight_stage_overview_story_summary",
                "insight_build_compressed_context",
                "insight_stage_compressed_context",
                "insight_build_timeline",
                "insight_stage_timeline",
            }
        ):
            credential_sections = ("chat",)
        else:
            raise JobConflict(f"unsupported derived step: {kind}")
        config = self._with_credentials(
            raw_config,
            section_names=credential_sections,
        )
        book_id = _required_string(config.get("bookId"), "Insight bookId")
        run_id_value = config.get("runId")
        run_id = (
            _required_string(run_id_value, "Insight runId")
            if run_id_value is not None
            else None
        )
        scope = _required_string(config.get("scope"), "Insight scope")
        if scope not in {"full", "chapter", "page", "incremental"}:
            raise JobConflict("frozen Insight scope is invalid")
        if layer_match is not None and run_id is None:
            raise JobConflict("summary layer step is missing its analysis run")
        full_stage = (
            scope == "full"
            and run_id is not None
            and (
                layer_match is not None
                or kind.startswith("insight_stage_")
            )
        )
        if full_stage:
            if run_id is None:
                raise JobConflict("full Insight stage is missing its analysis run")
            frozen = self.repository.snapshot_for_run(run_id=run_id)
        else:
            frozen_inputs = config.get("analysisInputs")
            if (
                not book_id
                or not isinstance(frozen_inputs, list)
                or not all(
                    isinstance(value, Mapping) for value in frozen_inputs
                )
            ):
                raise JobConflict(
                    "derived job has an invalid frozen input snapshot"
                )
            frozen = self.repository.snapshot(
                book_id=book_id,
                frozen_inputs=frozen_inputs,
            )
            expected = _required_sha256(
                config.get("analysisInputFingerprint"),
                "Insight analysisInputFingerprint",
            )
            if frozen.fingerprint != expected:
                raise JobConflict(
                    "frozen Insight input fingerprint is invalid"
                )
            source_run_id = config.get("sourceRunId")
            source_run_status = config.get("sourceRunStatus")
            if source_run_id is not None:
                source_run_id = _required_string(
                    source_run_id,
                    "Insight sourceRunId",
                )
            if source_run_status is not None:
                source_run_status = _required_string(
                    source_run_status,
                    "Insight sourceRunStatus",
                )
                if source_run_status not in FINAL_ANALYSIS_RUN_STATUSES:
                    raise JobConflict("frozen Insight source run status is invalid")
            if (source_run_id is None) != (source_run_status is None):
                raise JobConflict(
                    "frozen Insight source run identity is incomplete"
                )
            frozen = AnalysisInputSnapshot(
                book_id=frozen.book_id,
                source_run_id=source_run_id,
                source_run_status=source_run_status,
                result_ids=frozen.result_ids,
                pages=frozen.pages,
                fingerprint=frozen.fingerprint,
            )
        try:
            if layer_match is not None:
                if run_id is None:
                    raise JobConflict(
                        "summary layer step is missing its analysis run"
                    )
                layer_index = int(layer_match.group(1))
                layer_units = self.repository.layer_units(
                    run_id=run_id,
                    layer_index=layer_index,
                    config=config,
                )
                if not layer_units:
                    raise InsightConflict("Insight layer has no units")
                layer_definition = _required_mapping(
                    layer_units[0].get("layer"),
                    "frozen Insight layer definition",
                )
                layer_name = _required_string(
                    layer_definition.get("name"),
                    "frozen Insight layer name",
                )
                completed_units = []
                for unit in layer_units:
                    content = self.algorithms.build_layer(
                        unit["inputs"],
                        layer=unit["layer"],
                        config=config,
                    )
                    if not isinstance(content, Mapping) or not contains_nonempty_text(content):
                        raise InsightConflict(
                            "summary layer algorithm returned an empty or non-object result"
                        )
                    completed_units.append(
                        {
                            **unit,
                            "content": dict(content),
                        }
                    )
                checkpoint: dict[str, Any] = {}

                def publish(connection: Connection) -> None:
                    checkpoint.update(
                        self.repository.publish_layer(
                            connection,
                            run_id=run_id,
                            layer_index=layer_index,
                            layer_name=layer_name,
                            units=completed_units,
                        )
                    )
            elif kind in {
                "insight_build_overview",
                "insight_stage_overview_no_spoiler",
                "insight_stage_overview_story_summary",
            }:
                template = {
                    "insight_stage_overview_no_spoiler": "no_spoiler",
                    "insight_stage_overview_story_summary": "story_summary",
                }.get(kind)
                if template is None:
                    template = _required_string(
                        config.get("template"),
                        "Insight overview template",
                    )
                if template not in OVERVIEW_TEMPLATES:
                    raise JobConflict("unsupported Insight overview template")
                summary_inputs = self.repository.summary_inputs(frozen)
                payload = self.algorithms.build_overview(
                    summary_inputs,
                    template=template,
                    config=config,
                )
                if not isinstance(payload, Mapping):
                    raise InsightConflict(
                        "overview algorithm returned a non-object result"
                    )
                _required_text(
                    payload.get("title"),
                    "overview result title",
                )
                _required_text(
                    payload.get("content"),
                    "overview result content",
                )
                checkpoint = {}

                def publish(connection: Connection) -> None:
                    checkpoint.update(
                        self.repository.publish_artifact(
                            connection=connection,
                            frozen=frozen,
                            kind="overview",
                            template=template,
                            payload=payload,
                            activate=not full_stage,
                        )
                    )
            elif kind in {
                "insight_build_compressed_context",
                "insight_stage_compressed_context",
            }:
                summary_inputs = self.repository.summary_inputs(frozen)
                payload = self.algorithms.build_compressed_context(
                    summary_inputs,
                    config=config,
                )
                if not isinstance(payload, Mapping) or not contains_nonempty_text(payload):
                    raise InsightConflict(
                        "compressed context algorithm returned an empty or non-object result"
                    )
                checkpoint = {}

                def publish(connection: Connection) -> None:
                    checkpoint.update(
                        self.repository.publish_artifact(
                            connection=connection,
                            frozen=frozen,
                            kind="compressed_context",
                            template="default",
                            payload=payload,
                            activate=not full_stage,
                        )
                    )
            elif kind in {
                "insight_build_timeline",
                "insight_stage_timeline",
            }:
                timeline_inputs = list(self.repository.summary_inputs(frozen))
                compressed_input = self.repository.compressed_context_input(frozen)
                if compressed_input is not None:
                    timeline_inputs.append(compressed_input)
                timeline = self.algorithms.build_timeline(
                    timeline_inputs,
                    config=config,
                )
                if not isinstance(timeline, Mapping):
                    raise InsightConflict(
                        "timeline algorithm returned a non-object result"
                    )
                checkpoint = {}

                def publish(connection: Connection) -> None:
                    checkpoint.update(
                        self.repository.publish_timeline(
                            connection=connection,
                            frozen=frozen,
                            result=timeline,
                            activate=not full_stage,
                        )
                    )
            elif kind in {
                "insight_build_vectors",
                "insight_stage_vectors",
            }:
                vector_build = self._build_vectors(
                    fence=fence,
                    step=step,
                    frozen=frozen,
                    config=config,
                )
                checkpoint = {
                    key: value
                    for key, value in vector_build.items()
                    if not key.startswith("__")
                }
                if vector_build.get("__control_drained__"):
                    return {
                        **checkpoint,
                        "__already_published__": True,
                        "__control_drained__": True,
                    }

                def publish(connection: Connection) -> None:
                    job_status = connection.execute(
                        select(jobs.c.status).where(jobs.c.id == fence.job_id)
                    ).scalar_one()
                    if job_status == "running":
                        checkpoint.update(
                            self.repository.publish_vector_generation(
                                connection=connection,
                                frozen=frozen,
                                generation=vector_build["generation"],
                                page_count=vector_build["pageCount"],
                                event_count=vector_build["eventCount"],
                                activate=not full_stage,
                            )
                        )
                    elif job_status in {"pausing", "cancelling"}:
                        self.repository.checkpoint_vector_generation(
                            connection=connection,
                            frozen=frozen,
                            generation=vector_build["generation"],
                            page_count=vector_build["pageCount"],
                            event_count=vector_build["eventCount"],
                        )
                    else:
                        raise JobConflict(
                            "vector job left a publishable control state"
                        )
                completed = self.jobs.complete_step(
                    fence,
                    step_id=step_id,
                    checkpoint=checkpoint,
                    publisher=publish,
                    defer_on_control=True,
                )
                return {
                    **checkpoint,
                    "__already_published__": True,
                    **({"__control_drained__": True} if not completed else {}),
                }
            else:
                raise JobConflict(f"unsupported derived step: {kind}")
            self.jobs.complete_step(
                fence,
                step_id=step_id,
                checkpoint=checkpoint,
                publisher=publish,
            )
            return {**checkpoint, "__already_published__": True}
        except AttemptFenced:
            raise

    def _build_vectors(
        self,
        *,
        fence: AttemptFence,
        step: Mapping[str, Any],
        frozen: AnalysisInputSnapshot,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        step_id = _required_string(step.get("stepId"), "Insight stepId")
        page_records: list[dict[str, Any]] = []
        event_records: list[dict[str, Any]] = []
        for index, page in enumerate(frozen.pages):
            analysis = _required_mapping(
                page.get("analysis"),
                f"vector page {index + 1} analysis",
            )
            summary = _required_string(
                analysis.get("page_summary"),
                f"vector page {index + 1} page_summary",
            ).strip()
            if not summary:
                raise InsightConflict(
                    f"vector page {index + 1} page_summary must not be blank"
                )
            page_id = _required_string(
                page.get("pageId"),
                f"vector page {index + 1} pageId",
            )
            page_number = _required_integer(
                page.get("pageNumber"),
                f"vector page {index + 1} pageNumber",
                minimum=1,
            )
            page_records.append(
                {
                    "id": f"page-{page_id}",
                    "document": summary,
                    "metadata": {
                        "book_id": frozen.book_id,
                        "page_id": page_id,
                        "page_number": page_number,
                        "type": "page",
                    },
                }
            )
        event_records.extend(self._layer_zero_event_records(frozen))
        raw_previous = step.get("checkpoint")
        if raw_previous is not None and not isinstance(raw_previous, Mapping):
            raise InsightConflict("vector checkpoint must be an object")
        resume = isinstance(raw_previous, Mapping) and bool(raw_previous)
        if resume:
            previous = _required_mapping(raw_previous, "vector checkpoint")
            if set(previous) != {
                "generation",
                "pageCount",
                "eventCount",
                "pageTotal",
                "eventTotal",
                "coverage",
            }:
                raise InsightConflict("vector checkpoint fields are invalid")
            generation = _required_integer(
                previous["generation"],
                "vector checkpoint generation",
                minimum=1,
            )
            page_count = _required_integer(
                previous["pageCount"],
                "vector checkpoint pageCount",
            )
            event_count = _required_integer(
                previous["eventCount"],
                "vector checkpoint eventCount",
            )
            page_total = _required_integer(
                previous["pageTotal"],
                "vector checkpoint pageTotal",
            )
            event_total = _required_integer(
                previous["eventTotal"],
                "vector checkpoint eventTotal",
            )
            coverage = previous["coverage"]
            if (
                isinstance(coverage, bool)
                or not isinstance(coverage, Real)
                or not math.isfinite(float(coverage))
            ):
                raise InsightConflict("vector checkpoint coverage is invalid")
            expected_total = len(page_records) + len(event_records)
            expected_coverage = (
                1.0
                if expected_total == 0
                else (page_count + event_count) / expected_total
            )
            if (
                page_total != len(page_records)
                or event_total != len(event_records)
                or not math.isclose(float(coverage), expected_coverage)
            ):
                raise InsightConflict("vector checkpoint does not match current inputs")
        else:
            generation = self.repository.next_vector_generation(frozen.book_id)
            page_count = 0
            event_count = 0
        if (
            generation < 1
            or page_count < 0
            or event_count < 0
            or page_count > len(page_records)
            or event_count > len(event_records)
        ):
            raise InsightConflict("vector checkpoint is invalid")
        checkpoint: dict[str, Any] = {
            "generation": generation,
            "pageCount": page_count,
            "eventCount": event_count,
            "pageTotal": len(page_records),
            "eventTotal": len(event_records),
        }

        def checkpoint_batch(kind: str, count: int) -> bool:
            if kind not in {"pages", "events"}:
                raise InsightConflict("vector store checkpoint kind is invalid")
            if isinstance(count, bool) or not isinstance(count, int):
                raise InsightConflict("vector store checkpoint count is invalid")
            if kind == "pages":
                if count < checkpoint["pageCount"] or count > len(page_records):
                    raise InsightConflict("vector page checkpoint is invalid")
                checkpoint["pageCount"] = count
            else:
                if count < checkpoint["eventCount"] or count > len(event_records):
                    raise InsightConflict("vector event checkpoint is invalid")
                checkpoint["eventCount"] = count
            total = len(page_records) + len(event_records)
            completed = checkpoint["pageCount"] + checkpoint["eventCount"]
            checkpoint["coverage"] = 1.0 if total == 0 else completed / total

            def publish_partial(connection: Connection) -> None:
                self.repository.checkpoint_vector_generation(
                    connection=connection,
                    frozen=frozen,
                    generation=generation,
                    page_count=checkpoint["pageCount"],
                    event_count=checkpoint["eventCount"],
                )

            status = self.jobs.checkpoint_step(
                fence,
                step_id=step_id,
                checkpoint=checkpoint,
                publisher=publish_partial,
            )
            return status == "running"

        try:
            result = self.vector_store.publish_batches(
                book_id=frozen.book_id,
                generation=generation,
                page_batches=self._embedding_batches(
                    page_records[page_count:],
                    config=config,
                ),
                event_batches=self._embedding_batches(
                    event_records[event_count:],
                    config=config,
                ),
                resume=resume,
                initial_page_count=page_count,
                initial_event_count=event_count,
                expected_page_count=len(page_records),
                expected_event_count=len(event_records),
                on_batch=checkpoint_batch,
            )
        except AttemptFenced:
            raise
        except Exception:
            self.repository.fail_vector_generation(
                book_id=frozen.book_id,
                generation=generation,
            )
            raise
        try:
            if not isinstance(result, Mapping):
                raise InsightConflict("vector store returned an invalid result")
            if set(result) != {"pageCount", "eventCount", "completed"}:
                raise InsightConflict("vector store result fields are invalid")
            if (
                isinstance(result["pageCount"], bool)
                or not isinstance(result["pageCount"], int)
                or isinstance(result["eventCount"], bool)
                or not isinstance(result["eventCount"], int)
                or not isinstance(result["completed"], bool)
            ):
                raise InsightConflict("vector store result types are invalid")
            if (
                result["pageCount"] < page_count
                or result["pageCount"] > len(page_records)
                or result["eventCount"] < event_count
                or result["eventCount"] > len(event_records)
            ):
                raise InsightConflict("vector store result counts are invalid")
            checkpoint["pageCount"] = result["pageCount"]
            checkpoint["eventCount"] = result["eventCount"]
            if not result["completed"]:
                return {**checkpoint, "__control_drained__": True}
            if (
                result["pageCount"] != len(page_records)
                or result["eventCount"] != len(event_records)
            ):
                raise InsightConflict("vector store completed with incomplete coverage")
        except Exception:
            self.repository.fail_vector_generation(
                book_id=frozen.book_id,
                generation=generation,
            )
            raise
        checkpoint["coverage"] = 1.0
        return {
            **checkpoint,
        }

    def _embedding_batches(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        config: Mapping[str, Any],
        batch_size: int = 64,
    ) -> Iterable[
        tuple[Sequence[Mapping[str, Any]], Sequence[Sequence[float]]]
    ]:
        expected_dimension: int | None = None
        for offset in range(0, len(records), batch_size):
            batch = records[offset : offset + batch_size]
            documents = [
                _required_string(
                    row.get("document"),
                    f"embedding record {offset + index + 1} document",
                )
                for index, row in enumerate(batch)
            ]
            embeddings = list(
                self.algorithms.embed_documents(documents, config=config)
            )
            if len(embeddings) != len(batch):
                raise InsightConflict("embedding result count mismatch")
            for index, embedding in enumerate(embeddings):
                if (
                    not isinstance(embedding, Sequence)
                    or isinstance(embedding, (str, bytes, bytearray))
                    or not embedding
                ):
                    raise InsightConflict(
                        f"embedding result {offset + index + 1} is invalid"
                    )
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(float(value))
                    for value in embedding
                ):
                    raise InsightConflict(
                        f"embedding result {offset + index + 1} contains invalid values"
                    )
                if expected_dimension is None:
                    expected_dimension = len(embedding)
                elif len(embedding) != expected_dimension:
                    raise InsightConflict("embedding result dimensions do not match")
            yield batch, embeddings

    def _layer_zero_event_records(
        self,
        frozen: AnalysisInputSnapshot,
    ) -> list[dict[str, Any]]:
        if not frozen.source_run_id:
            return []
        with self.engine.connect() as connection:
            layers = list(
                connection.execute(
                    select(
                        analysis_layer_results.c.id,
                        analysis_layer_results.c.content_json,
                    )
                    .where(
                        analysis_layer_results.c.run_id
                        == frozen.source_run_id,
                        analysis_layer_results.c.layer_index == 0,
                        analysis_layer_results.c.status.in_(
                            ("staging", "published")
                        ),
                    )
                    .order_by(analysis_layer_results.c.unit_index)
                ).mappings()
            )
            page_rows = list(
                connection.execute(
                    select(
                        analysis_layer_result_pages.c.layer_result_id,
                        analysis_layer_result_pages.c.page_id_snapshot,
                        analysis_layer_result_pages.c.page_number_snapshot,
                    )
                    .where(
                        analysis_layer_result_pages.c.layer_result_id.in_(
                            tuple(
                                _required_string(
                                    row["id"],
                                    "stored analysis layer id",
                                )
                                for row in layers
                            )
                        )
                    )
                    .order_by(
                        analysis_layer_result_pages.c.layer_result_id,
                        analysis_layer_result_pages.c.ordinal,
                    )
                )
            ) if layers else []
        pages_by_layer: dict[str, list[tuple[str, int]]] = {}
        for layer_result_id, page_id, page_number in page_rows:
            pages_by_layer.setdefault(
                _required_string(
                    layer_result_id,
                    "stored analysis layer page layer id",
                ),
                [],
            ).append(
                (
                    _required_string(
                        page_id,
                        "stored analysis layer page id",
                    ),
                    _required_integer(
                        page_number,
                        "stored analysis layer page number",
                        minimum=1,
                    ),
                )
            )
        records: list[dict[str, Any]] = []
        for layer in layers:
            layer_id = _required_string(
                layer["id"],
                "stored analysis layer id",
            )
            content = _json_object(
                layer["content_json"],
                "analysis layer content",
            )
            page_refs = pages_by_layer.get(layer_id, [])
            if not page_refs:
                raise InsightConflict(
                    "analysis layer has no covered pages; clear current Insight data"
                )
            raw_events = content.get("key_events", [])
            if not isinstance(raw_events, list):
                raise InsightConflict(
                    "analysis layer key_events must be an array; clear current Insight data"
                )
            for index, event in enumerate(raw_events, start=1):
                if not isinstance(event, Mapping):
                    raise InsightConflict(
                        "analysis layer event must be an object; clear current Insight data"
                    )
                summary = event.get("summary")
                importance = event.get("importance", "normal")
                if not isinstance(summary, str) or not summary.strip():
                    raise InsightConflict(
                        "analysis layer event summary is invalid; clear current Insight data"
                    )
                if not isinstance(importance, str):
                    raise InsightConflict(
                        "analysis layer event importance is invalid; clear current Insight data"
                    )
                text = summary.strip()
                records.append(
                    {
                        "id": f"event-{layer_id}-{index}",
                        "document": text,
                        "metadata": {
                            "book_id": frozen.book_id,
                            "page_id": page_refs[0][0],
                            "page_number": page_refs[0][1],
                            "page_ids_json": _json(
                                [value[0] for value in page_refs]
                            ),
                            "page_numbers_json": _json(
                                [value[1] for value in page_refs]
                            ),
                            "importance": importance,
                            "type": "event",
                        },
                    }
                )
        return records

    def _with_credentials(
        self,
        config: Mapping[str, Any],
        *,
        section_names: Sequence[str],
    ) -> dict[str, Any]:
        try:
            return self.credentials.resolve_credential_sections(
                config,
                section_names,
            )
        except LookupError as exc:
            raise JobConflict(
                "frozen Insight credential version no longer exists"
            ) from exc


def _publication_status(
    frozen: AnalysisInputSnapshot,
    current: AnalysisInputSnapshot,
) -> str:
    if frozen.fingerprint != current.fingerprint:
        return "stale"
    if frozen.source_run_status == "completed_with_errors":
        return "degraded"
    return "ready"


def _analysis_input_fingerprint(
    pages_payload: Sequence[Mapping[str, Any]],
) -> str:
    """Hash the same immutable identity fields before and after publication."""

    canonical: list[dict[str, object]] = []
    for index, page in enumerate(pages_payload, start=1):
        canonical.append(
            {
                "resultId": _required_string(
                    page.get("resultId"),
                    f"analysis input {index} resultId",
                ),
                "pageId": _required_string(
                    page.get("pageId"),
                    f"analysis input {index} pageId",
                ),
                "pageNumber": _required_integer(
                    page.get("pageNumber"),
                    f"analysis input {index} pageNumber",
                    minimum=1,
                ),
                "sourceChecksum": _required_sha256(
                    page.get("sourceChecksum"),
                    f"analysis input {index} sourceChecksum",
                ),
                "currentSourceChecksum": _required_sha256(
                    page.get("currentSourceChecksum"),
                    f"analysis input {index} currentSourceChecksum",
                ),
            }
        )
    return hashlib.sha256(_json(canonical).encode("utf-8")).hexdigest()


def _layer_prompt_type(
    *,
    layer_index: int,
    layer_count: int,
    align_to_chapter: bool,
) -> str:
    if layer_index == 0:
        return "batch_analysis"
    if layer_index == layer_count - 1:
        return "book_overview"
    if align_to_chapter:
        return "chapter_summary"
    return "segment_summary"


def _page_context(pages: Sequence[Mapping[str, Any]]) -> str:
    contexts = []
    for index, page in enumerate(pages, start=1):
        analysis = _required_mapping(
            page.get("analysis"),
            f"Insight context input {index} analysis",
        )
        if not analysis:
            raise InsightConflict(
                f"Insight context input {index} analysis must not be empty"
            )
        contexts.append(_page_context_label(page) + "\n" + _json(analysis))
    if not contexts:
        raise InsightConflict("Insight context inputs must not be empty")
    return "\n\n".join(contexts)


def _page_context_label(page: Mapping[str, Any]) -> str:
    raw_page_ids = page.get("pageIds")
    raw_page_numbers = page.get("pageNumbers")
    if raw_page_ids is None:
        page_ids = [_required_string(page.get("pageId"), "Insight context pageId")]
    else:
        if not isinstance(raw_page_ids, list) or not raw_page_ids:
            raise InsightConflict("Insight context pageIds must be a non-empty array")
        page_ids = [
            _required_string(value, "Insight context pageId")
            for value in raw_page_ids
        ]
    if raw_page_numbers is None:
        page_numbers = [
            _required_integer(
                page.get("pageNumber"),
                "Insight context pageNumber",
                minimum=1,
            )
        ]
    else:
        if not isinstance(raw_page_numbers, list) or not raw_page_numbers:
            raise InsightConflict(
                "Insight context pageNumbers must be a non-empty array"
            )
        page_numbers = [
            _required_integer(
                value,
                "Insight context pageNumber",
                minimum=1,
            )
            for value in raw_page_numbers
        ]
    if len(page_ids) != len(page_numbers):
        raise InsightConflict("Insight context page references do not match")
    page_range = (
        str(page_numbers[0])
        if len(page_numbers) == 1
        else f"{page_numbers[0]}-{page_numbers[-1]}"
    )
    return f"第 {page_range} 页（page_ids={_json(page_ids)}）"
