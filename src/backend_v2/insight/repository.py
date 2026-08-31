"""Relational fact source and publication transactions for Manga Insight."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import base64
import binascii
from datetime import datetime, timedelta
import hashlib
import json
import math
from typing import Any
import uuid

from sqlalchemy import Engine, case, delete, func, insert, or_, select, update
from sqlalchemy.engine import Connection

from src.backend_v2.auth.ownership import effective_owner_id
from src.backend_v2.serialization import canonical_json as _json
from src.backend_v2.timestamps import iso_utc as _iso, utcnow
from src.backend_v2.redaction import redact_sensitive_text
from src.backend_v2.insight.page_schema import (
    InvalidPageAnalysis,
    validate_persisted_page_analysis,
)
from src.backend_v2.jobs.repository import decode_job_progress
from src.backend_v2.storage.database import immediate_transaction
from src.backend_v2.storage.schema import (
    JOB_STATUSES,
    NONTERMINAL_JOB_STATUSES,
    analysis_artifacts,
    analysis_heads,
    analysis_layer_results,
    analysis_page_results,
    analysis_run_targets,
    analysis_runs,
    assets,
    books,
    chapters,
    idempotency_records,
    jobs,
    note_citations,
    notes,
    page_assets,
    pages,
    timeline_versions,
    timeline_events,
    vector_generations,
)


ANALYSIS_RUN_SCOPES = frozenset({"full", "incremental", "chapter", "page"})
ANALYSIS_RUN_STATUSES = frozenset(
    {"staging", "completed", "completed_with_errors", "failed", "cancelled"}
)
ANALYSIS_TARGET_STATUSES = frozenset(
    {"pending", "completed", "failed", "conflict"}
)
OVERVIEW_TEMPLATES = frozenset(
    {
        "no_spoiler",
        "story_summary",
        "recap",
        "character_guide",
        "world_setting",
        "highlights",
        "reading_notes",
    }
)


class InsightNotFound(LookupError):
    pass


class InsightConflict(RuntimeError):
    pass


class InsightLocked(InsightConflict):
    pass


def _idempotency_replay(
    connection: Connection,
    *,
    scope: str,
    key: str,
    payload: Mapping[str, Any],
    now: datetime,
) -> tuple[str, dict[str, Any] | None]:
    if not isinstance(key, str) or not key or len(key) > 200:
        raise ValueError(
            "Idempotency-Key is required and must be at most 200 characters"
        )
    request_hash = hashlib.sha256(_json(dict(payload)).encode("utf-8")).hexdigest()
    row = connection.execute(
        select(
            idempotency_records.c.request_hash,
            idempotency_records.c.response_json,
            idempotency_records.c.expires_at,
        ).where(
            idempotency_records.c.scope == scope,
            idempotency_records.c.key == key,
            idempotency_records.c.owner_user_id == effective_owner_id(),
        )
    ).mappings().one_or_none()
    if row is None:
        return request_hash, None
    expires_at = _required_datetime(
        row["expires_at"],
        "idempotency record expiresAt",
    )
    if expires_at <= now:
        connection.execute(
            delete(idempotency_records).where(
                idempotency_records.c.scope == scope,
                idempotency_records.c.key == key,
                idempotency_records.c.owner_user_id == effective_owner_id(),
            )
        )
        return request_hash, None
    if row["request_hash"] != request_hash:
        raise InsightConflict(
            "Idempotency-Key was reused for a different Insight mutation"
        )
    response = _json_object(
        row["response_json"],
        "idempotency response",
    )
    return request_hash, response


def _record_idempotency(
    connection: Connection,
    *,
    scope: str,
    key: str,
    request_hash: str,
    response: Mapping[str, Any],
    http_status: int,
    resource_type: str | None,
    resource_id: str | None,
    now: datetime,
) -> None:
    connection.execute(
        insert(idempotency_records).values(
            owner_user_id=effective_owner_id(),
            scope=scope,
            key=key,
            request_hash=request_hash,
            http_status=http_status,
            response_json=_json(dict(response)),
            resource_type=resource_type,
            resource_id=resource_id,
            created_at=now,
            expires_at=now + timedelta(days=7),
        )
    )


def _load_json(value: object, field: str) -> object:
    if not isinstance(value, str):
        raise InsightConflict(f"stored {field} is missing; clear current Insight data")
    try:
        return json.loads(value)
    except (TypeError, ValueError) as exc:
        raise InsightConflict(
            f"stored {field} is invalid; clear current Insight data"
        ) from exc


def _json_object(value: object, field: str) -> dict[str, Any]:
    parsed = _load_json(value, field)
    if not isinstance(parsed, Mapping):
        raise InsightConflict(f"stored {field} must be an object")
    return dict(parsed)


def _json_array(value: object, field: str) -> list[Any]:
    parsed = _load_json(value, field)
    if not isinstance(parsed, list):
        raise InsightConflict(f"stored {field} must be an array")
    return parsed


def _optional_json_object(value: object, field: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return _json_object(value, field)


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise InsightConflict(
            f"stored {field} must be a non-empty string; clear current Insight data"
        )
    return value


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, field)


def _required_integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InsightConflict(
            f"stored {field} must be an integer of at least {minimum}; "
            "clear current Insight data"
        )
    return value


def _required_datetime(value: object, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise InsightConflict(
            f"stored {field} must be a timestamp; clear current Insight data"
        )
    return value


def _optional_datetime(value: object, field: str) -> datetime | None:
    if value is None:
        return None
    return _required_datetime(value, field)


def _required_sha256(value: object, field: str) -> str:
    text = _required_string(value, field)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise InsightConflict(
            f"stored {field} must be a lowercase SHA-256 digest; "
            "clear current Insight data"
        )
    return text


def _required_boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise InsightConflict(
            f"stored {field} must be a boolean; clear current Insight data"
        )
    return value


def contains_nonempty_text(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return any(contains_nonempty_text(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return any(contains_nonempty_text(item) for item in value)
    return False


def _page_analysis(
    value: object,
    field: str,
    *,
    page_id: object | None = None,
    page_number: object | None = None,
    source_asset_id: object | None = None,
    source_checksum: object | None = None,
) -> dict[str, Any]:
    parsed = _json_object(value, field)
    try:
        payload = validate_persisted_page_analysis(parsed)
    except InvalidPageAnalysis as exc:
        raise InsightConflict(
            f"stored {field} is invalid; clear current Insight data"
        ) from exc
    expected = {
        "page_id": page_id,
        "page_number_snapshot": page_number,
        "source_asset_id": source_asset_id,
        "source_checksum": source_checksum,
    }
    for key, expected_value in expected.items():
        if expected_value is not None and payload[key] != expected_value:
            raise InsightConflict(
                f"stored {field} identity does not match its row; "
                "clear current Insight data"
            )
    return payload


def _encode_note_cursor(updated_at: datetime, note_id: str) -> str:
    raw = f"{updated_at.isoformat()}|{note_id}".encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_note_cursor(value: str) -> tuple[datetime, str]:
    if not isinstance(value, str) or not value:
        raise ValueError("invalid note cursor")
    try:
        padded = value + "=" * (-len(value) % 4)
        decoded = base64.b64decode(
            padded.encode("ascii"),
            altchars=b"-_",
            validate=True,
        ).decode("utf-8")
        timestamp, note_id = decoded.rsplit("|", 1)
        parsed = datetime.fromisoformat(timestamp)
    except (binascii.Error, ValueError, UnicodeError) as exc:
        raise ValueError("invalid note cursor") from exc
    if (
        not note_id
        or parsed.tzinfo is not None
        or _encode_note_cursor(parsed, note_id) != value
    ):
        raise ValueError("invalid note cursor")
    return parsed, note_id


def _optional_note_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be null or a trimmed non-empty string")
    return value


def _validate_note_metadata(
    *,
    kind: str,
    tags: Sequence[str],
    question: str | None,
    comment: str | None,
) -> tuple[str, list[str], dict[str, str | None]]:
    if kind not in {"text", "qa"}:
        raise ValueError("note kind must be text or qa")
    if isinstance(tags, (str, bytes)) or not isinstance(tags, Sequence):
        raise ValueError("note tags must be an array")
    normalized_tags = list(tags)
    if any(
        not isinstance(value, str)
        or not value
        or value != value.strip()
        for value in normalized_tags
    ):
        raise ValueError("note tags must be trimmed non-empty strings")
    if len(set(normalized_tags)) != len(normalized_tags):
        raise ValueError("note tags must be unique")
    normalized_question = _optional_note_text(question, "note question")
    normalized_comment = _optional_note_text(comment, "note comment")
    if kind == "qa" and normalized_question is None:
        raise ValueError("qa note question is required")
    if kind == "text" and normalized_question is not None:
        raise ValueError("text note question must be null")
    return kind, normalized_tags, {
        "question": normalized_question,
        "comment": normalized_comment,
    }


def _stored_note_metadata(value: object) -> dict[str, str | None]:
    metadata = _json_object(value, "note metadata")
    if set(metadata) != {"question", "comment"}:
        raise InsightConflict(
            "stored note metadata is obsolete; clear current Insight data"
        )
    normalized: dict[str, str | None] = {}
    for field in ("question", "comment"):
        raw = metadata[field]
        if raw is not None and (
            not isinstance(raw, str)
            or not raw
            or raw != raw.strip()
        ):
            raise InsightConflict(
                "stored note metadata is invalid; clear current Insight data"
            )
        normalized[field] = raw
    return normalized


def _normalize_note_citations(
    citations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(citations, (str, bytes)) or not isinstance(
        citations,
        Sequence,
    ):
        raise ValueError("citations must be an array")
    normalized: list[dict[str, Any]] = []
    for value in citations:
        if not isinstance(value, Mapping):
            raise ValueError("every citation must be an object")
        citation = dict(value)
        unknown = set(citation) - {"pageId", "excerpt", "score"}
        if unknown:
            raise ValueError(
                "citation has unknown fields: " + ", ".join(sorted(unknown))
            )
        page_id = citation.get("pageId")
        if not isinstance(page_id, str) or not page_id:
            raise ValueError("every citation requires pageId")
        excerpt = citation.get("excerpt", "")
        if not isinstance(excerpt, str):
            raise ValueError("citation excerpt must be a string")
        score = citation.get("score")
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise ValueError("citation score must be a finite number or null")
        normalized.append(citation)
    page_ids = [citation["pageId"] for citation in normalized]
    if len(set(page_ids)) != len(page_ids):
        raise ValueError("citation pageIds must be unique")
    return normalized


def mark_book_insight_derived_stale(
    connection: Connection,
    *,
    book_id: str,
    now: datetime,
) -> None:
    """Invalidate current whole-book derivatives after content facts change."""

    if connection.execute(
        select(analysis_heads.c.id)
        .where(analysis_heads.c.book_id == book_id)
        .limit(1)
    ).scalar_one_or_none() is None:
        return
    active_book_run = select(analysis_heads.c.active_run_id).where(
        analysis_heads.c.book_id == book_id,
        analysis_heads.c.page_id.is_(None),
    ).scalar_subquery()
    connection.execute(
        update(analysis_layer_results)
        .where(
            analysis_layer_results.c.run_id == active_book_run,
            analysis_layer_results.c.status == "published",
        )
        .values(status="stale", updated_at=now)
    )
    connection.execute(
        update(analysis_artifacts)
        .where(
            analysis_artifacts.c.book_id == book_id,
            analysis_artifacts.c.is_active.is_(True),
            analysis_artifacts.c.status.in_(("ready", "degraded")),
        )
        .values(
            status="stale",
            revision=analysis_artifacts.c.revision + 1,
            updated_at=now,
        )
    )
    connection.execute(
        update(timeline_versions)
        .where(
            timeline_versions.c.book_id == book_id,
            timeline_versions.c.is_active.is_(True),
            timeline_versions.c.status.in_(("ready", "degraded")),
        )
        .values(status="stale", updated_at=now)
    )
    connection.execute(
        update(vector_generations)
        .where(
            vector_generations.c.book_id == book_id,
            vector_generations.c.is_active.is_(True),
            vector_generations.c.status.in_(("ready", "degraded")),
        )
        .values(status="stale", updated_at=now)
    )


class InsightRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    @staticmethod
    def insert_run(
        connection: Connection,
        *,
        run_id: str,
        job_id: str,
        book_id: str,
        scope: str,
        config: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
    ) -> None:
        now = utcnow()
        connection.execute(
            insert(analysis_runs).values(
                id=run_id,
                owner_user_id=effective_owner_id(),
                book_id=book_id,
                job_id=job_id,
                scope=scope,
                status="staging",
                config_json=_json(dict(config)),
                schema_version=2,
                target_count=len(targets),
                success_count=0,
                failed_count=0,
                created_at=now,
                updated_at=now,
            )
        )
        connection.execute(
            insert(analysis_run_targets),
            [
                {
                    "run_id": run_id,
                    "ordinal": index,
                    "page_id": target["page_id"],
                    "chapter_id": target["chapter_id"],
                    "source_asset_id": target["source_asset_id"],
                    "source_checksum": target["source_checksum"],
                    "page_id_snapshot": target["page_id"],
                    "page_number_snapshot": target["page_number"],
                    "status": "pending",
                }
                for index, target in enumerate(targets, start=1)
            ],
        )

    def run_target(
        self,
        *,
        run_id: str,
        page_id: str,
    ) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(analysis_run_targets).where(
                    analysis_run_targets.c.run_id == run_id,
                    analysis_run_targets.c.page_id_snapshot == page_id,
                )
            ).mappings().one_or_none()
        if row is None:
            raise InsightNotFound("analysis run target not found")
        return self._run_target_dict(row, run_id=run_id, page_id=page_id)

    @staticmethod
    def _run_target_dict(
        row: Mapping[str, Any],
        *,
        run_id: str,
        page_id: str,
    ) -> dict[str, Any]:
        stored_run_id = _required_string(
            row["run_id"],
            "analysis run target run id",
        )
        stored_page_id = _required_string(
            row["page_id_snapshot"],
            "analysis run target page id snapshot",
        )
        if stored_run_id != run_id or stored_page_id != page_id:
            raise InsightConflict(
                "stored analysis run target identity is invalid; "
                "clear current Insight data"
            )
        status = _required_string(row["status"], "analysis run target status")
        if status not in ANALYSIS_TARGET_STATUSES:
            raise InsightConflict(
                "stored analysis run target status is invalid; "
                "clear current Insight data"
            )
        return {
            "run_id": stored_run_id,
            "ordinal": _required_integer(
                row["ordinal"],
                "analysis run target ordinal",
                minimum=1,
            ),
            "page_id": _optional_string(
                row["page_id"],
                "analysis run target current page id",
            ),
            "chapter_id": _optional_string(
                row["chapter_id"],
                "analysis run target current chapter id",
            ),
            "source_asset_id": _required_string(
                row["source_asset_id"],
                "analysis run target source asset id",
            ),
            "source_checksum": _required_sha256(
                row["source_checksum"],
                "analysis run target source checksum",
            ),
            "page_id_snapshot": stored_page_id,
            "page_number_snapshot": _required_integer(
                row["page_number_snapshot"],
                "analysis run target page number snapshot",
                minimum=1,
            ),
            "status": status,
            "error": _optional_json_object(
                row["error_json"],
                "analysis run target error",
            ),
        }

    @staticmethod
    def _batch_group_value(
        row: Mapping[str, Any],
        grouping: str,
    ) -> str | int | None:
        if grouping == "global":
            return None
        if grouping == "chapter":
            return _optional_string(
                row["chapter_id"],
                "analysis run target current chapter id",
            )
        if grouping == "contiguous":
            page_number = _required_integer(
                row["page_number_snapshot"],
                "analysis run target page number snapshot",
                minimum=1,
            )
            ordinal = _required_integer(
                row["ordinal"],
                "analysis run target ordinal",
                minimum=1,
            )
            return page_number - ordinal
        raise ValueError("Insight batch grouping is invalid")

    @staticmethod
    def _batch_group_starts(
        connection: Connection,
        *,
        run_id: str,
        grouping: str,
    ) -> dict[str | int | None, int]:
        if grouping == "global":
            first = connection.execute(
                select(func.min(analysis_run_targets.c.ordinal)).where(
                    analysis_run_targets.c.run_id == run_id
                )
            ).scalar_one()
            return {
                None: _required_integer(
                    first,
                    "analysis run first target ordinal",
                    minimum=1,
                )
            }
        if grouping == "chapter":
            rows = connection.execute(
                select(
                    analysis_run_targets.c.chapter_id,
                    func.min(analysis_run_targets.c.ordinal),
                )
                .where(analysis_run_targets.c.run_id == run_id)
                .group_by(analysis_run_targets.c.chapter_id)
            )
            return {
                _optional_string(chapter_id, "analysis run target chapter id"):
                _required_integer(
                    first_ordinal,
                    "analysis run chapter first ordinal",
                    minimum=1,
                )
                for chapter_id, first_ordinal in rows
            }
        if grouping == "contiguous":
            group_key = (
                analysis_run_targets.c.page_number_snapshot
                - analysis_run_targets.c.ordinal
            ).label("group_key")
            rows = connection.execute(
                select(
                    group_key,
                    func.min(analysis_run_targets.c.ordinal),
                )
                .where(analysis_run_targets.c.run_id == run_id)
                .group_by(group_key)
            )
            return {
                int(key): _required_integer(
                    first_ordinal,
                    "analysis run contiguous range first ordinal",
                    minimum=1,
                )
                for key, first_ordinal in rows
            }
        raise ValueError("Insight batch grouping is invalid")

    @classmethod
    def _batch_start_ordinal(
        cls,
        row: Mapping[str, Any],
        *,
        grouping: str,
        group_starts: Mapping[str | int | None, int],
        pages_per_batch: int,
    ) -> int:
        ordinal = _required_integer(
            row["ordinal"],
            "analysis run target ordinal",
            minimum=1,
        )
        group = cls._batch_group_value(row, grouping)
        first_ordinal = group_starts.get(group)
        if first_ordinal is None or ordinal < first_ordinal:
            raise InsightConflict(
                "stored analysis run batch grouping is invalid; "
                "clear current Insight data"
            )
        return (
            first_ordinal
            + ((ordinal - first_ordinal) // pages_per_batch) * pages_per_batch
        )

    def batch_window(
        self,
        *,
        run_id: str,
        target: Mapping[str, Any],
        pages_per_batch: int,
        grouping: str,
    ) -> tuple[int, list[dict[str, Any]]]:
        """Return the target's frozen batch without crossing its group."""

        with self.engine.connect() as connection:
            group_starts = self._batch_group_starts(
                connection,
                run_id=run_id,
                grouping=grouping,
            )
            first_ordinal = self._batch_start_ordinal(
                target,
                grouping=grouping,
                group_starts=group_starts,
                pages_per_batch=pages_per_batch,
            )
            group = self._batch_group_value(target, grouping)
            conditions = [
                analysis_run_targets.c.run_id == run_id,
                analysis_run_targets.c.ordinal >= first_ordinal,
                analysis_run_targets.c.ordinal
                < first_ordinal + pages_per_batch,
            ]
            if grouping == "chapter":
                conditions.append(
                    analysis_run_targets.c.chapter_id.is_(None)
                    if group is None
                    else analysis_run_targets.c.chapter_id == group
                )
            elif grouping == "contiguous":
                conditions.append(
                    analysis_run_targets.c.page_number_snapshot
                    - analysis_run_targets.c.ordinal
                    == group
                )
            rows = list(
                connection.execute(
                    select(analysis_run_targets)
                    .where(*conditions)
                    .order_by(analysis_run_targets.c.ordinal)
                ).mappings()
            )
        return (
            first_ordinal,
            [
                self._run_target_dict(
                    row,
                    run_id=run_id,
                    page_id=str(row["page_id_snapshot"]),
                )
                for row in rows
            ],
        )

    def previous_successful_batches(
        self,
        *,
        run_id: str,
        before_ordinal: int,
        pages_per_batch: int,
        batch_count: int,
        grouping: str,
        context_chapter_id: str | None = None,
    ) -> list[list[dict[str, Any]]]:
        if batch_count == 0:
            return []
        batches: dict[int, list[tuple[int, dict[str, Any]]]] = {}
        with self.engine.connect() as connection:
            group_starts = self._batch_group_starts(
                connection,
                run_id=run_id,
                grouping=grouping,
            )
            conditions = [
                analysis_run_targets.c.run_id == run_id,
                analysis_run_targets.c.ordinal < before_ordinal,
            ]
            if context_chapter_id is not None:
                conditions.append(
                    analysis_run_targets.c.chapter_id == context_chapter_id
                )
            rows = connection.execute(
                select(
                    analysis_run_targets.c.ordinal,
                    analysis_run_targets.c.chapter_id,
                    analysis_run_targets.c.page_number_snapshot,
                    analysis_page_results.c.payload_json,
                )
                .join(
                    analysis_page_results,
                    (
                        analysis_page_results.c.run_id
                        == analysis_run_targets.c.run_id
                    )
                    & (
                        analysis_page_results.c.page_id_snapshot
                        == analysis_run_targets.c.page_id_snapshot
                    ),
                )
                .where(*conditions)
                .order_by(analysis_run_targets.c.ordinal.desc())
            ).mappings()
            for row in rows:
                ordinal = _required_integer(
                    row["ordinal"],
                    "analysis run target ordinal",
                    minimum=1,
                )
                batch_start = self._batch_start_ordinal(
                    row,
                    grouping=grouping,
                    group_starts=group_starts,
                    pages_per_batch=pages_per_batch,
                )
                if batch_start not in batches and len(batches) == batch_count:
                    break
                batches.setdefault(batch_start, []).append(
                    (
                        ordinal,
                        _json_object(
                            row["payload_json"],
                            "analysis page payload",
                        ),
                    )
                )
        return [
            [
                payload
                for _ordinal, payload in sorted(batches[batch_index])
            ]
            for batch_index in sorted(batches)
        ]

    def previous_active_batches(
        self,
        *,
        book_id: str,
        before_page_number: int,
        pages_per_batch: int,
        batch_count: int,
        align_to_chapter: bool,
    ) -> list[list[dict[str, Any]]]:
        """Rebuild prior published batches for an incremental run's context."""

        if batch_count == 0:
            return []
        source_pointer = page_assets.alias("insight_context_source")
        page_head = analysis_heads.alias("insight_context_head")
        active_result = analysis_page_results.alias("insight_context_result")
        with self.engine.connect() as connection:
            rows = list(
                connection.execute(
                    select(
                        pages.c.chapter_id,
                        assets.c.checksum.label("current_source_checksum"),
                        active_result.c.source_checksum.label(
                            "analysis_source_checksum"
                        ),
                        active_result.c.status.label("analysis_status"),
                        active_result.c.payload_json,
                    )
                    .join(chapters, chapters.c.id == pages.c.chapter_id)
                    .join(
                        source_pointer,
                        (source_pointer.c.page_id == pages.c.id)
                        & (source_pointer.c.role == "source"),
                    )
                    .join(assets, assets.c.id == source_pointer.c.asset_id)
                    .join(
                        page_head,
                        page_head.c.page_id == pages.c.id,
                        isouter=True,
                    )
                    .join(
                        active_result,
                        active_result.c.id == page_head.c.active_result_id,
                        isouter=True,
                    )
                    .where(chapters.c.book_id == book_id)
                    .order_by(chapters.c.ordinal, pages.c.ordinal)
                ).mappings()
            )

        numbered = [
            {**dict(row), "page_number_snapshot": page_number}
            for page_number, row in enumerate(rows, start=1)
        ]
        grouped: list[list[dict[str, Any]]] = []
        if align_to_chapter:
            chapter_rows: dict[str, list[dict[str, Any]]] = {}
            chapter_order: list[str] = []
            for row in numbered:
                chapter_id = _required_string(
                    row["chapter_id"],
                    "active analysis chapter id",
                )
                if chapter_id not in chapter_rows:
                    chapter_order.append(chapter_id)
                    chapter_rows[chapter_id] = []
                chapter_rows[chapter_id].append(row)
            for chapter_id in chapter_order:
                values = chapter_rows[chapter_id]
                grouped.extend(
                    values[offset : offset + pages_per_batch]
                    for offset in range(0, len(values), pages_per_batch)
                )
        else:
            grouped = [
                numbered[offset : offset + pages_per_batch]
                for offset in range(0, len(numbered), pages_per_batch)
            ]

        valid: list[list[dict[str, Any]]] = []
        for batch in grouped:
            if not batch or int(batch[-1]["page_number_snapshot"]) >= before_page_number:
                continue
            payloads: list[dict[str, Any]] = []
            for row in batch:
                if (
                    row["analysis_status"] != "published"
                    or row["analysis_source_checksum"]
                    != row["current_source_checksum"]
                    or row["payload_json"] is None
                ):
                    payloads = []
                    break
                payload = _json_object(
                    row["payload_json"],
                    "active analysis page payload",
                )
                payload["page_number_snapshot"] = int(
                    row["page_number_snapshot"]
                )
                payloads.append(payload)
            if payloads:
                valid.append(payloads)
        return valid[-batch_count:]

    @staticmethod
    def publish_page_success(
        connection: Connection,
        *,
        run_id: str,
        scope: str,
        page_id: str,
        source_asset_id: str,
        source_checksum: str,
        page_number: int,
        payload: Mapping[str, Any],
    ) -> str:
        if scope not in ANALYSIS_RUN_SCOPES:
            raise InsightConflict("analysis run scope is invalid")
        try:
            canonical_payload = validate_persisted_page_analysis(payload)
        except InvalidPageAnalysis as exc:
            raise InsightConflict("analysis page payload is invalid") from exc
        if (
            canonical_payload["page_id"] != page_id
            or canonical_payload["source_asset_id"] != source_asset_id
            or canonical_payload["source_checksum"] != source_checksum
            or canonical_payload["page_number_snapshot"] != page_number
        ):
            raise InsightConflict("analysis page payload identity does not match")
        now = utcnow()
        existing = connection.execute(
            select(analysis_page_results.c.id).where(
                analysis_page_results.c.run_id == run_id,
                analysis_page_results.c.page_id_snapshot == page_id,
            )
        ).scalar_one_or_none()
        result_id = (
            _required_string(existing, "analysis page result id")
            if existing is not None
            else str(uuid.uuid4())
        )
        values = {
            "page_id": page_id,
            "source_asset_id": source_asset_id,
            "source_checksum": source_checksum,
            "page_id_snapshot": page_id,
            "page_number_snapshot": page_number,
            "payload_json": _json(canonical_payload),
            "schema_version": 2,
            "status": "staging" if scope == "full" else "published",
            "updated_at": now,
        }
        if existing is None:
            connection.execute(
                insert(analysis_page_results).values(
                    id=result_id,
                    run_id=run_id,
                    created_at=now,
                    **values,
                )
            )
        else:
            connection.execute(
                update(analysis_page_results)
                .where(analysis_page_results.c.id == result_id)
                .values(**values)
            )
        target_changed = connection.execute(
            update(analysis_run_targets)
            .where(
                analysis_run_targets.c.run_id == run_id,
                analysis_run_targets.c.page_id_snapshot == page_id,
            )
            .values(status="completed", error_json=None)
        )
        if target_changed.rowcount != 1:
            raise InsightConflict("analysis run target is missing")
        InsightRepository._refresh_run_counts(connection, run_id, now)
        if scope != "full":
            book_id = _required_string(
                connection.execute(
                select(analysis_runs.c.book_id).where(analysis_runs.c.id == run_id)
                ).scalar_one(),
                "analysis run book id",
            )
            InsightRepository._upsert_page_head(
                connection,
                book_id=book_id,
                page_id=page_id,
                run_id=run_id,
                result_id=result_id,
                now=now,
            )
            mark_book_insight_derived_stale(
                connection,
                book_id=book_id,
                now=now,
            )
        return result_id

    @staticmethod
    def copy_page_successes(
        connection: Connection,
        *,
        run_id: str,
        scope: str,
        copies: Sequence[Mapping[str, Any]],
    ) -> None:
        """Copy immutable successes into a retry run with one count refresh."""

        if not copies:
            return
        if scope not in ANALYSIS_RUN_SCOPES:
            raise InsightConflict("analysis run scope is invalid")
        now = utcnow()
        prepared: list[dict[str, Any]] = []
        result_ids_by_page: dict[str, str] = {}
        for index, copy in enumerate(copies, start=1):
            if set(copy) != {
                "page_id",
                "source_asset_id",
                "source_checksum",
                "page_number",
                "payload",
            }:
                raise InsightConflict(f"analysis retry copy {index} fields are invalid")
            page_id = _required_string(
                copy["page_id"],
                f"analysis retry copy {index} page id",
            )
            if page_id in result_ids_by_page:
                raise InsightConflict("analysis retry copy page ids must be unique")
            source_asset_id = _required_string(
                copy["source_asset_id"],
                f"analysis retry copy {index} source asset id",
            )
            source_checksum = _required_sha256(
                copy["source_checksum"],
                f"analysis retry copy {index} source checksum",
            )
            page_number = _required_integer(
                copy["page_number"],
                f"analysis retry copy {index} page number",
                minimum=1,
            )
            try:
                payload = validate_persisted_page_analysis(copy["payload"])
            except InvalidPageAnalysis as exc:
                raise InsightConflict(
                    f"analysis retry copy {index} payload is invalid"
                ) from exc
            if (
                payload["page_id"] != page_id
                or payload["source_asset_id"] != source_asset_id
                or payload["source_checksum"] != source_checksum
                or payload["page_number_snapshot"] != page_number
            ):
                raise InsightConflict(
                    f"analysis retry copy {index} identity does not match"
                )
            result_id = str(uuid.uuid4())
            result_ids_by_page[page_id] = result_id
            prepared.append(
                {
                    "id": result_id,
                    "run_id": run_id,
                    "page_id": page_id,
                    "source_asset_id": source_asset_id,
                    "source_checksum": source_checksum,
                    "page_id_snapshot": page_id,
                    "page_number_snapshot": page_number,
                    "payload_json": _json(payload),
                    "schema_version": 2,
                    "status": "staging" if scope == "full" else "published",
                    "created_at": now,
                    "updated_at": now,
                }
            )
        connection.execute(insert(analysis_page_results), prepared)
        page_ids = tuple(result_ids_by_page)
        targets_changed = connection.execute(
            update(analysis_run_targets)
            .where(
                analysis_run_targets.c.run_id == run_id,
                analysis_run_targets.c.page_id_snapshot.in_(page_ids),
            )
            .values(status="completed", error_json=None)
        )
        if targets_changed.rowcount != len(page_ids):
            raise InsightConflict("analysis retry targets are incomplete")
        InsightRepository._refresh_run_counts(connection, run_id, now)
        if scope == "full":
            return

        book_id = _required_string(
            connection.execute(
                select(analysis_runs.c.book_id).where(analysis_runs.c.id == run_id)
            ).scalar_one(),
            "analysis run book id",
        )
        for page_id, result_id in result_ids_by_page.items():
            InsightRepository._upsert_page_head(
                connection,
                book_id=book_id,
                page_id=page_id,
                run_id=run_id,
                result_id=result_id,
                now=now,
            )
        mark_book_insight_derived_stale(
            connection,
            book_id=book_id,
            now=now,
        )

    @staticmethod
    def publish_page_failure(
        connection: Connection,
        *,
        run_id: str,
        page_id: str,
        code: str,
        message: str,
    ) -> None:
        now = utcnow()
        changed = connection.execute(
            update(analysis_run_targets)
            .where(
                analysis_run_targets.c.run_id == run_id,
                analysis_run_targets.c.page_id_snapshot == page_id,
                analysis_run_targets.c.status != "completed",
            )
            .values(
                status="failed",
                error_json=_json(
                    {
                        "code": code,
                        "message": redact_sensitive_text(message),
                    }
                ),
            )
        )
        if changed.rowcount:
            InsightRepository._refresh_run_counts(connection, run_id, now)

    @staticmethod
    def validate_run_sources(
        connection: Connection,
        *,
        run_id: str,
    ) -> dict[str, Any]:
        """Fence completed page results against the current immutable source."""

        now = utcnow()
        targets = list(
            connection.execute(
                select(analysis_run_targets)
                .where(analysis_run_targets.c.run_id == run_id)
                .order_by(analysis_run_targets.c.ordinal)
            ).mappings()
        )
        if not targets:
            raise InsightNotFound("analysis run not found")
        target_statuses: list[str] = []
        seen_target_page_ids: set[str] = set()
        for index, target in enumerate(targets, start=1):
            status = _required_string(
                target["status"],
                f"analysis run target {index} status",
            )
            if status not in ANALYSIS_TARGET_STATUSES:
                raise InsightConflict(
                    "stored analysis run target status is invalid; "
                    "clear current Insight data"
                )
            ordinal = _required_integer(
                target["ordinal"],
                f"analysis run target {index} ordinal",
                minimum=1,
            )
            if ordinal != index:
                raise InsightConflict(
                    "stored analysis run target order is invalid; "
                    "clear current Insight data"
                )
            target_page_id = _required_string(
                target["page_id_snapshot"],
                f"analysis run target {index} page id snapshot",
            )
            if target_page_id in seen_target_page_ids:
                raise InsightConflict(
                    "stored analysis run target pages are duplicated; "
                    "clear current Insight data"
                )
            seen_target_page_ids.add(target_page_id)
            _required_integer(
                target["page_number_snapshot"],
                f"analysis run target {index} page number snapshot",
                minimum=1,
            )
            _required_string(
                target["source_asset_id"],
                f"analysis run target {index} source asset id",
            )
            _required_sha256(
                target["source_checksum"],
                f"analysis run target {index} source checksum",
            )
            target_statuses.append(status)
        if "pending" in target_statuses:
            raise InsightConflict("analysis run still has pending targets")
        completed_targets = [
            target
            for target, status in zip(targets, target_statuses)
            if status == "completed"
        ]
        current_by_page: dict[str, Mapping[str, Any]] = {}
        current_page_ids = []
        for target in completed_targets:
            current_page_id = _optional_string(
                target["page_id"],
                "analysis run target current page id",
            )
            snapshot_page_id = _required_string(
                target["page_id_snapshot"],
                "analysis run target page id snapshot",
            )
            if current_page_id is not None and current_page_id != snapshot_page_id:
                raise InsightConflict(
                    "stored analysis target current page is invalid; "
                    "clear current Insight data"
                )
            if current_page_id is not None:
                current_page_ids.append(current_page_id)
        if current_page_ids:
            for row in connection.execute(
                    select(
                        page_assets.c.page_id,
                        page_assets.c.asset_id,
                        assets.c.checksum,
                    )
                    .join(assets, assets.c.id == page_assets.c.asset_id)
                    .where(
                        page_assets.c.page_id.in_(current_page_ids),
                        page_assets.c.role == "source",
                    )
                ).mappings():
                current_page_id = _required_string(
                    row["page_id"],
                    "current analysis page id",
                )
                if current_page_id in current_by_page:
                    raise InsightConflict(
                        "current analysis page source is duplicated; "
                        "clear current Insight data"
                    )
                _required_string(
                    row["asset_id"],
                    "current analysis source asset id",
                )
                _required_sha256(
                    row["checksum"],
                    "current analysis source checksum",
                )
                current_by_page[current_page_id] = row
        changed: list[str] = []
        for target in completed_targets:
            current_page_id = _optional_string(
                target["page_id"],
                "analysis run target current page id",
            )
            current = (
                current_by_page.get(current_page_id)
                if current_page_id is not None
                else None
            )
            if (
                current is not None
                and current["asset_id"] == target["source_asset_id"]
                and current["checksum"] == target["source_checksum"]
            ):
                continue
            changed.append(
                _required_string(
                    target["page_id_snapshot"],
                    "analysis run target page id snapshot",
                )
            )
        if changed:
            connection.execute(
                update(analysis_run_targets)
                .where(
                    analysis_run_targets.c.run_id == run_id,
                    analysis_run_targets.c.page_id_snapshot.in_(changed),
                    analysis_run_targets.c.status == "completed",
                )
                .values(
                    status="conflict",
                    error_json=_json(
                        {
                            "code": "SOURCE_CHANGED",
                            "message": "page source changed before publication",
                        }
                    ),
                )
            )
        InsightRepository._refresh_run_counts(connection, run_id, now)
        counts: dict[str, int] = {}
        for status_value, count_value in connection.execute(
                select(analysis_run_targets.c.status, func.count())
                .where(analysis_run_targets.c.run_id == run_id)
                .group_by(analysis_run_targets.c.status)
            ):
            status = _required_string(status_value, "analysis run target count status")
            if status not in ANALYSIS_TARGET_STATUSES:
                raise InsightConflict(
                    "stored analysis run target status is invalid; "
                    "clear current Insight data"
                )
            counts[status] = _required_integer(
                count_value,
                "analysis run target status count",
            )
        return {
            "runId": run_id,
            "successCount": counts.get("completed", 0),
            "failedCount": counts.get("failed", 0)
            + counts.get("conflict", 0),
            "sourceConflictPageIds": changed,
        }

    @staticmethod
    def finalize_run(
        connection: Connection,
        *,
        run_id: str,
    ) -> dict[str, Any]:
        now = utcnow()
        run = connection.execute(
            select(analysis_runs).where(analysis_runs.c.id == run_id)
        ).mappings().one_or_none()
        if run is None:
            raise InsightNotFound("analysis run not found")
        scope = _required_string(run["scope"], "analysis run scope")
        if scope not in ANALYSIS_RUN_SCOPES:
            raise InsightConflict(
                "stored analysis run scope is invalid; clear current Insight data"
            )
        run_status = _required_string(run["status"], "analysis run status")
        if run_status != "staging":
            raise InsightConflict("analysis run is not staging")
        if _required_integer(
            run["schema_version"],
            "analysis run schema version",
            minimum=1,
        ) != 2:
            raise InsightConflict(
                "stored analysis run schema is obsolete; "
                "clear current Insight data"
            )
        book_id = _required_string(run["book_id"], "analysis run book id")
        InsightRepository.validate_run_sources(connection, run_id=run_id)

        refreshed = list(
            connection.execute(
                select(analysis_run_targets)
                .where(analysis_run_targets.c.run_id == run_id)
                .order_by(analysis_run_targets.c.ordinal)
            ).mappings()
        )
        if _required_integer(
            run["target_count"],
            "analysis run target count",
        ) != len(refreshed):
            raise InsightConflict(
                "stored analysis run target count is inconsistent; "
                "clear current Insight data"
            )
        successful: list[Mapping[str, Any]] = []
        missing: list[str] = []
        for target in refreshed:
            status = _required_string(target["status"], "analysis run target status")
            if status not in ANALYSIS_TARGET_STATUSES:
                raise InsightConflict(
                    "stored analysis run target status is invalid; "
                    "clear current Insight data"
                )
            page_id = _required_string(
                target["page_id_snapshot"],
                "analysis run target page id snapshot",
            )
            if status == "completed":
                successful.append(target)
            else:
                missing.append(page_id)
        success_count = len(successful)
        failed_count = len(refreshed) - success_count
        if success_count == 0:
            raise InsightConflict("analysis run has no publishable page results")

        final_status = "completed_with_errors" if failed_count else "completed"
        if scope == "full":
            config = _json_object(run["config_json"], "analysis run config")
            analysis_config = config.get("analysis")
            if not isinstance(analysis_config, Mapping):
                raise InsightConflict(
                    "stored analysis run config is invalid; clear current Insight data"
                )
            layers = analysis_config.get("layers")
            if not isinstance(layers, list):
                raise InsightConflict(
                    "stored analysis layer config is invalid; clear current Insight data"
                )
            expected_layers: dict[int, str] = {}
            for index, layer in enumerate(layers, start=1):
                if not isinstance(layer, Mapping):
                    raise InsightConflict(
                        "stored analysis layer config is invalid; "
                        "clear current Insight data"
                    )
                layer_index = _required_integer(
                    layer.get("index"),
                    f"analysis layer config {index} index",
                )
                if layer_index != index - 1:
                    raise InsightConflict(
                        "stored analysis layer order is invalid; "
                        "clear current Insight data"
                    )
                layer_name = _required_string(
                    layer.get("name"),
                    f"analysis layer config {index} name",
                )
                if not layer_name.strip():
                    raise InsightConflict(
                        "stored analysis layer name is blank; "
                        "clear current Insight data"
                    )
                if layer_index in expected_layers:
                    raise InsightConflict(
                        "stored analysis layer indices are duplicated; "
                        "clear current Insight data"
                    )
                expected_layers[layer_index] = layer_name
            staged_layer_rows = list(
                connection.execute(
                    select(analysis_layer_results)
                    .where(analysis_layer_results.c.run_id == run_id)
                    .order_by(
                        analysis_layer_results.c.layer_index,
                        analysis_layer_results.c.unit_index,
                    )
                ).mappings()
            )
            layer_units: dict[int, set[int]] = {}
            layer_zero_event_count = 0
            for row in staged_layer_rows:
                result_id = _required_string(
                    row["id"],
                    "analysis layer result id",
                )
                if _required_string(
                    row["run_id"],
                    "analysis layer result run id",
                ) != run_id or _required_string(
                    row["status"],
                    "analysis layer result status",
                ) != "staging":
                    raise InsightConflict(
                        "full analysis layer result is not staging"
                    )
                layer_index = _required_integer(
                    row["layer_index"],
                    "analysis layer result index",
                )
                unit_index = _required_integer(
                    row["unit_index"],
                    "analysis layer result unit index",
                )
                expected_name = expected_layers.get(layer_index)
                if expected_name is None or _required_string(
                    row["layer_name"],
                    "analysis layer result name",
                ) != expected_name:
                    raise InsightConflict(
                        "analysis layer result identity is invalid"
                    )
                units = layer_units.setdefault(layer_index, set())
                if unit_index in units:
                    raise InsightConflict(
                        "analysis layer result units are duplicated"
                    )
                units.add(unit_index)
                page_range = _json_object(
                    row["page_range_snapshot_json"],
                    "analysis layer page range",
                )
                if set(page_range) != {"start", "end"}:
                    raise InsightConflict(
                        "analysis layer page range is invalid"
                    )
                range_start = _required_integer(
                    page_range["start"],
                    "analysis layer page range start",
                    minimum=1,
                )
                _required_integer(
                    page_range["end"],
                    "analysis layer page range end",
                    minimum=range_start,
                )
                content = _json_object(
                    row["content_json"],
                    "analysis layer content",
                )
                if not contains_nonempty_text(content):
                    raise InsightConflict("analysis layer content is empty")
                if layer_index == 0:
                    key_events = content.get("key_events", [])
                    if not isinstance(key_events, list):
                        raise InsightConflict(
                            "analysis layer key_events must be an array"
                        )
                    layer_zero_event_count += len(key_events)
                _required_sha256(
                    row["input_fingerprint"],
                    f"analysis layer result {result_id} input fingerprint",
                )
            if set(layer_units) != set(expected_layers) or any(
                units != set(range(len(units)))
                for units in layer_units.values()
            ):
                raise InsightConflict(
                    "full analysis run is missing required summary layers"
                )
            required_artifacts = {
                ("compressed_context", "default"),
                ("overview", "no_spoiler"),
                ("overview", "story_summary"),
            }
            staged_artifacts = list(
                connection.execute(
                    select(analysis_artifacts).where(
                        analysis_artifacts.c.run_id == run_id,
                    )
                ).mappings()
            )
            staged_artifact_keys: set[tuple[str, str]] = set()
            dependency_fingerprints: set[str] = set()
            for row in staged_artifacts:
                artifact_id = _required_string(
                    row["id"],
                    "analysis artifact id",
                )
                key = (
                    _required_string(row["kind"], "analysis artifact kind"),
                    _required_string(
                        row["template"],
                        "analysis artifact template",
                    ),
                )
                if key in staged_artifact_keys:
                    raise InsightConflict(
                        "staged analysis artifacts are duplicated; "
                        "clear current Insight data"
                    )
                staged_artifact_keys.add(key)
                if (
                    _required_string(
                        row["book_id"],
                        "analysis artifact book id",
                    )
                    != book_id
                    or _required_string(
                        row["run_id"],
                        "analysis artifact run id",
                    )
                    != run_id
                    or _required_string(
                        row["status"],
                        "analysis artifact status",
                    )
                    != "building"
                    or _required_boolean(
                        row["is_active"],
                        "analysis artifact active flag",
                    )
                ):
                    raise InsightConflict(
                        "staged analysis artifact identity is invalid"
                    )
                _required_integer(
                    row["revision"],
                    "analysis artifact revision",
                    minimum=1,
                )
                dependency_fingerprints.add(
                    _required_sha256(
                        row["dependency_fingerprint"],
                        "analysis artifact dependency fingerprint",
                    )
                )
                if row["asset_id"] is not None:
                    raise InsightConflict(
                        "staged analysis artifact has an unexpected asset"
                    )
                payload = _json_object(
                    row["payload_json"],
                    f"analysis artifact {artifact_id} payload",
                )
                if not contains_nonempty_text(payload):
                    raise InsightConflict("analysis artifact payload is empty")
                if key[0] == "overview" and (
                    not isinstance(payload.get("title"), str)
                    or not payload["title"].strip()
                    or not isinstance(payload.get("content"), str)
                    or not payload["content"].strip()
                ):
                    raise InsightConflict("overview artifact payload is invalid")
            if staged_artifact_keys != required_artifacts:
                raise InsightConflict(
                    "full analysis run is missing required overview artifacts"
                )
            staged_timelines = list(
                connection.execute(
                    select(timeline_versions).where(
                        timeline_versions.c.run_id == run_id
                    )
                ).mappings()
            )
            staged_vectors = list(
                connection.execute(
                    select(vector_generations).where(
                        vector_generations.c.run_id == run_id
                    )
                ).mappings()
            )
            if len(staged_timelines) != 1 or len(staged_vectors) != 1:
                raise InsightConflict(
                    "full analysis run is missing timeline or vector generation"
                )
            staged_timeline = staged_timelines[0]
            staged_vector = staged_vectors[0]
            staged_timeline_id = _required_string(
                staged_timeline["id"],
                "staged timeline id",
            )
            staged_vector_id = _required_string(
                staged_vector["id"],
                "staged vector generation id",
            )
            timeline_mode = _required_string(
                staged_timeline["mode"],
                "staged timeline mode",
            )
            if timeline_mode not in {"enhanced", "compressed", "simple"}:
                raise InsightConflict("staged timeline mode is invalid")
            if (
                _required_string(
                    staged_timeline["book_id"],
                    "staged timeline book id",
                )
                != book_id
                or _required_string(
                    staged_timeline["run_id"],
                    "staged timeline run id",
                )
                != run_id
                or _required_string(
                    staged_timeline["status"],
                    "staged timeline status",
                )
                != "building"
                or _required_boolean(
                    staged_timeline["is_active"],
                    "staged timeline active flag",
                )
            ):
                raise InsightConflict("staged timeline identity is invalid")
            dependency_fingerprints.add(
                _required_sha256(
                    staged_timeline["dependency_fingerprint"],
                    "staged timeline dependency fingerprint",
                )
            )
            timeline_content = _json_object(
                staged_timeline["content_json"],
                "staged timeline content",
            )
            story_summary = timeline_content.get("story_summary")
            fallback_reason = timeline_content.get("fallback_reason")
            if (
                not isinstance(story_summary, str)
                or (timeline_mode != "simple" and not story_summary.strip())
                or timeline_content.get("requested_mode") != "enhanced"
                or timeline_content.get("actual_mode") != timeline_mode
                or _required_boolean(
                    timeline_content.get("degraded"),
                    "staged timeline degraded flag",
                )
                != (timeline_mode != "enhanced")
                or (
                    timeline_mode == "enhanced"
                    and fallback_reason is not None
                )
                or (
                    timeline_mode != "enhanced"
                    and (
                        not isinstance(fallback_reason, str)
                        or not fallback_reason.strip()
                    )
                )
            ):
                raise InsightConflict("staged timeline content is invalid")
            _required_integer(
                connection.execute(
                    select(func.count(timeline_events.c.id)).where(
                        timeline_events.c.timeline_version_id
                        == staged_timeline_id
                    )
                ).scalar_one(),
                "staged timeline event count",
                minimum=1,
            )
            if (
                _required_string(
                    staged_vector["book_id"],
                    "staged vector book id",
                )
                != book_id
                or _required_string(
                    staged_vector["run_id"],
                    "staged vector run id",
                )
                != run_id
                or _required_string(
                    staged_vector["status"],
                    "staged vector status",
                )
                != "building"
                or _required_boolean(
                    staged_vector["is_active"],
                    "staged vector active flag",
                )
            ):
                raise InsightConflict("staged vector identity is invalid")
            _required_integer(
                staged_vector["generation"],
                "staged vector generation",
                minimum=1,
            )
            dependency_fingerprints.add(
                _required_sha256(
                    staged_vector["dependency_fingerprint"],
                    "staged vector dependency fingerprint",
                )
            )
            if (
                _required_integer(
                    staged_vector["page_count"],
                    "staged vector page count",
                )
                != success_count
                or _required_integer(
                    staged_vector["event_count"],
                    "staged vector event count",
                )
                != layer_zero_event_count
                or len(dependency_fingerprints) != 1
            ):
                raise InsightConflict(
                    "staged derived generations are inconsistent"
                )
            targets_by_page = {
                _required_string(
                    target["page_id_snapshot"],
                    "analysis target page id",
                ): target
                for target in refreshed
            }
            result_rows: dict[str, Mapping[str, Any]] = {}
            for row in connection.execute(
                    select(analysis_page_results).where(
                        analysis_page_results.c.run_id == run_id
                    )
                ).mappings():
                result_page_id = _required_string(
                    row["page_id_snapshot"],
                    "analysis page result page id",
                )
                result_id = _required_string(row["id"], "analysis page result id")
                if result_page_id in result_rows:
                    raise InsightConflict(
                        "analysis page results are duplicated; clear current Insight data"
                    )
                target = targets_by_page.get(result_page_id)
                if target is None or target["status"] not in {
                    "completed",
                    "conflict",
                }:
                    raise InsightConflict(
                        "analysis run contains an unexpected page result"
                    )
                result_status = _required_string(
                    row["status"],
                    "analysis page result status",
                )
                if result_status != "staging":
                    raise InsightConflict(
                        "full analysis page result is not staging"
                    )
                if _required_integer(
                    row["schema_version"],
                    "analysis page result schema version",
                    minimum=1,
                ) != 2:
                    raise InsightConflict(
                        "stored page analysis schema is obsolete; "
                        "clear current Insight data"
                    )
                source_asset_id = _required_string(
                    row["source_asset_id"],
                    "analysis page result source asset id",
                )
                source_checksum = _required_sha256(
                    row["source_checksum"],
                    "analysis page result source checksum",
                )
                page_number = _required_integer(
                    row["page_number_snapshot"],
                    "analysis page result page number",
                    minimum=1,
                )
                if (
                    _required_string(
                        row["run_id"],
                        "analysis page result run id",
                    )
                    != run_id
                    or source_asset_id != target["source_asset_id"]
                    or source_checksum != target["source_checksum"]
                    or page_number != target["page_number_snapshot"]
                    or _optional_string(
                        row["page_id"],
                        "analysis page result current page id",
                    )
                    != _optional_string(
                        target["page_id"],
                        "analysis target current page id",
                    )
                ):
                    raise InsightConflict(
                        "analysis page result identity is inconsistent; "
                        "clear current Insight data"
                    )
                _page_analysis(
                    row["payload_json"],
                    "analysis page payload",
                    page_id=result_page_id,
                    page_number=page_number,
                    source_asset_id=source_asset_id,
                    source_checksum=source_checksum,
                )
                result_rows[result_page_id] = {**row, "id": result_id}
            expected_result_pages = {
                page_id
                for page_id, target in targets_by_page.items()
                if target["status"] in {"completed", "conflict"}
            }
            if set(result_rows) != expected_result_pages:
                raise InsightConflict(
                    "analysis run page results are incomplete"
                )
            for target in successful:
                page_id = _required_string(
                    target["page_id_snapshot"],
                    "successful analysis target page id",
                )
                result = result_rows.get(page_id)
                if result is None:
                    raise InsightConflict(
                        f"analysis result missing for successful page {page_id}"
                    )
                published = connection.execute(
                    update(analysis_page_results)
                    .where(
                        analysis_page_results.c.id == result["id"],
                        analysis_page_results.c.status == "staging",
                    )
                    .values(status="published", updated_at=now)
                )
                if published.rowcount != 1:
                    raise InsightConflict(
                        "analysis page result publication was fenced"
                    )
                InsightRepository._upsert_page_head(
                    connection,
                    book_id=book_id,
                    page_id=page_id,
                    run_id=run_id,
                    result_id=_required_string(
                        result["id"],
                        "analysis page result id",
                    ),
                    now=now,
                )
            InsightRepository._upsert_book_head(
                connection,
                book_id=book_id,
                run_id=run_id,
                now=now,
            )
            derived_status = (
                "degraded"
                if final_status == "completed_with_errors"
                else "ready"
            )
            connection.execute(
                update(analysis_layer_results)
                .where(analysis_layer_results.c.run_id == run_id)
                .values(status="published", updated_at=now)
            )
            connection.execute(
                update(analysis_artifacts)
                .where(
                    analysis_artifacts.c.book_id == book_id,
                    analysis_artifacts.c.is_active.is_(True),
                )
                .values(is_active=False, updated_at=now)
            )
            connection.execute(
                update(analysis_artifacts)
                .where(
                    analysis_artifacts.c.run_id == run_id,
                    analysis_artifacts.c.status == "building",
                )
                .values(
                    status=derived_status,
                    is_active=True,
                    updated_at=now,
                )
            )
            connection.execute(
                update(timeline_versions)
                .where(
                    timeline_versions.c.book_id == book_id,
                    timeline_versions.c.is_active.is_(True),
                )
                .values(is_active=False, updated_at=now)
            )
            connection.execute(
                update(timeline_versions)
                .where(timeline_versions.c.id == staged_timeline_id)
                .values(
                    status=derived_status,
                    is_active=True,
                    updated_at=now,
                )
            )
            connection.execute(
                update(vector_generations)
                .where(
                    vector_generations.c.book_id == book_id,
                    vector_generations.c.is_active.is_(True),
                )
                .values(is_active=False, updated_at=now)
            )
            connection.execute(
                update(vector_generations)
                .where(vector_generations.c.id == staged_vector_id)
                .values(
                    status=derived_status,
                    is_active=True,
                    updated_at=now,
                )
            )

        connection.execute(
            update(analysis_runs)
            .where(analysis_runs.c.id == run_id)
            .values(
                status=final_status,
                missing_page_ids_json=_json(missing),
                success_count=success_count,
                failed_count=failed_count,
                published_at=now,
                updated_at=now,
            )
        )
        if scope != "full":
            mark_book_insight_derived_stale(
                connection,
                book_id=book_id,
                now=now,
            )
        return {
            "runId": run_id,
            "status": final_status,
            "successCount": success_count,
            "failedCount": failed_count,
            "missingPageIds": missing,
        }

    @staticmethod
    def mark_run_failed(
        connection: Connection,
        *,
        run_id: str,
    ) -> None:
        now = utcnow()
        run_status = connection.execute(
            select(analysis_runs.c.status).where(analysis_runs.c.id == run_id)
        ).scalar_one_or_none()
        if run_status is None:
            raise InsightNotFound("analysis run not found")
        current_status = _required_string(run_status, "analysis run status")
        if current_status == "failed":
            return
        if current_status != "staging":
            raise InsightConflict("analysis run is no longer staging")
        targets = list(
            connection.execute(
                select(
                    analysis_run_targets.c.page_id_snapshot,
                    analysis_run_targets.c.status,
                )
                .where(analysis_run_targets.c.run_id == run_id)
                .order_by(analysis_run_targets.c.ordinal)
            ).mappings()
        )
        success_count = 0
        missing: list[str] = []
        for target in targets:
            page_id = _required_string(
                target["page_id_snapshot"],
                "analysis run missing page id",
            )
            status = _required_string(
                target["status"],
                "analysis run target status",
            )
            if status not in ANALYSIS_TARGET_STATUSES:
                raise InsightConflict(
                    "stored analysis run target status is invalid; "
                    "clear current Insight data"
                )
            if status == "completed":
                success_count += 1
            else:
                missing.append(page_id)
        changed = connection.execute(
            update(analysis_runs)
            .where(
                analysis_runs.c.id == run_id,
                analysis_runs.c.status == "staging",
            )
            .values(
                status="failed",
                success_count=success_count,
                failed_count=len(targets) - success_count,
                updated_at=now,
                missing_page_ids_json=_json(missing),
            )
        )
        if changed.rowcount != 1:
            raise InsightNotFound("analysis run not found")

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            run = connection.execute(
                select(analysis_runs).where(analysis_runs.c.id == run_id)
            ).mappings().one_or_none()
            targets = list(
                connection.execute(
                    select(analysis_run_targets)
                    .where(analysis_run_targets.c.run_id == run_id)
                    .order_by(analysis_run_targets.c.ordinal)
                ).mappings()
            )
        if run is None:
            raise InsightNotFound("analysis run not found")
        scope = _required_string(run["scope"], "analysis run scope")
        if scope not in ANALYSIS_RUN_SCOPES:
            raise InsightConflict(
                "stored analysis run scope is invalid; clear current Insight data"
            )
        status = _required_string(run["status"], "analysis run status")
        if status not in ANALYSIS_RUN_STATUSES:
            raise InsightConflict(
                "stored analysis run status is invalid; clear current Insight data"
            )
        missing_page_ids = _json_array(
            run["missing_page_ids_json"],
            "analysis run missing page ids",
        )
        if any(
            not isinstance(page_id, str) or not page_id
            for page_id in missing_page_ids
        ) or len(set(missing_page_ids)) != len(missing_page_ids):
            raise InsightConflict(
                "stored analysis run missing page ids are invalid; "
                "clear current Insight data"
            )
        target_items = []
        completed_count = 0
        failed_target_count = 0
        seen_target_page_ids: set[str] = set()
        for index, row in enumerate(targets, start=1):
            ordinal = _required_integer(
                row["ordinal"],
                f"analysis run target {index} ordinal",
                minimum=1,
            )
            if ordinal != index:
                raise InsightConflict(
                    "stored analysis run target order is invalid; "
                    "clear current Insight data"
                )
            target_status = _required_string(
                row["status"],
                f"analysis run target {index} status",
            )
            if target_status not in ANALYSIS_TARGET_STATUSES:
                raise InsightConflict(
                    "stored analysis run target status is invalid; "
                    "clear current Insight data"
                )
            error = _optional_json_object(
                row["error_json"],
                "analysis target error",
            )
            if target_status in {"failed", "conflict"}:
                if (
                    error is None
                    or set(error) != {"code", "message"}
                    or any(
                        not isinstance(error[field], str) or not error[field]
                        for field in ("code", "message")
                    )
                ):
                    raise InsightConflict(
                        "stored analysis target error is invalid; "
                        "clear current Insight data"
                    )
                failed_target_count += 1
            elif error is not None:
                raise InsightConflict(
                    "stored analysis target error is inconsistent; "
                    "clear current Insight data"
                )
            if target_status == "completed":
                completed_count += 1
            target_page_id = _required_string(
                row["page_id_snapshot"],
                f"analysis run target {index} page id",
            )
            if target_page_id in seen_target_page_ids:
                raise InsightConflict(
                    "stored analysis run target pages are duplicated; "
                    "clear current Insight data"
                )
            seen_target_page_ids.add(target_page_id)
            target_items.append(
                {
                    "pageId": target_page_id,
                    "pageNumber": _required_integer(
                        row["page_number_snapshot"],
                        f"analysis run target {index} page number",
                        minimum=1,
                    ),
                    "status": target_status,
                    "error": error,
                }
            )
        target_count = _required_integer(
            run["target_count"],
            "analysis run target count",
        )
        success_count = _required_integer(
            run["success_count"],
            "analysis run success count",
        )
        stored_failed_count = _required_integer(
            run["failed_count"],
            "analysis run failed count",
        )
        expected_failed_count = (
            failed_target_count
            if status == "staging"
            else len(target_items) - completed_count
        )
        if (
            target_count != len(target_items)
            or success_count != completed_count
            or stored_failed_count != expected_failed_count
        ):
            raise InsightConflict(
                "stored analysis run counts are inconsistent; clear current Insight data"
            )
        if status != "staging":
            expected_missing = {
                item["pageId"]
                for item in target_items
                if item["status"] != "completed"
            }
            if set(missing_page_ids) != expected_missing:
                raise InsightConflict(
                    "stored analysis run missing pages are inconsistent; "
                    "clear current Insight data"
                )
        return {
            "runId": _required_string(run["id"], "analysis run id"),
            "jobId": _optional_string(run["job_id"], "analysis run job id"),
            "bookId": _required_string(run["book_id"], "analysis run book id"),
            "scope": scope,
            "status": status,
            "targetCount": target_count,
            "successCount": success_count,
            "failedCount": stored_failed_count,
            "missingPageIds": missing_page_ids,
            "createdAt": _iso(
                _required_datetime(run["created_at"], "analysis run createdAt")
            ),
            "publishedAt": _iso(
                _optional_datetime(run["published_at"], "analysis run publishedAt")
            ),
            "targets": target_items,
        }

    def bootstrap(self) -> dict[str, Any]:
        with self.engine.connect() as connection:
            book_rows = list(
                connection.execute(
                    select(
                        books.c.id,
                        books.c.title,
                        books.c.cover_asset_id,
                        books.c.updated_at,
                    )
                    .where(
                        books.c.kind == "library",
                        books.c.owner_user_id == effective_owner_id(),
                    )
                    .order_by(books.c.updated_at.desc(), books.c.title)
                ).mappings()
            )
            page_count_rows = connection.execute(
                select(chapters.c.book_id, func.count(pages.c.id))
                .join(pages, pages.c.chapter_id == chapters.c.id)
                .group_by(chapters.c.book_id)
            ).tuples().all()
            head_count_rows = connection.execute(
                select(analysis_heads.c.book_id, func.count())
                .where(analysis_heads.c.page_id.is_not(None))
                .group_by(analysis_heads.c.book_id)
            ).tuples().all()
            active_run_rows = list(
                connection.execute(
                    select(
                        analysis_heads.c.book_id,
                        analysis_runs.c.id,
                        analysis_runs.c.status,
                        analysis_runs.c.published_at,
                    )
                    .join(
                        analysis_runs,
                        analysis_runs.c.id == analysis_heads.c.active_run_id,
                    )
                    .where(analysis_heads.c.page_id.is_(None))
                ).mappings()
            )
            active_job_rows = list(
                connection.execute(
                    select(
                        jobs.c.id,
                        jobs.c.book_id,
                        jobs.c.kind,
                        jobs.c.status,
                        jobs.c.latest_progress_json,
                    ).where(
                        jobs.c.kind.in_(
                            (
                                "insight_analysis",
                                "insight_export",
                                "vector_rebuild",
                                "continuation",
                                "derived_rebuild",
                            )
                        ),
                        jobs.c.status.in_(NONTERMINAL_JOB_STATUSES),
                        jobs.c.owner_user_id == effective_owner_id(),
                    )
                ).mappings()
            )

        page_counts: dict[str, int] = {}
        for raw_book_id, raw_count in page_count_rows:
            count_book_id = _required_string(raw_book_id, "page count book id")
            if count_book_id in page_counts:
                raise InsightConflict("stored page counts are duplicated")
            page_counts[count_book_id] = _required_integer(
                raw_count,
                "page count",
            )
        head_counts: dict[str, int] = {}
        for raw_book_id, raw_count in head_count_rows:
            count_book_id = _required_string(
                raw_book_id,
                "analysis head count book id",
            )
            if count_book_id in head_counts:
                raise InsightConflict("stored analysis head counts are duplicated")
            head_counts[count_book_id] = _required_integer(
                raw_count,
                "analysis head count",
            )
        active_runs: dict[str, Mapping[str, Any]] = {}
        for row in active_run_rows:
            active_book_id = _required_string(
                row["book_id"],
                "active analysis run book id",
            )
            if active_book_id in active_runs:
                raise InsightConflict("stored active analysis runs are duplicated")
            active_status = _required_string(
                row["status"],
                "active analysis run status",
            )
            if active_status not in {"completed", "completed_with_errors"}:
                raise InsightConflict(
                    "stored active analysis run status is invalid; "
                    "clear current Insight data"
                )
            _required_string(row["id"], "active analysis run id")
            _required_datetime(
                row["published_at"],
                "active analysis run publishedAt",
            )
            active_runs[active_book_id] = row

        active_jobs = []
        for row in active_job_rows:
            kind = _required_string(row["kind"], "active job kind")
            if kind not in {
                "insight_analysis",
                "insight_export",
                "vector_rebuild",
                "continuation",
                "derived_rebuild",
            }:
                raise InsightConflict("stored active Insight job kind is invalid")
            status = _required_string(row["status"], "active job status")
            if status not in NONTERMINAL_JOB_STATUSES:
                raise InsightConflict("stored active Insight job status is invalid")
            active_jobs.append(
                {
                    "jobId": _required_string(row["id"], "active job id"),
                    "bookId": _required_string(
                        row["book_id"],
                        "active job book id",
                    ),
                    "kind": kind,
                    "status": status,
                    "progress": decode_job_progress(row),
                }
            )

        items = []
        for row in book_rows:
            book_id = _required_string(row["id"], "Insight book id")
            active = active_runs.get(book_id)
            cover_asset_id = _optional_string(
                row["cover_asset_id"],
                "Insight book cover asset id",
            )
            items.append(
                {
                    "bookId": book_id,
                    "title": _required_string(row["title"], "Insight book title"),
                    "coverUrl": (
                        f"/api/v2/assets/{cover_asset_id}"
                        if cover_asset_id is not None
                        else None
                    ),
                    "pageCount": page_counts.get(book_id, 0),
                    "analyzedPageCount": head_counts.get(book_id, 0),
                    "activeRun": (
                        {
                            "runId": _required_string(
                                active["id"],
                                "active analysis run id",
                            ),
                            "status": _required_string(
                                active["status"],
                                "active analysis run status",
                            ),
                            "publishedAt": _iso(
                                _required_datetime(
                                    active["published_at"],
                                    "active analysis run publishedAt",
                                )
                            ),
                        }
                        if active is not None
                        else None
                    ),
                }
            )
        return {
            "books": items,
            "activeJobs": active_jobs,
            "qa": {"available": False, "reason": "select_book"},
        }

    def list_overview_templates(self, book_id: str) -> dict[str, Any]:
        """List active overview templates in one query.

        The overview panel only needs presence information for its selector;
        fetching every possible artifact separately turns an empty book into a
        burst of expected 404 responses.
        """
        with self.engine.connect() as connection:
            self._assert_book(connection, book_id)
            templates = list(
                connection.execute(
                    select(analysis_artifacts.c.template)
                    .where(
                        analysis_artifacts.c.book_id == book_id,
                        analysis_artifacts.c.kind == "overview",
                        analysis_artifacts.c.is_active.is_(True),
                    )
                    .order_by(analysis_artifacts.c.template)
                ).scalars()
            )
        items: list[str] = []
        for template_value in templates:
            template = _required_string(
                template_value,
                "overview artifact template",
            )
            if template not in OVERVIEW_TEMPLATES:
                raise InsightConflict(
                    "stored overview template is invalid; clear current Insight data"
                )
            if template in items:
                raise InsightConflict(
                    "stored overview templates are duplicated; "
                    "clear current Insight data"
                )
            items.append(template)
        return {"items": items}

    def list_recent_page_analyses(
        self,
        *,
        book_id: str,
        limit: int = 5,
    ) -> dict[str, Any]:
        if limit < 1 or limit > 20:
            raise ValueError("limit must be between 1 and 20")
        numbered_pages = self._numbered_book_pages_statement(book_id).subquery(
            "insight_recent_numbered_pages"
        )
        with self.engine.connect() as connection:
            self._assert_book(connection, book_id)
            rows = list(
                connection.execute(
                    select(
                        analysis_heads.c.page_id,
                        numbered_pages.c.page_number,
                        analysis_page_results.c.page_id.label(
                            "result_page_id"
                        ),
                        analysis_page_results.c.page_id_snapshot,
                        analysis_page_results.c.page_number_snapshot,
                        analysis_page_results.c.source_asset_id,
                        analysis_page_results.c.source_checksum,
                        analysis_page_results.c.schema_version,
                        analysis_page_results.c.status,
                        analysis_page_results.c.payload_json,
                        analysis_page_results.c.created_at,
                    )
                    .join(
                        analysis_page_results,
                        analysis_page_results.c.id
                        == analysis_heads.c.active_result_id,
                    )
                    .join(
                        numbered_pages,
                        numbered_pages.c.page_id == analysis_heads.c.page_id,
                    )
                    .where(
                        analysis_heads.c.book_id == book_id,
                        analysis_heads.c.page_id.is_not(None),
                    )
                    .order_by(
                        analysis_page_results.c.created_at.desc(),
                        numbered_pages.c.page_number.desc(),
                    )
                    .limit(limit)
                ).mappings()
            )
        items = []
        for row in rows:
            page_id = _required_string(row["page_id"], "recent analysis page id")
            page_number = _required_integer(
                row["page_number"],
                "recent analysis page number",
                minimum=1,
            )
            if (
                _required_string(
                    row["result_page_id"],
                    "recent analysis current page id",
                )
                != page_id
                or _required_string(
                    row["page_id_snapshot"],
                    "recent analysis page id snapshot",
                )
                != page_id
                or _required_string(
                    row["status"],
                    "recent analysis status",
                )
                != "published"
                or _required_integer(
                    row["schema_version"],
                    "recent analysis schema version",
                    minimum=1,
                )
                != 2
            ):
                raise InsightConflict(
                    "stored recent page analysis is invalid; "
                    "clear current Insight data"
                )
            snapshot_page_number = _required_integer(
                row["page_number_snapshot"],
                "recent analysis page number snapshot",
                minimum=1,
            )
            payload = _page_analysis(
                row["payload_json"],
                "analysis page payload",
                page_id=page_id,
                page_number=snapshot_page_number,
                source_asset_id=_required_string(
                    row["source_asset_id"],
                    "recent analysis source asset id",
                ),
                source_checksum=_required_sha256(
                    row["source_checksum"],
                    "recent analysis source checksum",
                ),
            )
            items.append(
                {
                    "pageId": page_id,
                    "displayPageNumber": page_number,
                    "summary": payload["page_summary"],
                    "generatedAt": _iso(
                        _required_datetime(
                            row["created_at"],
                            "recent analysis generatedAt",
                        )
                    ),
                }
            )
        return {"items": items}

    def list_chapters(self, book_id: str) -> dict[str, Any]:
        page_rows = self._book_page_statement(book_id).subquery(
            "insight_chapter_page_rows"
        )
        analysis_state = case(
            (
                page_rows.c.latest_job_status.in_(
                    NONTERMINAL_JOB_STATUSES
                )
                & page_rows.c.latest_target_status.is_not(None),
                "running",
            ),
            (
                page_rows.c.latest_target_status.in_(
                    ("failed", "conflict")
                )
                & page_rows.c.active_result_id.is_(None)
                & or_(
                    page_rows.c.latest_job_status.is_(None),
                    ~page_rows.c.latest_job_status.in_(
                        NONTERMINAL_JOB_STATUSES
                    ),
                ),
                "failed",
            ),
            (page_rows.c.active_result_id.is_(None), "not_analyzed"),
            (
                or_(
                    page_rows.c.analysis_source_checksum.is_(None),
                    page_rows.c.analysis_source_checksum
                    != page_rows.c.source_checksum,
                    page_rows.c.analysis_page_number.is_(None),
                    page_rows.c.analysis_page_number
                    != page_rows.c.page_number,
                ),
                "stale",
            ),
            (
                page_rows.c.book_run_id.is_not(None)
                & (page_rows.c.page_run_id != page_rows.c.book_run_id)
                & page_rows.c.book_target_status.in_(
                    ("failed", "conflict")
                )
                & (
                    page_rows.c.page_head_updated_at
                    <= page_rows.c.book_head_updated_at
                ),
                "stale",
            ),
            else_="ready",
        ).label("analysis_state")
        state_rows = select(
            page_rows.c.chapter_id,
            analysis_state,
        ).subquery("insight_chapter_states")
        states = (
            "ready",
            "stale",
            "running",
            "failed",
            "not_analyzed",
        )
        with self.engine.connect() as connection:
            self._assert_book(connection, book_id)
            rows = list(
                connection.execute(
                    select(
                        chapters.c.id,
                        chapters.c.title,
                        chapters.c.ordinal,
                        func.count(state_rows.c.analysis_state).label(
                            "page_count"
                        ),
                        *(
                            func.sum(
                                case(
                                    (
                                        state_rows.c.analysis_state == state,
                                        1,
                                    ),
                                    else_=0,
                                )
                            ).label(state)
                            for state in states
                        ),
                    )
                    .outerjoin(
                        state_rows,
                        state_rows.c.chapter_id == chapters.c.id,
                    )
                    .where(chapters.c.book_id == book_id)
                    .group_by(
                        chapters.c.id,
                        chapters.c.title,
                        chapters.c.ordinal,
                    )
                    .order_by(chapters.c.ordinal)
                ).mappings()
            )
        items: list[dict[str, Any]] = []
        for row in rows:
            analysis_counts = {
                state: _required_integer(
                    row[state],
                    f"chapter {state} analysis count",
                )
                for state in states
            }
            page_count = _required_integer(
                row["page_count"],
                "chapter page count",
            )
            if sum(analysis_counts.values()) != page_count:
                raise InsightConflict(
                    "stored chapter analysis counts are inconsistent; "
                    "clear current Insight data"
                )
            items.append(
                {
                    "chapterId": _required_string(
                        row["id"],
                        "Insight chapter id",
                    ),
                    "title": _required_string(
                        row["title"],
                        "Insight chapter title",
                    ),
                    "ordinal": _required_integer(
                        row["ordinal"],
                        "Insight chapter ordinal",
                        minimum=1,
                    ),
                    "pageCount": page_count,
                    "analysisCounts": analysis_counts,
                }
            )
        return {"items": items}

    def list_pages(
        self,
        *,
        book_id: str,
        chapter_id: str | None,
        after: int,
        limit: int,
    ) -> dict[str, Any]:
        if isinstance(after, bool) or not isinstance(after, int) or after < 0:
            raise ValueError("cursor must be nonnegative")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
            or limit > 200
        ):
            raise ValueError("limit must be between 1 and 200")
        if chapter_id is not None and (
            not isinstance(chapter_id, str) or not chapter_id
        ):
            raise ValueError("chapterId must be a non-empty string")
        with self.engine.connect() as connection:
            self._assert_book(connection, book_id)
            book_pages = self._book_page_statement(book_id).subquery()
            statement = select(book_pages)
            if chapter_id is not None:
                if connection.execute(
                    select(chapters.c.id).where(
                        chapters.c.id == chapter_id,
                        chapters.c.book_id == book_id,
                    )
                ).scalar_one_or_none() is None:
                    raise InsightNotFound("chapter not found")
                statement = statement.where(
                    book_pages.c.chapter_id == chapter_id
                )
            rows = list(
                connection.execute(
                    statement.order_by(book_pages.c.page_number)
                    .offset(after)
                    .limit(limit + 1)
                ).mappings()
            )
        has_more = len(rows) > limit
        window = rows[:limit]
        items: list[dict[str, Any]] = []
        for row in window:
            page_id = _required_string(row["page_id"], "Insight page id")
            source_asset_id = _required_string(
                row["source_asset_id"],
                "Insight page source asset id",
            )
            thumbnail_asset_id = _optional_string(
                row["thumbnail_asset_id"],
                "Insight page thumbnail asset id",
            )
            active_result_id = _optional_string(
                row["active_result_id"],
                "Insight page active analysis id",
            )
            items.append(
                {
                    "pageId": page_id,
                    "chapterId": _required_string(
                        row["chapter_id"],
                        "Insight page chapter id",
                    ),
                    "displayPageNumber": _required_integer(
                        row["page_number"],
                        "Insight page number",
                        minimum=1,
                    ),
                    "sourceAssetId": source_asset_id,
                    "thumbnailUrl": (
                        f"/api/v2/assets/{thumbnail_asset_id}"
                        if thumbnail_asset_id is not None
                        else None
                    ),
                    "analysisState": self._state_for_row(row),
                    "activeAnalysisId": active_result_id,
                }
            )
        return {
            "items": items,
            "nextCursor": after + len(window) if has_more else None,
        }

    def page_detail(
        self,
        *,
        page_id: str,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        source_pointer = page_assets.alias("insight_page_source")
        with self.engine.connect() as connection:
            page = connection.execute(
                select(
                    pages.c.id,
                    pages.c.chapter_id,
                    pages.c.ordinal,
                    chapters.c.book_id,
                    chapters.c.title.label("chapter_title"),
                    source_pointer.c.asset_id.label("source_asset_id"),
                    assets.c.checksum.label("source_checksum"),
                )
                .join(chapters, chapters.c.id == pages.c.chapter_id)
                .join(
                    source_pointer,
                    (source_pointer.c.page_id == pages.c.id)
                    & (source_pointer.c.role == "source"),
                )
                .join(assets, assets.c.id == source_pointer.c.asset_id)
                .where(pages.c.id == page_id)
            ).mappings().one_or_none()
            if page is None:
                raise InsightNotFound("page not found")
            book_id = _required_string(page["book_id"], "Insight page book id")
            chapter_id = _required_string(
                page["chapter_id"],
                "Insight page chapter id",
            )
            source_asset_id = _required_string(
                page["source_asset_id"],
                "Insight page source asset id",
            )
            source_checksum = _required_sha256(
                page["source_checksum"],
                "Insight page source checksum",
            )
            page_number = self._page_number(
                connection,
                page_id=page_id,
                book_id=book_id,
            )
            preview = run_id is not None
            if run_id is not None:
                if not isinstance(run_id, str) or not run_id:
                    raise ValueError("runId must be a non-empty string")
                preview_run = connection.execute(
                    select(
                        analysis_runs.c.book_id,
                        analysis_runs.c.status,
                    ).where(analysis_runs.c.id == run_id)
                ).mappings().one_or_none()
                if preview_run is None or _required_string(
                    preview_run["book_id"],
                    "preview analysis run book id",
                ) != book_id:
                    raise InsightNotFound("analysis run does not belong to page book")
                preview_run_status = _required_string(
                    preview_run["status"],
                    "preview analysis run status",
                )
                if preview_run_status not in ANALYSIS_RUN_STATUSES:
                    raise InsightConflict(
                        "stored preview analysis run status is invalid; "
                        "clear current Insight data"
                    )
                preview_target_status = connection.execute(
                    select(analysis_run_targets.c.status).where(
                        analysis_run_targets.c.run_id == run_id,
                        analysis_run_targets.c.page_id_snapshot == page_id,
                    )
                ).scalar_one_or_none()
                if preview_target_status is None:
                    raise InsightNotFound("analysis run does not target page")
                preview_target_status = _required_string(
                    preview_target_status,
                    "preview analysis target status",
                )
                if preview_target_status not in ANALYSIS_TARGET_STATUSES:
                    raise InsightConflict(
                        "stored preview analysis target status is invalid; "
                        "clear current Insight data"
                    )
                result = connection.execute(
                    select(
                        analysis_page_results,
                        analysis_runs.c.status.label("run_status"),
                        analysis_runs.c.book_id.label("run_book_id"),
                    )
                    .join(
                        analysis_runs,
                        analysis_runs.c.id == analysis_page_results.c.run_id,
                    )
                    .where(
                        analysis_page_results.c.run_id == run_id,
                        analysis_page_results.c.page_id_snapshot == page_id,
                    )
                ).mappings().one_or_none()
            else:
                preview_run_status = None
                preview_target_status = None
                result = connection.execute(
                    select(
                        analysis_page_results,
                        analysis_runs.c.status.label("run_status"),
                        analysis_runs.c.book_id.label("run_book_id"),
                        analysis_heads.c.active_run_id.label("head_run_id"),
                        analysis_heads.c.updated_at.label(
                            "head_updated_at"
                        ),
                    )
                    .join(
                        analysis_heads,
                        analysis_heads.c.active_result_id
                        == analysis_page_results.c.id,
                    )
                    .join(
                        analysis_runs,
                        analysis_runs.c.id == analysis_page_results.c.run_id,
                    )
                    .where(analysis_heads.c.page_id == page_id)
                ).mappings().one_or_none()
            book_head = connection.execute(
                select(
                    analysis_heads.c.active_run_id,
                    analysis_heads.c.updated_at,
                ).where(
                    analysis_heads.c.book_id == book_id,
                    analysis_heads.c.page_id.is_(None),
                )
            ).mappings().one_or_none()
            book_head_run = (
                _required_string(
                    book_head["active_run_id"],
                    "active book analysis run id",
                )
                if book_head is not None
                else None
            )
            book_head_updated_at = (
                _required_datetime(
                    book_head["updated_at"],
                    "active book analysis head updatedAt",
                )
                if book_head is not None
                else None
            )
            book_head_target_status = (
                connection.execute(
                    select(analysis_run_targets.c.status).where(
                        analysis_run_targets.c.run_id == book_head_run,
                        analysis_run_targets.c.page_id_snapshot == page_id,
                    )
                ).scalar_one_or_none()
                if book_head_run is not None
                else None
            )
            if not preview:
                book_page_rows = self._book_page_statement(book_id).subquery(
                    "insight_detail_page_rows"
                )
                current_state_row = connection.execute(
                    select(book_page_rows).where(
                        book_page_rows.c.page_id == page_id
                    )
                ).mappings().one()
            else:
                current_state_row = None

        if result is None:
            analysis = None
            if preview:
                if (
                    preview_run_status == "staging"
                    and preview_target_status == "pending"
                ):
                    state = "running"
                elif preview_target_status in {"failed", "conflict"}:
                    state = "failed"
                elif preview_target_status == "completed":
                    raise InsightConflict(
                        "completed preview target is missing its page result; "
                        "clear current Insight data"
                    )
                else:
                    state = "not_analyzed"
            else:
                state = (
                    "not_analyzed"
                    if current_state_row is None
                    else self._state_for_row(current_state_row)
                )
            stale_reasons: list[str] = []
        else:
            result_book_id = _required_string(
                result["run_book_id"],
                "analysis result book id",
            )
            if result_book_id != book_id:
                raise InsightConflict(
                    "stored page analysis belongs to another book; "
                    "clear current Insight data"
                )
            _required_string(
                result["id"],
                "analysis page result id",
            )
            result_run_id = _required_string(
                result["run_id"],
                "analysis page result run id",
            )
            result_status = _required_string(
                result["status"],
                "analysis page result status",
            )
            if result_status not in {"staging", "published", "stale"}:
                raise InsightConflict("stored page analysis status is invalid")
            if not preview and result_status != "published":
                raise InsightConflict(
                    "stored active page analysis is not published; "
                    "clear current Insight data"
                )
            if _required_integer(
                result["schema_version"],
                "analysis page result schema version",
                minimum=1,
            ) != 2:
                raise InsightConflict(
                    "stored page analysis schema is obsolete; "
                    "clear current Insight data"
                )
            result_page_id = _required_string(
                result["page_id_snapshot"],
                "analysis page result page id snapshot",
            )
            result_current_page_id = _optional_string(
                result["page_id"],
                "analysis page result current page id",
            )
            if result_page_id != page_id or result_current_page_id != page_id:
                raise InsightConflict(
                    "stored page analysis identity is invalid; "
                    "clear current Insight data"
                )
            result_page_number = _required_integer(
                result["page_number_snapshot"],
                "analysis page result page number snapshot",
                minimum=1,
            )
            analysis = _page_analysis(
                result["payload_json"],
                "analysis page payload",
                page_id=page_id,
                page_number=result_page_number,
                source_asset_id=_required_string(
                    result["source_asset_id"],
                    "analysis page source asset id",
                ),
                source_checksum=_required_sha256(
                    result["source_checksum"],
                    "analysis page source checksum",
                ),
            )
            stale_reasons = []
            if analysis["source_asset_id"] != source_asset_id:
                stale_reasons.append("source_changed")
            elif analysis["source_checksum"] != source_checksum:
                stale_reasons.append("source_changed")
            if result_page_number != page_number:
                stale_reasons.append("page_order_changed")
            result_run_status = _required_string(
                result["run_status"],
                "analysis page result run status",
            )
            if result_run_status not in ANALYSIS_RUN_STATUSES:
                raise InsightConflict("stored analysis run status is invalid")
            active_book_target_status = _optional_string(
                book_head_target_status,
                "active book analysis target status",
            )
            if (
                active_book_target_status is not None
                and active_book_target_status not in ANALYSIS_TARGET_STATUSES
            ):
                raise InsightConflict(
                    "stored active book analysis target status is invalid"
                )
            if active_book_target_status == "pending":
                raise InsightConflict(
                    "stored active book analysis still has a pending target; "
                    "clear current Insight data"
                )
            if (
                not preview
                and book_head_run is not None
                and result_run_id
                != book_head_run
                and active_book_target_status in {"failed", "conflict"}
                and _required_datetime(
                    result["head_updated_at"],
                    "active page analysis head updatedAt",
                )
                <= _required_datetime(
                    book_head_updated_at,
                    "active book analysis head updatedAt",
                )
            ):
                stale_reasons.append("fallback_from_previous_run")
            if not preview:
                head_run_id = _required_string(
                    result["head_run_id"],
                    "active page analysis run id",
                )
                if head_run_id != result_run_id:
                    raise InsightConflict(
                        "stored page analysis head is inconsistent; "
                        "clear current Insight data"
                    )
            result_state = "stale" if stale_reasons else "ready"
            if current_state_row is None:
                state = result_state
            else:
                state = self._state_for_row(current_state_row)
                if state not in {result_state, "running"}:
                    raise InsightConflict(
                        "stored page analysis state is inconsistent; "
                        "clear current Insight data"
                    )
        return {
            "pageId": page_id,
            "bookId": book_id,
            "chapterId": chapter_id,
            "chapterTitle": _required_string(
                page["chapter_title"],
                "Insight page chapter title",
            ),
            "displayPageNumber": page_number,
            "sourceAssetId": source_asset_id,
            "sourceUrl": f"/api/v2/assets/{source_asset_id}",
            "analysisState": state,
            "staleReasons": stale_reasons,
            "preview": preview,
            "analysis": analysis,
            "runId": result_run_id if result is not None else None,
            "generatedAt": (
                _iso(
                    _required_datetime(
                        result["created_at"],
                        "analysis page generatedAt",
                    )
                )
                if result is not None
                else None
            ),
        }

    def list_notes(
        self,
        *,
        book_id: str,
        cursor: str | None = None,
        limit: int = 50,
        kind: str | None = None,
        include_content: bool = False,
    ) -> dict[str, Any]:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 200
        ):
            raise ValueError("note limit must be between 1 and 200")
        if kind is not None and kind not in {"text", "qa"}:
            raise ValueError("note kind must be text or qa")
        cursor_value = _decode_note_cursor(cursor) if cursor else None
        with self.engine.connect() as connection:
            self._assert_book(connection, book_id)
            statement = select(notes).where(notes.c.book_id == book_id)
            if kind is not None:
                statement = statement.where(notes.c.kind == kind)
            if cursor_value is not None:
                cursor_time, cursor_id = cursor_value
                statement = statement.where(
                    or_(
                        notes.c.updated_at < cursor_time,
                        (
                            (notes.c.updated_at == cursor_time)
                            & (notes.c.id < cursor_id)
                        ),
                    )
                )
            rows = list(
                connection.execute(
                    statement.order_by(
                        notes.c.updated_at.desc(),
                        notes.c.id.desc(),
                    ).limit(limit + 1)
                ).mappings()
            )
            has_more = len(rows) > limit
            selected_rows = rows[:limit]
            citations_by_note: dict[str, list[Mapping[str, Any]]] = {}
            if selected_rows:
                selected_note_ids = [
                    _required_string(row["id"], "note id")
                    for row in selected_rows
                ]
                for citation in connection.execute(
                    select(note_citations)
                    .where(
                        note_citations.c.note_id.in_(
                            selected_note_ids
                        )
                    )
                    .order_by(
                        note_citations.c.note_id,
                        note_citations.c.ordinal,
                    )
                ).mappings():
                    citation_note_id = _required_string(
                        citation["note_id"],
                        "note citation note id",
                    )
                    if citation_note_id not in selected_note_ids:
                        raise InsightConflict(
                            "stored note citation belongs to another note; "
                            "clear current Insight data"
                        )
                    citations_by_note.setdefault(
                        citation_note_id,
                        [],
                    ).append(citation)
            items = [
                self._note_dto(
                    connection,
                    row,
                    summary=not include_content,
                    citations=citations_by_note.get(
                        _required_string(row["id"], "note id"),
                        (),
                    ),
                )
                for row in selected_rows
            ]
        return {
            "items": items,
            "nextCursor": (
                _encode_note_cursor(
                    _required_datetime(
                        selected_rows[-1]["updated_at"],
                        "note updatedAt",
                    ),
                    _required_string(selected_rows[-1]["id"], "note id"),
                )
                if has_more and selected_rows
                else None
            ),
        }

    def get_note(self, *, note_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(notes).where(notes.c.id == note_id)
            ).mappings().one_or_none()
            if row is None:
                raise InsightNotFound("note not found")
            return self._note_dto(connection, row)

    def create_note(
        self,
        *,
        idempotency_key: str,
        book_id: str,
        title: str,
        content: str,
        citations: Sequence[Mapping[str, Any]] = (),
        kind: str = "text",
        tags: Sequence[str] = (),
        question: str | None = None,
        comment: str | None = None,
    ) -> dict[str, Any]:
        if (
            not isinstance(title, str)
            or not title
            or title != title.strip()
            or len(title) > 500
        ):
            raise ValueError("note title must contain 1-500 trimmed characters")
        if not isinstance(content, str):
            raise ValueError("note content must be a string")
        kind, tags, metadata = _validate_note_metadata(
            kind=kind,
            tags=tags,
            question=question,
            comment=comment,
        )
        if kind == "qa" and not content.strip():
            raise ValueError("qa note content is required")
        normalized_citations = _normalize_note_citations(citations)
        request_payload = {
            "bookId": book_id,
            "title": title,
            "content": content,
            "citations": normalized_citations,
            "kind": kind,
            "tags": tags,
            "question": metadata["question"],
            "comment": metadata["comment"],
        }
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            request_hash, replay = _idempotency_replay(
                connection,
                scope="POST:createInsightNote",
                key=idempotency_key,
                payload=request_payload,
                now=now,
            )
            if replay is not None:
                return replay
            self._assert_book(connection, book_id)
            note_id = str(uuid.uuid4())
            connection.execute(
                insert(notes).values(
                    id=note_id,
                    owner_user_id=effective_owner_id(),
                    book_id=book_id,
                    title=title,
                    content=content,
                    kind=kind,
                    tags_json=_json(tags),
                    comments_json=_json(metadata),
                )
            )
            self._replace_citations(
                connection,
                note_id=note_id,
                book_id=book_id,
                citations=normalized_citations,
            )
            row = connection.execute(
                select(notes).where(notes.c.id == note_id)
            ).mappings().one()
            response = self._note_dto(connection, row)
            _record_idempotency(
                connection,
                scope="POST:createInsightNote",
                key=idempotency_key,
                request_hash=request_hash,
                response=response,
                http_status=201,
                resource_type="note",
                resource_id=note_id,
                now=now,
            )
            return response

    def update_note(
        self,
        *,
        idempotency_key: str,
        note_id: str,
        base_revision: int,
        title: str,
        content: str,
        citations: Sequence[Mapping[str, Any]] = (),
        kind: str = "text",
        tags: Sequence[str] = (),
        question: str | None = None,
        comment: str | None = None,
    ) -> dict[str, Any]:
        if (
            not isinstance(title, str)
            or not title
            or title != title.strip()
            or len(title) > 500
        ):
            raise ValueError("note title must contain 1-500 trimmed characters")
        if not isinstance(content, str):
            raise ValueError("note content must be a string")
        kind, tags, metadata = _validate_note_metadata(
            kind=kind,
            tags=tags,
            question=question,
            comment=comment,
        )
        if kind == "qa" and not content.strip():
            raise ValueError("qa note content is required")
        if (
            isinstance(base_revision, bool)
            or not isinstance(base_revision, int)
            or base_revision < 1
        ):
            raise ValueError("baseRevision must be an integer of at least 1")
        normalized_citations = _normalize_note_citations(citations)
        request_payload = {
            "baseRevision": base_revision,
            "title": title,
            "content": content,
            "citations": normalized_citations,
            "kind": kind,
            "tags": tags,
            "question": metadata["question"],
            "comment": metadata["comment"],
        }
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            scope = f"PATCH:updateInsightNote:{note_id}"
            request_hash, replay = _idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                payload=request_payload,
                now=now,
            )
            if replay is not None:
                return replay
            current = connection.execute(
                select(notes.c.book_id, notes.c.revision).where(notes.c.id == note_id)
            ).mappings().one_or_none()
            if current is None:
                raise InsightNotFound("note not found")
            if _required_integer(
                current["revision"],
                "note revision",
                minimum=1,
            ) != base_revision:
                raise InsightConflict("note revision changed")
            changed = connection.execute(
                update(notes)
                .where(
                    notes.c.id == note_id,
                    notes.c.revision == base_revision,
                )
                .values(
                    title=title,
                    content=content,
                    kind=kind,
                    tags_json=_json(tags),
                    comments_json=_json(metadata),
                    revision=base_revision + 1,
                    updated_at=now,
                )
            )
            if changed.rowcount != 1:
                raise InsightConflict("note revision changed")
            self._replace_citations(
                connection,
                note_id=note_id,
                book_id=_required_string(current["book_id"], "note book id"),
                citations=normalized_citations,
            )
            row = connection.execute(
                select(notes).where(notes.c.id == note_id)
            ).mappings().one()
            response = self._note_dto(connection, row)
            _record_idempotency(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                response=response,
                http_status=200,
                resource_type="note",
                resource_id=note_id,
                now=now,
            )
            return response

    def delete_note(
        self,
        *,
        idempotency_key: str,
        note_id: str,
        base_revision: int,
    ) -> None:
        if (
            isinstance(base_revision, bool)
            or not isinstance(base_revision, int)
            or base_revision < 1
        ):
            raise ValueError("baseRevision must be an integer of at least 1")
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            scope = f"DELETE:deleteInsightNote:{note_id}"
            request_hash, replay = _idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                payload={"baseRevision": base_revision},
                now=now,
            )
            if replay is not None:
                return
            changed = connection.execute(
                delete(notes).where(
                    notes.c.id == note_id,
                    notes.c.revision == base_revision,
                )
            )
            if changed.rowcount != 1:
                exists_row = connection.execute(
                    select(notes.c.id).where(notes.c.id == note_id)
                ).scalar_one_or_none()
                if exists_row is None:
                    raise InsightNotFound("note not found")
                raise InsightConflict("note revision changed")
            _record_idempotency(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                response={"deleted": True},
                http_status=200,
                resource_type="note",
                resource_id=note_id,
                now=now,
            )

    @staticmethod
    def _numbered_book_pages_statement(book_id: str):
        return (
            select(
                pages.c.id.label("page_id"),
                func.row_number()
                .over(order_by=(chapters.c.ordinal, pages.c.ordinal))
                .label("page_number"),
            )
            .join(chapters, chapters.c.id == pages.c.chapter_id)
            .where(chapters.c.book_id == book_id)
        )

    @staticmethod
    def _book_page_statement(book_id: str):
        source_pointer = page_assets.alias("insight_source_pointer")
        thumbnail_pointer = page_assets.alias("insight_thumbnail_pointer")
        page_head = analysis_heads.alias("insight_page_head")
        book_head = analysis_heads.alias("insight_book_head")
        book_run_target = analysis_run_targets.alias(
            "insight_book_run_target"
        )
        ranked_targets = (
            select(
                analysis_run_targets.c.page_id_snapshot.label("page_id"),
                analysis_run_targets.c.status.label("target_status"),
                jobs.c.status.label("job_status"),
                func.row_number()
                .over(
                    partition_by=analysis_run_targets.c.page_id_snapshot,
                    order_by=(
                        analysis_runs.c.created_at.desc(),
                        analysis_runs.c.id.desc(),
                    ),
                )
                .label("target_rank"),
            )
            .join(
                analysis_runs,
                analysis_runs.c.id == analysis_run_targets.c.run_id,
            )
            .join(jobs, jobs.c.id == analysis_runs.c.job_id, isouter=True)
            .where(analysis_runs.c.book_id == book_id)
            .subquery("insight_ranked_targets")
        )
        latest_target = (
            select(
                ranked_targets.c.page_id,
                ranked_targets.c.target_status,
                ranked_targets.c.job_status,
            )
            .where(ranked_targets.c.target_rank == 1)
            .subquery("insight_latest_target")
        )
        return (
            select(
                pages.c.id.label("page_id"),
                pages.c.ordinal.label("page_ordinal"),
                chapters.c.id.label("chapter_id"),
                chapters.c.title.label("chapter_title"),
                chapters.c.ordinal.label("chapter_ordinal"),
                func.row_number()
                .over(order_by=(chapters.c.ordinal, pages.c.ordinal))
                .label("page_number"),
                source_pointer.c.asset_id.label("source_asset_id"),
                assets.c.checksum.label("source_checksum"),
                thumbnail_pointer.c.asset_id.label("thumbnail_asset_id"),
                page_head.c.active_result_id,
                page_head.c.active_run_id.label("page_run_id"),
                page_head.c.updated_at.label("page_head_updated_at"),
                book_head.c.active_run_id.label("book_run_id"),
                book_head.c.updated_at.label("book_head_updated_at"),
                book_run_target.c.status.label("book_target_status"),
                analysis_page_results.c.source_checksum.label(
                    "analysis_source_checksum"
                ),
                analysis_page_results.c.page_number_snapshot.label(
                    "analysis_page_number"
                ),
                latest_target.c.target_status.label("latest_target_status"),
                latest_target.c.job_status.label("latest_job_status"),
            )
            .join(chapters, chapters.c.id == pages.c.chapter_id)
            .join(
                source_pointer,
                (source_pointer.c.page_id == pages.c.id)
                & (source_pointer.c.role == "source"),
            )
            .join(assets, assets.c.id == source_pointer.c.asset_id)
            .join(
                thumbnail_pointer,
                (thumbnail_pointer.c.page_id == pages.c.id)
                & (thumbnail_pointer.c.role == "thumbnail_source"),
                isouter=True,
            )
            .join(
                page_head,
                page_head.c.page_id == pages.c.id,
                isouter=True,
            )
            .join(
                book_head,
                (book_head.c.book_id == chapters.c.book_id)
                & book_head.c.page_id.is_(None),
                isouter=True,
            )
            .join(
                book_run_target,
                (book_run_target.c.run_id == book_head.c.active_run_id)
                & (
                    book_run_target.c.page_id_snapshot
                    == pages.c.id
                ),
                isouter=True,
            )
            .join(
                analysis_page_results,
                analysis_page_results.c.id == page_head.c.active_result_id,
                isouter=True,
            )
            .join(
                latest_target,
                latest_target.c.page_id == pages.c.id,
                isouter=True,
            )
            .where(chapters.c.book_id == book_id)
        )

    @staticmethod
    def _state_for_row(row: Mapping[str, Any]) -> str:
        latest_job = _optional_string(
            row["latest_job_status"],
            "latest Insight job status",
        )
        if latest_job is not None and latest_job not in JOB_STATUSES:
            raise InsightConflict("stored latest Insight job status is invalid")
        latest_target = _optional_string(
            row["latest_target_status"],
            "latest Insight target status",
        )
        if (
            latest_target is not None
            and latest_target not in ANALYSIS_TARGET_STATUSES
        ):
            raise InsightConflict("stored latest Insight target status is invalid")
        active_result_id = _optional_string(
            row["active_result_id"],
            "active page analysis id",
        )
        analysis_checksum = (
            None
            if row["analysis_source_checksum"] is None
            else _required_sha256(
                row["analysis_source_checksum"],
                "active page analysis source checksum",
            )
        )
        source_checksum = _required_sha256(
            row["source_checksum"],
            "page source checksum",
        )
        analysis_page_number = (
            None
            if row["analysis_page_number"] is None
            else _required_integer(
                row["analysis_page_number"],
                "active page analysis page number",
                minimum=1,
            )
        )
        current_page_number = _required_integer(
            row["page_number"],
            "current Insight page number",
            minimum=1,
        )
        page_run_id = _optional_string(
            row["page_run_id"],
            "active page analysis run id",
        )
        page_head_updated_at = (
            None
            if row["page_head_updated_at"] is None
            else _required_datetime(
                row["page_head_updated_at"],
                "active page analysis head updatedAt",
            )
        )
        book_run_id = _optional_string(
            row["book_run_id"],
            "active book analysis run id",
        )
        book_head_updated_at = (
            None
            if row["book_head_updated_at"] is None
            else _required_datetime(
                row["book_head_updated_at"],
                "active book analysis head updatedAt",
            )
        )
        book_target_status = _optional_string(
            row["book_target_status"],
            "active book analysis target status",
        )
        if (
            book_target_status is not None
            and book_target_status not in ANALYSIS_TARGET_STATUSES
        ):
            raise InsightConflict(
                "stored active book analysis target status is invalid"
            )
        if (book_run_id is None) != (book_head_updated_at is None) or (
            book_run_id is None and book_target_status is not None
        ):
            raise InsightConflict(
                "stored active book analysis head is incomplete; "
                "clear current Insight data"
            )
        if book_target_status == "pending":
            raise InsightConflict(
                "stored active book analysis still has a pending target; "
                "clear current Insight data"
            )
        if active_result_id is None:
            if (
                analysis_checksum is not None
                or analysis_page_number is not None
                or page_run_id is not None
                or page_head_updated_at is not None
            ):
                raise InsightConflict(
                    "stored page analysis head is incomplete; "
                    "clear current Insight data"
                )
        elif (
            analysis_checksum is None
            or analysis_page_number is None
            or page_run_id is None
            or page_head_updated_at is None
        ):
            raise InsightConflict(
                "stored page analysis result is incomplete; "
                "clear current Insight data"
            )
        if (
            latest_job in NONTERMINAL_JOB_STATUSES
            and latest_target is not None
        ):
            return "running"
        if active_result_id is None:
            if (
                latest_target in {"failed", "conflict"}
                and latest_job not in NONTERMINAL_JOB_STATUSES
            ):
                return "failed"
            return "not_analyzed"
        if analysis_checksum != source_checksum:
            return "stale"
        if analysis_page_number != current_page_number:
            return "stale"
        if (
            book_run_id is not None
            and page_run_id != book_run_id
            and book_target_status in {"failed", "conflict"}
            and page_head_updated_at <= _required_datetime(
                book_head_updated_at,
                "active book analysis head updatedAt",
            )
        ):
            return "stale"
        return "ready"

    @staticmethod
    def _upsert_page_head(
        connection: Connection,
        *,
        book_id: str,
        page_id: str,
        run_id: str,
        result_id: str,
        now: datetime,
    ) -> None:
        head_id = connection.execute(
            select(analysis_heads.c.id).where(analysis_heads.c.page_id == page_id)
        ).scalar_one_or_none()
        values = {
            "book_id": book_id,
            "active_run_id": run_id,
            "active_result_id": result_id,
            "updated_at": now,
        }
        if head_id is None:
            connection.execute(
                insert(analysis_heads).values(
                    id=str(uuid.uuid4()),
                    page_id=page_id,
                    created_at=now,
                    **values,
                )
            )
        else:
            connection.execute(
                update(analysis_heads)
                .where(analysis_heads.c.id == head_id)
                .values(**values)
            )

    @staticmethod
    def _upsert_book_head(
        connection: Connection,
        *,
        book_id: str,
        run_id: str,
        now: datetime,
    ) -> None:
        head_id = connection.execute(
            select(analysis_heads.c.id).where(
                analysis_heads.c.book_id == book_id,
                analysis_heads.c.page_id.is_(None),
            )
        ).scalar_one_or_none()
        if head_id is None:
            connection.execute(
                insert(analysis_heads).values(
                    id=str(uuid.uuid4()),
                    book_id=book_id,
                    page_id=None,
                    active_run_id=run_id,
                    active_result_id=None,
                    created_at=now,
                    updated_at=now,
                )
            )
        else:
            connection.execute(
                update(analysis_heads)
                .where(analysis_heads.c.id == head_id)
                .values(active_run_id=run_id, updated_at=now)
            )

    @staticmethod
    def _refresh_run_counts(
        connection: Connection,
        run_id: str,
        now: datetime,
    ) -> None:
        counts: dict[str, int] = {}
        for raw_status, raw_count in connection.execute(
            select(
                analysis_run_targets.c.status,
                func.count(),
            )
            .where(analysis_run_targets.c.run_id == run_id)
            .group_by(analysis_run_targets.c.status)
        ):
            status = _required_string(raw_status, "analysis target status")
            if status not in ANALYSIS_TARGET_STATUSES or status in counts:
                raise InsightConflict(
                    "stored analysis target counts are invalid; "
                    "clear current Insight data"
                )
            counts[status] = _required_integer(
                raw_count,
                "analysis target count",
            )
        changed = connection.execute(
            update(analysis_runs)
            .where(analysis_runs.c.id == run_id)
            .values(
                success_count=counts.get("completed", 0),
                failed_count=counts.get("failed", 0)
                + counts.get("conflict", 0),
                updated_at=now,
            )
        )
        if changed.rowcount != 1:
            raise InsightNotFound("analysis run not found")

    @staticmethod
    def _assert_book(connection: Connection, book_id: str) -> None:
        if connection.execute(
            select(books.c.id).where(
                books.c.id == book_id,
                books.c.kind == "library",
                books.c.owner_user_id == effective_owner_id(),
            )
        ).scalar_one_or_none() is None:
            raise InsightNotFound("book not found")

    @classmethod
    def _page_number(
        cls,
        connection: Connection,
        *,
        page_id: str,
        book_id: str,
    ) -> int:
        book_pages = cls._numbered_book_pages_statement(book_id).subquery()
        page_number = connection.execute(
            select(book_pages.c.page_number).where(
                book_pages.c.page_id == page_id
            )
        ).scalar_one_or_none()
        if page_number is None:
            raise InsightNotFound("page not found")
        return _required_integer(
            page_number,
            "Insight page number",
            minimum=1,
        )

    @staticmethod
    def _replace_citations(
        connection: Connection,
        *,
        note_id: str,
        book_id: str,
        citations: Sequence[Mapping[str, Any]],
    ) -> None:
        normalized = _normalize_note_citations(citations)
        page_ids = [citation["pageId"] for citation in normalized]
        connection.execute(
            delete(note_citations).where(note_citations.c.note_id == note_id)
        )
        if not page_ids:
            return
        book_pages = InsightRepository._numbered_book_pages_statement(
            book_id
        ).subquery()
        page_numbers: dict[str, int] = {}
        for row in connection.execute(
            select(book_pages.c.page_id, book_pages.c.page_number).where(
                book_pages.c.page_id.in_(page_ids)
            )
        ).mappings():
            citation_page_id = _required_string(
                row["page_id"],
                "citation page id",
            )
            if citation_page_id in page_numbers:
                raise InsightConflict(
                    "stored book page numbers are duplicated; "
                    "clear current Insight data"
                )
            page_numbers[citation_page_id] = _required_integer(
                row["page_number"],
                "citation page number",
                minimum=1,
            )
        if not set(page_ids).issubset(page_numbers):
            raise ValueError("all citation pages must belong to the note book")
        active_results: dict[str, str] = {}
        for row in connection.execute(
            select(
                analysis_heads.c.page_id,
                analysis_heads.c.active_result_id,
            ).where(analysis_heads.c.page_id.in_(page_ids))
        ).mappings():
            active_page_id = _required_string(
                row["page_id"],
                "citation active analysis page id",
            )
            if active_page_id in active_results:
                raise InsightConflict(
                    "stored page analysis heads are duplicated; "
                    "clear current Insight data"
                )
            active_results[active_page_id] = _required_string(
                row["active_result_id"],
                "citation source analysis id",
            )
        connection.execute(
            insert(note_citations),
            [
                {
                    "note_id": note_id,
                    "ordinal": ordinal,
                    "page_id": page_id,
                    "page_id_snapshot": page_id,
                    "page_number_snapshot": page_numbers[page_id],
                    "source_analysis_id": active_results.get(page_id),
                    "excerpt": normalized[ordinal - 1].get("excerpt", ""),
                    "score": normalized[ordinal - 1].get("score"),
                }
                for ordinal, page_id in enumerate(page_ids, 1)
            ],
        )

    @staticmethod
    def _note_dto(
        connection: Connection,
        row: Mapping[str, Any],
        *,
        summary: bool = False,
        citations: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if citations is None:
            citations = list(
                connection.execute(
                    select(note_citations)
                    .where(note_citations.c.note_id == row["id"])
                    .order_by(note_citations.c.ordinal)
                ).mappings()
            )
        note_id = _required_string(row["id"], "note id")
        book_id = _required_string(row["book_id"], "note book id")
        title = _required_string(row["title"], "note title")
        content = row["content"]
        if not isinstance(content, str):
            raise InsightConflict(
                "stored note content is invalid; clear current Insight data"
            )
        kind = _required_string(row["kind"], "note kind")
        if kind not in {"text", "qa"}:
            raise InsightConflict(
                "stored note kind is invalid; clear current Insight data"
            )
        tags = _json_array(row["tags_json"], "note tags")
        if any(
            not isinstance(value, str)
            or not value
            or value != value.strip()
            for value in tags
        ) or len(set(tags)) != len(tags):
            raise InsightConflict(
                "stored note tags are invalid; clear current Insight data"
            )
        metadata = _stored_note_metadata(row["comments_json"])
        if kind == "qa" and metadata["question"] is None:
            raise InsightConflict(
                "stored QA note question is missing; clear current Insight data"
            )
        if kind == "text" and metadata["question"] is not None:
            raise InsightConflict(
                "stored text note metadata is invalid; clear current Insight data"
            )
        citation_items: list[dict[str, Any]] = []
        for expected_ordinal, citation in enumerate(citations, start=1):
            ordinal = _required_integer(
                citation["ordinal"],
                "note citation ordinal",
                minimum=1,
            )
            if ordinal != expected_ordinal:
                raise InsightConflict(
                    "stored note citation order is invalid; "
                    "clear current Insight data"
                )
            citation_note_id = _required_string(
                citation["note_id"],
                "note citation note id",
            )
            if citation_note_id != note_id:
                raise InsightConflict(
                    "stored citation belongs to another note; "
                    "clear current Insight data"
                )
            citation_page_id = _optional_string(
                citation["page_id"],
                "note citation current page id",
            )
            page_id_snapshot = _required_string(
                citation["page_id_snapshot"],
                "note citation page id snapshot",
            )
            source_analysis_id = _optional_string(
                citation["source_analysis_id"],
                "note citation source analysis id",
            )
            excerpt = citation["excerpt"]
            if not isinstance(excerpt, str):
                raise InsightConflict(
                    "stored note citation excerpt is invalid; "
                    "clear current Insight data"
                )
            score = citation["score"]
            if score is not None and (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                raise InsightConflict(
                    "stored note citation score is invalid; "
                    "clear current Insight data"
                )
            citation_items.append(
                {
                    "pageId": citation_page_id,
                    "pageIdSnapshot": page_id_snapshot,
                    "pageNumberSnapshot": _required_integer(
                        citation["page_number_snapshot"],
                        "note citation page number snapshot",
                        minimum=1,
                    ),
                    "sourceAnalysisId": source_analysis_id,
                    "excerpt": excerpt,
                    "score": score,
                }
            )
        return {
            "noteId": note_id,
            "bookId": book_id,
            "title": title,
            "content": None if summary else content,
            "excerpt": content[:300] if summary else None,
            "kind": kind,
            "tags": tags,
            "question": metadata["question"],
            "comment": metadata["comment"],
            "revision": _required_integer(
                row["revision"],
                "note revision",
                minimum=1,
            ),
            "citations": citation_items,
            "createdAt": _iso(
                _required_datetime(row["created_at"], "note createdAt")
            ),
            "updatedAt": _iso(
                _required_datetime(row["updated_at"], "note updatedAt")
            ),
        }
