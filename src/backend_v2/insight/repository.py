"""Relational fact source and publication transactions for Manga Insight."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import base64
from datetime import datetime
import json
from typing import Any
import uuid

from sqlalchemy import Engine, case, delete, func, insert, or_, select, update
from sqlalchemy.engine import Connection

from src.backend_v2.serialization import canonical_json as _json
from src.backend_v2.timestamps import iso_utc as _iso, utcnow
from src.backend_v2.redaction import redact_sensitive_text
from src.backend_v2.storage.database import immediate_transaction
from src.backend_v2.storage.schema import (
    analysis_artifacts,
    analysis_heads,
    analysis_layer_results,
    analysis_page_results,
    analysis_run_targets,
    analysis_runs,
    assets,
    books,
    chapters,
    jobs,
    note_citations,
    notes,
    page_assets,
    pages,
    timeline_versions,
    vector_generations,
)


NONTERMINAL_JOB_STATUSES = (
    "queued",
    "running",
    "pausing",
    "paused",
    "cancelling",
    "interrupted",
)


class InsightNotFound(LookupError):
    pass


class InsightConflict(RuntimeError):
    pass


class InsightLocked(InsightConflict):
    pass


def _load(value: str | None, default: object) -> object:
    if not value:
        return default
    return json.loads(value)


def _encode_note_cursor(updated_at: datetime, note_id: str) -> str:
    raw = f"{updated_at.isoformat()}|{note_id}".encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_note_cursor(value: str) -> tuple[datetime, str]:
    try:
        padded = value + "=" * (-len(value) % 4)
        decoded = base64.urlsafe_b64decode(
            padded.encode("ascii")
        ).decode("utf-8")
        timestamp, note_id = decoded.rsplit("|", 1)
        parsed = datetime.fromisoformat(timestamp)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("invalid note cursor") from exc
    if not note_id:
        raise ValueError("invalid note cursor")
    return parsed, note_id


def _normalize_note_metadata(
    *,
    kind: str,
    tags: Sequence[str],
    comments: Sequence[Mapping[str, Any] | str],
) -> tuple[str, list[str], list[dict[str, Any]]]:
    if kind not in {"text", "qa"}:
        raise ValueError("note kind must be text or qa")
    normalized_tags = list(
        dict.fromkeys(
            str(value).strip()
            for value in tags
            if str(value).strip()
        )
    )
    if len(normalized_tags) > 100 or any(
        len(value) > 100 for value in normalized_tags
    ):
        raise ValueError("note tags exceed the allowed size")
    normalized_comments = []
    for value in comments:
        if isinstance(value, str):
            comment = {"text": value}
        elif isinstance(value, Mapping):
            comment = dict(value)
        else:
            raise ValueError("note comments must be strings or objects")
        text_value = str(comment.get("text", "")).strip()
        if not text_value or len(text_value) > 10_000:
            raise ValueError(
                "every note comment must contain 1-10000 characters"
            )
        normalized_comments.append({**comment, "text": text_value})
    if len(normalized_comments) > 1000:
        raise ValueError("note has too many comments")
    return kind, normalized_tags, normalized_comments


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
        return dict(row)

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
        now = utcnow()
        existing = connection.execute(
            select(analysis_page_results.c.id).where(
                analysis_page_results.c.run_id == run_id,
                analysis_page_results.c.page_id_snapshot == page_id,
            )
        ).scalar_one_or_none()
        result_id = str(existing or uuid.uuid4())
        values = {
            "page_id": page_id,
            "source_asset_id": source_asset_id,
            "source_checksum": source_checksum,
            "page_id_snapshot": page_id,
            "page_number_snapshot": page_number,
            "payload_json": _json(dict(payload)),
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
        connection.execute(
            update(analysis_run_targets)
            .where(
                analysis_run_targets.c.run_id == run_id,
                analysis_run_targets.c.page_id_snapshot == page_id,
            )
            .values(status="completed", error_json=None)
        )
        InsightRepository._refresh_run_counts(connection, run_id, now)
        if scope != "full":
            book_id = connection.execute(
                select(analysis_runs.c.book_id).where(analysis_runs.c.id == run_id)
            ).scalar_one()
            InsightRepository._upsert_page_head(
                connection,
                book_id=str(book_id),
                page_id=page_id,
                run_id=run_id,
                result_id=result_id,
                now=now,
            )
            InsightRepository._mark_derived_stale(
                connection,
                book_id=str(book_id),
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
        now = utcnow()
        prepared: list[dict[str, Any]] = []
        result_ids_by_page: dict[str, str] = {}
        for copy in copies:
            page_id = str(copy["page_id"])
            result_id = str(uuid.uuid4())
            result_ids_by_page[page_id] = result_id
            prepared.append(
                {
                    "id": result_id,
                    "run_id": run_id,
                    "page_id": page_id,
                    "source_asset_id": str(copy["source_asset_id"]),
                    "source_checksum": str(copy["source_checksum"]),
                    "page_id_snapshot": page_id,
                    "page_number_snapshot": int(copy["page_number"]),
                    "payload_json": _json(dict(copy["payload"])),
                    "schema_version": 2,
                    "status": "staging" if scope == "full" else "published",
                    "created_at": now,
                    "updated_at": now,
                }
            )
        connection.execute(insert(analysis_page_results), prepared)
        page_ids = tuple(result_ids_by_page)
        connection.execute(
            update(analysis_run_targets)
            .where(
                analysis_run_targets.c.run_id == run_id,
                analysis_run_targets.c.page_id_snapshot.in_(page_ids),
            )
            .values(status="completed", error_json=None)
        )
        InsightRepository._refresh_run_counts(connection, run_id, now)
        if scope == "full":
            return

        book_id = connection.execute(
            select(analysis_runs.c.book_id).where(analysis_runs.c.id == run_id)
        ).scalar_one()
        for page_id, result_id in result_ids_by_page.items():
            InsightRepository._upsert_page_head(
                connection,
                book_id=str(book_id),
                page_id=page_id,
                run_id=run_id,
                result_id=result_id,
                now=now,
            )
        InsightRepository._mark_derived_stale(
            connection,
            book_id=str(book_id),
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
        if any(target["status"] == "pending" for target in targets):
            raise InsightConflict("analysis run still has pending targets")
        completed_targets = [
            target for target in targets if target["status"] == "completed"
        ]
        current_by_page: dict[str, Mapping[str, Any]] = {}
        current_page_ids = [
            str(target["page_id"])
            for target in completed_targets
            if target["page_id"] is not None
        ]
        if current_page_ids:
            current_by_page = {
                str(row["page_id"]): row
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
                ).mappings()
            }
        changed: list[str] = []
        for target in completed_targets:
            current = current_by_page.get(str(target["page_id"]))
            if (
                current is not None
                and str(current["asset_id"]) == str(target["source_asset_id"])
                and str(current["checksum"]) == str(target["source_checksum"])
            ):
                continue
            changed.append(str(target["page_id_snapshot"]))
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
        counts = {
            str(status): int(count)
            for status, count in connection.execute(
                select(analysis_run_targets.c.status, func.count())
                .where(analysis_run_targets.c.run_id == run_id)
                .group_by(analysis_run_targets.c.status)
            )
        }
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
        InsightRepository.validate_run_sources(connection, run_id=run_id)

        refreshed = list(
            connection.execute(
                select(analysis_run_targets)
                .where(analysis_run_targets.c.run_id == run_id)
                .order_by(analysis_run_targets.c.ordinal)
            ).mappings()
        )
        successful = [
            target for target in refreshed if target["status"] == "completed"
        ]
        missing = [
            str(target["page_id_snapshot"])
            for target in refreshed
            if target["status"] != "completed"
        ]
        success_count = len(successful)
        failed_count = len(refreshed) - success_count
        if success_count == 0:
            connection.execute(
                update(analysis_runs)
                .where(analysis_runs.c.id == run_id)
                .values(
                    status="failed",
                    success_count=0,
                    failed_count=failed_count,
                    missing_page_ids_json=_json(missing),
                    updated_at=now,
                )
            )
            raise InsightConflict("analysis run has no publishable page results")

        final_status = "completed_with_errors" if failed_count else "completed"
        if str(run["scope"]) == "full":
            config = _load(run["config_json"], {})
            layers = (
                config.get("analysis", {}).get("layers", [])
                if isinstance(config, Mapping)
                and isinstance(config.get("analysis"), Mapping)
                else []
            )
            expected_layer_indices = {
                int(layer["index"])
                for layer in layers
                if isinstance(layer, Mapping) and "index" in layer
            }
            actual_layer_indices = set(
                int(value)
                for value in connection.execute(
                    select(analysis_layer_results.c.layer_index)
                    .where(
                        analysis_layer_results.c.run_id == run_id,
                        analysis_layer_results.c.status == "staging",
                    )
                    .distinct()
                ).scalars()
            )
            if actual_layer_indices != expected_layer_indices:
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
                        analysis_artifacts.c.is_active.is_(False),
                        analysis_artifacts.c.status == "building",
                    )
                ).mappings()
            )
            if {
                (str(row["kind"]), str(row["template"]))
                for row in staged_artifacts
            } != required_artifacts:
                raise InsightConflict(
                    "full analysis run is missing required overview artifacts"
                )
            staged_timeline = connection.execute(
                select(timeline_versions.c.id).where(
                    timeline_versions.c.run_id == run_id,
                    timeline_versions.c.is_active.is_(False),
                    timeline_versions.c.status == "building",
                )
            ).scalar_one_or_none()
            staged_vector = connection.execute(
                select(vector_generations.c.id).where(
                    vector_generations.c.run_id == run_id,
                    vector_generations.c.is_active.is_(False),
                    vector_generations.c.status == "building",
                )
            ).scalar_one_or_none()
            if staged_timeline is None or staged_vector is None:
                raise InsightConflict(
                    "full analysis run is missing timeline or vector generation"
                )
            result_rows = {
                str(row["page_id_snapshot"]): row
                for row in connection.execute(
                    select(
                        analysis_page_results.c.id,
                        analysis_page_results.c.page_id_snapshot,
                    ).where(
                        analysis_page_results.c.run_id == run_id,
                        analysis_page_results.c.page_id_snapshot.in_(
                            [str(target["page_id_snapshot"]) for target in successful]
                        ),
                    )
                ).mappings()
            }
            for target in successful:
                page_id = str(target["page_id_snapshot"])
                result = result_rows.get(page_id)
                if result is None:
                    raise InsightConflict(
                        f"analysis result missing for successful page {page_id}"
                    )
                connection.execute(
                    update(analysis_page_results)
                    .where(analysis_page_results.c.id == result["id"])
                    .values(status="published", updated_at=now)
                )
                InsightRepository._upsert_page_head(
                    connection,
                    book_id=str(run["book_id"]),
                    page_id=page_id,
                    run_id=run_id,
                    result_id=str(result["id"]),
                    now=now,
                )
            InsightRepository._upsert_book_head(
                connection,
                book_id=str(run["book_id"]),
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
                    analysis_artifacts.c.book_id == run["book_id"],
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
                    timeline_versions.c.book_id == run["book_id"],
                    timeline_versions.c.is_active.is_(True),
                )
                .values(is_active=False, updated_at=now)
            )
            connection.execute(
                update(timeline_versions)
                .where(timeline_versions.c.id == staged_timeline)
                .values(
                    status=derived_status,
                    is_active=True,
                    updated_at=now,
                )
            )
            connection.execute(
                update(vector_generations)
                .where(
                    vector_generations.c.book_id == run["book_id"],
                    vector_generations.c.is_active.is_(True),
                )
                .values(is_active=False, updated_at=now)
            )
            connection.execute(
                update(vector_generations)
                .where(vector_generations.c.id == staged_vector)
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
        if str(run["scope"]) != "full":
            InsightRepository._mark_derived_stale(
                connection,
                book_id=str(run["book_id"]),
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
        missing = list(
            connection.execute(
                select(analysis_run_targets.c.page_id_snapshot).where(
                    analysis_run_targets.c.run_id == run_id,
                    analysis_run_targets.c.status != "completed",
                )
            ).scalars()
        )
        connection.execute(
            update(analysis_runs)
            .where(analysis_runs.c.id == run_id)
            .values(
                status="failed",
                updated_at=now,
                missing_page_ids_json=_json([str(value) for value in missing]),
            )
        )

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            run = connection.execute(
                select(analysis_runs).where(analysis_runs.c.id == run_id)
            ).mappings().one_or_none()
            if run is None:
                raise InsightNotFound("analysis run not found")
            targets = list(
                connection.execute(
                    select(analysis_run_targets)
                    .where(analysis_run_targets.c.run_id == run_id)
                    .order_by(analysis_run_targets.c.ordinal)
                ).mappings()
            )
        return {
            "runId": str(run["id"]),
            "jobId": run["job_id"],
            "bookId": str(run["book_id"]),
            "scope": str(run["scope"]),
            "status": str(run["status"]),
            "targetCount": int(run["target_count"]),
            "successCount": int(run["success_count"]),
            "failedCount": int(run["failed_count"]),
            "missingPageIds": _load(run["missing_page_ids_json"], []),
            "createdAt": _iso(run["created_at"]),
            "publishedAt": _iso(run["published_at"]),
            "targets": [
                {
                    "pageId": str(row["page_id_snapshot"]),
                    "pageNumber": int(row["page_number_snapshot"]),
                    "status": str(row["status"]),
                    "error": _load(row["error_json"], None),
                }
                for row in targets
            ],
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
                    .where(books.c.kind == "library")
                    .order_by(books.c.updated_at.desc(), books.c.title)
                ).mappings()
            )
            page_counts = dict(
                connection.execute(
                    select(chapters.c.book_id, func.count(pages.c.id))
                    .join(pages, pages.c.chapter_id == chapters.c.id)
                    .group_by(chapters.c.book_id)
                ).tuples().all()
            )
            head_counts = dict(
                connection.execute(
                    select(analysis_heads.c.book_id, func.count())
                    .where(analysis_heads.c.page_id.is_not(None))
                    .group_by(analysis_heads.c.book_id)
                ).tuples().all()
            )
            active_runs = {
                str(row["book_id"]): row
                for row in connection.execute(
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
            }
            active_jobs = [
                {
                    "jobId": str(row["id"]),
                    "bookId": row["book_id"],
                    "kind": str(row["kind"]),
                    "status": str(row["status"]),
                    "progress": _load(row["latest_progress_json"], {}),
                }
                for row in connection.execute(
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
                    )
                ).mappings()
            ]

        items = []
        for row in book_rows:
            book_id = str(row["id"])
            active = active_runs.get(book_id)
            items.append(
                {
                    "bookId": book_id,
                    "title": str(row["title"]),
                    "coverUrl": (
                        f"/api/v2/assets/{row['cover_asset_id']}"
                        if row["cover_asset_id"]
                        else None
                    ),
                    "pageCount": int(page_counts.get(book_id, 0)),
                    "analyzedPageCount": int(head_counts.get(book_id, 0)),
                    "activeRun": (
                        {
                            "runId": str(active["id"]),
                            "status": str(active["status"]),
                            "publishedAt": _iso(active["published_at"]),
                        }
                        if active
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
        return {"items": [str(template) for template in templates]}

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
            payload = _load(row["payload_json"], {})
            summary = (
                payload.get("page_summary")
                if isinstance(payload, Mapping)
                else None
            )
            items.append(
                {
                    "pageId": str(row["page_id"]),
                    "displayPageNumber": int(row["page_number"]),
                    "summary": summary if isinstance(summary, str) else None,
                    "generatedAt": _iso(row["created_at"]),
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
                & (page_rows.c.latest_target_status == "pending"),
                "running",
            ),
            (
                page_rows.c.latest_target_status.in_(
                    ("failed", "conflict")
                )
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
                ),
                "stale",
            ),
            (
                page_rows.c.book_run_id.is_not(None)
                & or_(
                    page_rows.c.page_run_id.is_(None),
                    page_rows.c.page_run_id != page_rows.c.book_run_id,
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
        return {
            "items": [
                {
                    "chapterId": str(row["id"]),
                    "title": str(row["title"]),
                    "ordinal": int(row["ordinal"]),
                    "pageCount": int(row["page_count"]),
                    "analysisCounts": {
                        state: int(row[state] or 0) for state in states
                    },
                }
                for row in rows
            ]
        }

    def list_pages(
        self,
        *,
        book_id: str,
        chapter_id: str | None,
        after: int,
        limit: int,
    ) -> dict[str, Any]:
        if after < 0:
            raise ValueError("cursor must be nonnegative")
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        with self.engine.connect() as connection:
            self._assert_book(connection, book_id)
            book_pages = self._book_page_statement(book_id).subquery()
            statement = select(book_pages)
            if chapter_id:
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
        items = [
            {
                "pageId": str(row["page_id"]),
                "chapterId": str(row["chapter_id"]),
                "displayPageNumber": int(row["page_number"]),
                "sourceAssetId": str(row["source_asset_id"]),
                "thumbnailUrl": (
                    f"/api/v2/assets/{row['thumbnail_asset_id']}"
                    if row["thumbnail_asset_id"]
                    else None
                ),
                "analysisState": self._state_for_row(row),
                "activeAnalysisId": row["active_result_id"],
            }
            for row in window
        ]
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
            page_number = self._page_number(
                connection,
                page_id=page_id,
                book_id=str(page["book_id"]),
            )
            preview = run_id is not None
            if run_id:
                result = connection.execute(
                    select(
                        analysis_page_results,
                        analysis_runs.c.status.label("run_status"),
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
                result = connection.execute(
                    select(
                        analysis_page_results,
                        analysis_runs.c.status.label("run_status"),
                        analysis_heads.c.active_run_id.label("head_run_id"),
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
            book_head_run = connection.execute(
                select(analysis_heads.c.active_run_id).where(
                    analysis_heads.c.book_id == page["book_id"],
                    analysis_heads.c.page_id.is_(None),
                )
            ).scalar_one_or_none()

        if result is None:
            analysis = None
            state = "not_analyzed"
            stale_reasons: list[str] = []
        else:
            analysis = _load(result["payload_json"], {})
            stale_reasons = []
            if str(result["source_checksum"]) != str(page["source_checksum"]):
                stale_reasons.append("source_changed")
            if (
                not preview
                and book_head_run is not None
                and str(result["run_id"]) != str(book_head_run)
            ):
                stale_reasons.append("fallback_from_previous_run")
            state = "stale" if stale_reasons else "ready"
        return {
            "pageId": page_id,
            "bookId": str(page["book_id"]),
            "chapterId": str(page["chapter_id"]),
            "chapterTitle": str(page["chapter_title"]),
            "displayPageNumber": page_number,
            "sourceAssetId": str(page["source_asset_id"]),
            "sourceUrl": f"/api/v2/assets/{page['source_asset_id']}",
            "analysisState": state,
            "staleReasons": stale_reasons,
            "preview": preview,
            "analysis": analysis,
            "runId": str(result["run_id"]) if result is not None else None,
            "generatedAt": _iso(result["created_at"]) if result is not None else None,
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
        if not 1 <= limit <= 200:
            raise ValueError("note limit must be between 1 and 200")
        if kind is not None and kind not in {"text", "qa"}:
            raise ValueError("note kind must be text or qa")
        cursor_value = _decode_note_cursor(cursor) if cursor else None
        with self.engine.connect() as connection:
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
                for citation in connection.execute(
                    select(note_citations)
                    .where(
                        note_citations.c.note_id.in_(
                            [str(row["id"]) for row in selected_rows]
                        )
                    )
                    .order_by(
                        note_citations.c.note_id,
                        note_citations.c.ordinal,
                    )
                ).mappings():
                    citations_by_note.setdefault(
                        str(citation["note_id"]),
                        [],
                    ).append(citation)
            items = [
                self._note_dto(
                    connection,
                    row,
                    summary=not include_content,
                    citations=citations_by_note.get(str(row["id"]), ()),
                )
                for row in selected_rows
            ]
        return {
            "items": items,
            "nextCursor": (
                _encode_note_cursor(
                    selected_rows[-1]["updated_at"],
                    str(selected_rows[-1]["id"]),
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
        book_id: str,
        title: str,
        content: str,
        citations: Sequence[Mapping[str, Any]] = (),
        kind: str = "text",
        tags: Sequence[str] = (),
        comments: Sequence[Mapping[str, Any] | str] = (),
    ) -> dict[str, Any]:
        title = title.strip()
        if not title or len(title) > 500:
            raise ValueError("note title must contain 1-500 characters")
        if len(content) > 1_000_000:
            raise ValueError("note content is too large")
        kind, tags, comments = _normalize_note_metadata(
            kind=kind,
            tags=tags,
            comments=comments,
        )
        note_id = str(uuid.uuid4())
        with immediate_transaction(self.engine) as connection:
            self._assert_book(connection, book_id)
            connection.execute(
                insert(notes).values(
                    id=note_id,
                    book_id=book_id,
                    title=title,
                    content=content,
                    kind=kind,
                    tags_json=_json(tags),
                    comments_json=_json(comments),
                )
            )
            self._replace_citations(
                connection,
                note_id=note_id,
                book_id=book_id,
                citations=citations,
            )
            row = connection.execute(
                select(notes).where(notes.c.id == note_id)
            ).mappings().one()
            return self._note_dto(connection, row)

    def update_note(
        self,
        *,
        note_id: str,
        base_revision: int,
        title: str,
        content: str,
        citations: Sequence[Mapping[str, Any]] = (),
        kind: str = "text",
        tags: Sequence[str] = (),
        comments: Sequence[Mapping[str, Any] | str] = (),
    ) -> dict[str, Any]:
        title = title.strip()
        if not title or len(title) > 500:
            raise ValueError("note title must contain 1-500 characters")
        if len(content) > 1_000_000:
            raise ValueError("note content is too large")
        kind, tags, comments = _normalize_note_metadata(
            kind=kind,
            tags=tags,
            comments=comments,
        )
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            current = connection.execute(
                select(notes.c.book_id, notes.c.revision).where(notes.c.id == note_id)
            ).mappings().one_or_none()
            if current is None:
                raise InsightNotFound("note not found")
            if int(current["revision"]) != base_revision:
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
                    comments_json=_json(comments),
                    revision=base_revision + 1,
                    updated_at=now,
                )
            )
            if changed.rowcount != 1:
                raise InsightConflict("note revision changed")
            self._replace_citations(
                connection,
                note_id=note_id,
                book_id=str(current["book_id"]),
                citations=citations,
            )
            row = connection.execute(
                select(notes).where(notes.c.id == note_id)
            ).mappings().one()
            return self._note_dto(connection, row)

    def delete_note(self, *, note_id: str, base_revision: int) -> None:
        with immediate_transaction(self.engine) as connection:
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
                book_head.c.active_run_id.label("book_run_id"),
                analysis_page_results.c.source_checksum.label(
                    "analysis_source_checksum"
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
        latest_job = row.get("latest_job_status")
        latest_target = row.get("latest_target_status")
        if latest_job in NONTERMINAL_JOB_STATUSES and latest_target == "pending":
            return "running"
        if latest_target in {"failed", "conflict"} and latest_job not in NONTERMINAL_JOB_STATUSES:
            return "failed"
        if row.get("active_result_id") is None:
            return "not_analyzed"
        if row.get("analysis_source_checksum") != row.get("source_checksum"):
            return "stale"
        if row.get("book_run_id") and row.get("page_run_id") != row.get("book_run_id"):
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
        counts = {
            str(status): int(count)
            for status, count in connection.execute(
                select(
                    analysis_run_targets.c.status,
                    func.count(),
                )
                .where(analysis_run_targets.c.run_id == run_id)
                .group_by(analysis_run_targets.c.status)
            )
        }
        connection.execute(
            update(analysis_runs)
            .where(analysis_runs.c.id == run_id)
            .values(
                success_count=counts.get("completed", 0),
                failed_count=counts.get("failed", 0)
                + counts.get("conflict", 0),
                updated_at=now,
            )
        )

    @staticmethod
    def _mark_derived_stale(
        connection: Connection,
        *,
        book_id: str,
        now: datetime,
    ) -> None:
        connection.execute(
            update(analysis_artifacts)
            .where(
                analysis_artifacts.c.book_id == book_id,
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

    @staticmethod
    def _assert_book(connection: Connection, book_id: str) -> None:
        if connection.execute(
            select(books.c.id).where(
                books.c.id == book_id,
                books.c.kind == "library",
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
        return int(page_number)

    @staticmethod
    def _replace_citations(
        connection: Connection,
        *,
        note_id: str,
        book_id: str,
        citations: Sequence[Mapping[str, Any]],
    ) -> None:
        normalized = [dict(value) for value in citations]
        page_ids = [
            str(value.get("pageId", ""))
            for value in normalized
        ]
        if any(not value for value in page_ids):
            raise ValueError("every citation requires pageId")
        if len(set(page_ids)) != len(page_ids):
            raise ValueError("citation pageIds must be unique")
        connection.execute(
            delete(note_citations).where(note_citations.c.note_id == note_id)
        )
        if not page_ids:
            return
        book_pages = InsightRepository._numbered_book_pages_statement(
            book_id
        ).subquery()
        page_numbers = {
            str(row["page_id"]): int(row["page_number"])
            for row in connection.execute(
                select(book_pages.c.page_id, book_pages.c.page_number).where(
                    book_pages.c.page_id.in_(page_ids)
                )
            ).mappings()
        }
        if not set(page_ids).issubset(page_numbers):
            raise ValueError("all citation pages must belong to the note book")
        active_results = {
            str(row["page_id"]): row["active_result_id"]
            for row in connection.execute(
                select(
                    analysis_heads.c.page_id,
                    analysis_heads.c.active_result_id,
                ).where(analysis_heads.c.page_id.in_(page_ids))
            ).mappings()
        }
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
                    "excerpt": str(
                        normalized[ordinal - 1].get("excerpt", "")
                    )[:2000],
                    "score": (
                        float(normalized[ordinal - 1]["score"])
                        if normalized[ordinal - 1].get("score") is not None
                        else None
                    ),
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
        comments = _load(row["comments_json"], [])
        return {
            "noteId": str(row["id"]),
            "bookId": str(row["book_id"]),
            "title": str(row["title"]),
            "content": (
                None if summary else str(row["content"])
            ),
            "excerpt": (
                str(row["content"])[:300] if summary else None
            ),
            "kind": str(row["kind"]),
            "tags": _load(row["tags_json"], []),
            "comments": (
                []
                if summary
                else comments
            ),
            "commentCount": len(comments),
            "revision": int(row["revision"]),
            "citations": [
                {
                    "pageId": citation["page_id"],
                    "pageIdSnapshot": str(citation["page_id_snapshot"]),
                    "pageNumberSnapshot": int(
                        citation["page_number_snapshot"]
                    ),
                    "sourceAnalysisId": citation["source_analysis_id"],
                    "excerpt": str(citation["excerpt"]),
                    "score": citation["score"],
                }
                for citation in citations
            ],
            "createdAt": _iso(row["created_at"]),
            "updatedAt": _iso(row["updated_at"]),
        }
