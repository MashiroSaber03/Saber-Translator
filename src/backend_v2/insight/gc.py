"""Reachability based garbage collection for Insight generations.

Insight publication deliberately keeps old runs isolated until a new head is
committed.  This collector is the other half of that protocol: it removes only
records which are no longer reachable from a published head, retained job,
continuation project, note citation, or live derived generation.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from sqlalchemy import Engine, delete, or_, select
from sqlalchemy.engine import Connection
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.schema import Table

from src.backend_v2.storage.database import immediate_transaction
from src.backend_v2.storage.schema import (
    analysis_artifacts,
    analysis_heads,
    analysis_page_results,
    analysis_runs,
    continuation_projects,
    jobs,
    note_citations,
    timeline_versions,
    vector_generations,
)
from src.backend_v2.timestamps import utcnow


DEFAULT_INSIGHT_GC_GRACE = timedelta(hours=1)
DEFAULT_INSIGHT_GC_LIMIT = 200


def _stored_id(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(
            f"stored {field} is invalid; clear current Insight data"
        )
    return value


class InsightReachabilityGarbageCollector:
    """Delete bounded batches of unreachable Insight database generations."""

    def __init__(
        self,
        engine: Engine,
        *,
        grace: timedelta = DEFAULT_INSIGHT_GC_GRACE,
        limit: int = DEFAULT_INSIGHT_GC_LIMIT,
    ) -> None:
        if not isinstance(grace, timedelta):
            raise TypeError("Insight GC grace must be a timedelta")
        if grace.total_seconds() < 0:
            raise ValueError("Insight GC grace must not be negative")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("Insight GC limit must be positive")
        self.engine = engine
        self.grace = grace
        self.limit = limit

    def collect(self, *, now: datetime | None = None) -> dict[str, int]:
        if now is not None and not isinstance(now, datetime):
            raise TypeError("Insight GC now must be a datetime")
        cutoff = (now or utcnow()) - self.grace
        with immediate_transaction(self.engine) as connection:
            # Inactive generations are not roots merely because they point to
            # the same analysis run as the current generation.  Retained jobs
            # are the exception: their partial previews must remain available
            # for the lifetime of the task history row.
            artifacts = self._delete_inactive_generations(
                connection,
                table=analysis_artifacts,
                cutoff=cutoff,
            )
            timelines = self._delete_inactive_generations(
                connection,
                table=timeline_versions,
                cutoff=cutoff,
            )
            vectors = self._delete_inactive_generations(
                connection,
                table=vector_generations,
                cutoff=cutoff,
            )

            reachable = self._run_is_reachable()
            run_ids = [
                _stored_id(value, "analysis run id")
                for value in connection.execute(
                    select(analysis_runs.c.id)
                    .where(
                        analysis_runs.c.updated_at <= cutoff,
                        ~reachable,
                    )
                    .order_by(analysis_runs.c.updated_at, analysis_runs.c.id)
                    .limit(self.limit)
                ).scalars()
            ]
            if run_ids:
                deleted = connection.execute(
                    delete(analysis_runs).where(analysis_runs.c.id.in_(run_ids))
                )
                if deleted.rowcount != len(run_ids):
                    raise RuntimeError("Insight run garbage collection was fenced")

        return {
            "analysisArtifacts": artifacts,
            "timelineVersions": timelines,
            "vectorGenerations": vectors,
            "analysisRuns": len(run_ids),
        }

    def _delete_inactive_generations(
        self,
        connection: Connection,
        *,
        table: Table,
        cutoff: datetime,
    ) -> int:
        run = analysis_runs.alias(f"{table.name}_retained_run")
        retained_by_run_job = (
            select(run.c.id)
            .where(
                run.c.id == table.c.run_id,
                run.c.job_id.is_not(None),
            )
            .correlate(table)
            .exists()
        )
        retained_by_job_projection = (
            select(jobs.c.id)
            .where(jobs.c.analysis_run_id == table.c.run_id)
            .correlate(table)
            .exists()
        )
        generation_ids = [
            _stored_id(value, f"{table.name} id")
            for value in connection.execute(
                select(table.c.id)
                .where(
                    table.c.is_active.is_(False),
                    table.c.updated_at <= cutoff,
                    ~or_(retained_by_run_job, retained_by_job_projection),
                )
                .order_by(table.c.updated_at, table.c.id)
                .limit(self.limit)
            ).scalars()
        ]
        if generation_ids:
            deleted = connection.execute(
                delete(table).where(table.c.id.in_(generation_ids))
            )
            if deleted.rowcount != len(generation_ids):
                raise RuntimeError(
                    f"{table.name} garbage collection was fenced"
                )
        return len(generation_ids)

    @staticmethod
    def _run_is_reachable() -> ColumnElement[bool]:
        retained_by_job_projection = (
            select(jobs.c.id)
            .where(jobs.c.analysis_run_id == analysis_runs.c.id)
            .correlate(analysis_runs)
            .exists()
        )
        published_head = (
            select(analysis_heads.c.id)
            .where(analysis_heads.c.active_run_id == analysis_runs.c.id)
            .correlate(analysis_runs)
            .exists()
        )
        continuation_source = (
            select(continuation_projects.c.id)
            .where(continuation_projects.c.source_run_id == analysis_runs.c.id)
            .correlate(analysis_runs)
            .exists()
        )
        cited_page_result = (
            select(note_citations.c.note_id)
            .select_from(
                note_citations.join(
                    analysis_page_results,
                    analysis_page_results.c.id
                    == note_citations.c.source_analysis_id,
                )
            )
            .where(analysis_page_results.c.run_id == analysis_runs.c.id)
            .correlate(analysis_runs)
            .exists()
        )
        surviving_generations = [
            select(table.c.id)
            .where(table.c.run_id == analysis_runs.c.id)
            .correlate(analysis_runs)
            .exists()
            for table in (
                analysis_artifacts,
                timeline_versions,
                vector_generations,
            )
        ]
        # Generations surviving the first pass include active rows and rows in
        # the grace window.  They temporarily root their source run, preventing
        # the bounded cleanup from detaching live or freshly-published data.
        return or_(
            analysis_runs.c.job_id.is_not(None),
            retained_by_job_projection,
            published_head,
            continuation_source,
            cited_page_result,
            *surviving_generations,
        )
