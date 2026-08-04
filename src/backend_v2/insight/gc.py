"""Reachability based garbage collection for Insight generations.

Insight publication deliberately keeps old runs isolated until a new head is
committed.  This collector is the other half of that protocol: it removes only
records which are no longer reachable from a published head, retained job,
continuation project, note citation, or live derived generation.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import Engine, delete, or_, select
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


class InsightReachabilityGarbageCollector:
    """Delete bounded batches of unreachable Insight database generations."""

    def __init__(
        self,
        engine: Engine,
        *,
        grace: timedelta = DEFAULT_INSIGHT_GC_GRACE,
        limit: int = DEFAULT_INSIGHT_GC_LIMIT,
    ) -> None:
        if grace.total_seconds() < 0:
            raise ValueError("Insight GC grace must not be negative")
        if limit < 1:
            raise ValueError("Insight GC limit must be positive")
        self.engine = engine
        self.grace = grace
        self.limit = limit

    def collect(self, *, now: datetime | None = None) -> dict[str, int]:
        cutoff = (now or utcnow()) - self.grace
        with immediate_transaction(self.engine) as connection:
            retained_job_runs = self._retained_job_run_ids(connection)

            # Inactive generations are not roots merely because they point to
            # the same analysis run as the current generation.  Retained jobs
            # are the exception: their partial previews must remain available
            # for the lifetime of the task history row.
            artifacts = self._delete_inactive_generations(
                connection,
                table=analysis_artifacts,
                cutoff=cutoff,
                retained_job_runs=retained_job_runs,
            )
            timelines = self._delete_inactive_generations(
                connection,
                table=timeline_versions,
                cutoff=cutoff,
                retained_job_runs=retained_job_runs,
            )
            vectors = self._delete_inactive_generations(
                connection,
                table=vector_generations,
                cutoff=cutoff,
                retained_job_runs=retained_job_runs,
            )

            reachable_runs = self._reachable_run_ids(
                connection,
                retained_job_runs=retained_job_runs,
            )
            conditions = [analysis_runs.c.updated_at <= cutoff]
            if reachable_runs:
                conditions.append(analysis_runs.c.id.not_in(reachable_runs))
            run_ids = [
                str(value)
                for value in connection.execute(
                    select(analysis_runs.c.id)
                    .where(*conditions)
                    .order_by(analysis_runs.c.updated_at, analysis_runs.c.id)
                    .limit(self.limit)
                ).scalars()
            ]
            if run_ids:
                connection.execute(
                    delete(analysis_runs).where(analysis_runs.c.id.in_(run_ids))
                )

        return {
            "analysisArtifacts": artifacts,
            "timelineVersions": timelines,
            "vectorGenerations": vectors,
            "analysisRuns": len(run_ids),
        }

    def _delete_inactive_generations(
        self,
        connection: Any,
        *,
        table: Table,
        cutoff: datetime,
        retained_job_runs: set[str],
    ) -> int:
        conditions = [
            table.c.is_active.is_(False),
            table.c.updated_at <= cutoff,
        ]
        if retained_job_runs:
            conditions.append(
                or_(
                    table.c.run_id.is_(None),
                    table.c.run_id.not_in(retained_job_runs),
                )
            )
        generation_ids = [
            str(value)
            for value in connection.execute(
                select(table.c.id)
                .where(*conditions)
                .order_by(table.c.updated_at, table.c.id)
                .limit(self.limit)
            ).scalars()
        ]
        if generation_ids:
            connection.execute(delete(table).where(table.c.id.in_(generation_ids)))
        return len(generation_ids)

    @staticmethod
    def _retained_job_run_ids(connection: Any) -> set[str]:
        run_ids = {
            str(value)
            for value in connection.execute(
                select(analysis_runs.c.id).where(
                    analysis_runs.c.job_id.is_not(None)
                )
            ).scalars()
        }
        run_ids.update(
            str(value)
            for value in connection.execute(
                select(jobs.c.analysis_run_id).where(
                    jobs.c.analysis_run_id.is_not(None)
                )
            ).scalars()
        )
        return run_ids

    @staticmethod
    def _reachable_run_ids(
        connection: Any,
        *,
        retained_job_runs: set[str],
    ) -> set[str]:
        run_ids = set(retained_job_runs)
        run_ids.update(
            str(value)
            for value in connection.execute(
                select(analysis_heads.c.active_run_id)
            ).scalars()
        )
        run_ids.update(
            str(value)
            for value in connection.execute(
                select(continuation_projects.c.source_run_id).where(
                    continuation_projects.c.source_run_id.is_not(None)
                )
            ).scalars()
        )
        run_ids.update(
            str(value)
            for value in connection.execute(
                select(analysis_page_results.c.run_id)
                .join(
                    note_citations,
                    note_citations.c.source_analysis_id
                    == analysis_page_results.c.id,
                )
                .where(note_citations.c.source_analysis_id.is_not(None))
            ).scalars()
        )
        # All generations which survived the first pass temporarily root their
        # source run.  This includes the one-hour publication grace window and
        # prevents a recent generation from being detached by run deletion.
        for table in (
            analysis_artifacts,
            timeline_versions,
            vector_generations,
        ):
            run_ids.update(
                str(value)
                for value in connection.execute(
                    select(table.c.run_id).where(table.c.run_id.is_not(None))
                ).scalars()
            )
        return run_ids
