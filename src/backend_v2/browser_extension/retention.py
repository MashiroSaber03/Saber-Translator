"""Lifecycle cleanup for temporary browser pages."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, delete, select

from src.backend_v2.jobs.repository import (
    InvalidJobTransition,
    JobNotFound,
    JobQueueRepository,
)
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.database import immediate_transaction
from src.backend_v2.storage.schema import (
    NONTERMINAL_JOB_STATUSES,
    books,
    browser_sessions,
    browser_session_pages,
    chapters,
    pages,
    page_assets,
    job_asset_inputs,
    job_artifacts,
    jobs,
    job_steps,
    job_items,
    job_step_asset_outputs,
)
from src.backend_v2.timestamps import utcnow


def cleanup_expired_browser_sessions(
    engine: Engine, *, data_root: Path | None = None,
) -> int:
    """Remove expired sessions that no active job still uses."""

    now = utcnow()
    # A browser crash cannot deliver its close message. Expired leases also
    # cancel outstanding work, rather than keeping abandoned jobs alive forever.
    with engine.connect() as connection:
        expired_jobs = list(
            connection.execute(
                select(jobs.c.id).where(
                    jobs.c.book_id.in_(
                        select(browser_sessions.c.book_id).where(
                            browser_sessions.c.expires_at <= now
                        )
                    ),
                    jobs.c.status.in_(NONTERMINAL_JOB_STATUSES),
                )
            ).scalars()
        )
    repository = JobQueueRepository(engine)
    for job_id in expired_jobs:
        try:
            repository.request_cancel(str(job_id))
        except (InvalidJobTransition, JobNotFound):
            pass  # The worker may have finished since the expired lease was read.
    with immediate_transaction(engine) as connection:
        book_ids = list(
            connection.execute(
                select(browser_sessions.c.book_id).where(
                    browser_sessions.c.expires_at <= now,
                    ~browser_sessions.c.book_id.in_(
                        select(jobs.c.book_id).where(
                            jobs.c.book_id.is_not(None),
                            jobs.c.status.in_(NONTERMINAL_JOB_STATUSES),
                        )
                    ),
                )
            ).scalars()
        )
        if not book_ids:
            return 0
        job_ids = set(
            connection.execute(
                select(jobs.c.id).where(jobs.c.book_id.in_(book_ids))
            ).scalars()
        )
        session_ids = select(browser_sessions.c.id).where(
            browser_sessions.c.book_id.in_(book_ids)
        )
        page_ids = (
            select(pages.c.id)
            .join(chapters, pages.c.chapter_id == chapters.c.id)
            .where(chapters.c.book_id.in_(book_ids))
        )
        asset_ids = set(
            connection.execute(
                select(browser_session_pages.c.source_asset_id)
                .where(browser_session_pages.c.session_id.in_(session_ids))
                .union(
                    select(browser_session_pages.c.thumbnail_asset_id).where(
                        browser_session_pages.c.session_id.in_(session_ids)
                    ),
                    select(page_assets.c.asset_id).where(
                        page_assets.c.page_id.in_(page_ids)
                    ),
                    select(job_asset_inputs.c.asset_id).where(
                        job_asset_inputs.c.job_id.in_(job_ids)
                    ),
                    select(job_artifacts.c.asset_id).where(
                        job_artifacts.c.job_id.in_(job_ids)
                    ),
                    select(job_step_asset_outputs.c.asset_id)
                    .join(job_steps, job_step_asset_outputs.c.job_step_id == job_steps.c.id)
                    .join(job_items, job_steps.c.job_item_id == job_items.c.id)
                    .where(job_items.c.job_id.in_(job_ids)),
                )
            ).scalars()
        )
        JobQueueRepository.delete_history_jobs(connection, candidates=job_ids, now=now)
        connection.execute(delete(books).where(books.c.id.in_(book_ids)))
    if data_root is not None and asset_ids:
        storage = AssetStorageService(data_root, engine)
        # Only this retired page's assets: mark, then remove unreferenced files.
        storage.collect_garbage(
            grace_seconds=0, asset_ids=asset_ids, batch_limit=len(asset_ids),
        )
        storage.collect_garbage(
            grace_seconds=0, asset_ids=asset_ids, batch_limit=len(asset_ids),
        )
    return len(book_ids)
