"""Worker-owned retention cleanup for browser sessions."""

from __future__ import annotations

from sqlalchemy import Engine, delete, select

from src.backend_v2.jobs.repository import JobQueueRepository
from src.backend_v2.storage.database import immediate_transaction
from src.backend_v2.storage.schema import (
    NONTERMINAL_JOB_STATUSES,
    books,
    browser_sessions,
    jobs,
)
from src.backend_v2.timestamps import utcnow


def cleanup_expired_browser_sessions(engine: Engine) -> int:
    """Remove expired sessions that no active job still uses."""

    now = utcnow()
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
        JobQueueRepository.delete_history_jobs(connection, candidates=job_ids, now=now)
        connection.execute(delete(books).where(books.c.id.in_(book_ids)))
    return len(book_ids)
