"""Retention cleanup shared by the local API and worker maintenance."""

from __future__ import annotations

from sqlalchemy import Engine, delete, select

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
        expired = list(
            connection.execute(
                select(browser_sessions.c.id, browser_sessions.c.book_id).where(
                    browser_sessions.c.expires_at <= now,
                    ~browser_sessions.c.chapter_id.in_(
                        select(jobs.c.chapter_id).where(
                            jobs.c.chapter_id.is_not(None),
                            jobs.c.status.in_(NONTERMINAL_JOB_STATUSES),
                        )
                    ),
                )
            ).mappings()
        )
        if expired:
            book_ids = [row["book_id"] for row in expired]
            connection.execute(delete(jobs).where(jobs.c.book_id.in_(book_ids)))
            connection.execute(delete(books).where(books.c.id.in_(book_ids)))
    return len(expired)
