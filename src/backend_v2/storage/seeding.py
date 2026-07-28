"""Idempotent system records required by every v2 database."""

from __future__ import annotations

from sqlalchemy import Engine, insert, select

from src.backend_v2.storage.schema import books, chapters, queue_state


QUICK_WORKSPACE_BOOK_ID = "00000000-0000-0000-0000-000000000001"
QUICK_WORKSPACE_CHAPTER_ID = "00000000-0000-0000-0000-000000000002"


def seed_system_records(engine: Engine) -> None:
    with engine.begin() as connection:
        quick_book_id = connection.execute(
            select(books.c.id).where(books.c.kind == "quick_workspace")
        ).scalar_one_or_none()
        if quick_book_id is None:
            connection.execute(
                insert(books).values(
                    id=QUICK_WORKSPACE_BOOK_ID,
                    kind="quick_workspace",
                    title="快速翻译",
                )
            )
            quick_book_id = QUICK_WORKSPACE_BOOK_ID

        quick_chapter_id = connection.execute(
            select(chapters.c.id)
            .where(chapters.c.book_id == quick_book_id)
            .order_by(chapters.c.ordinal)
            .limit(1)
        ).scalar_one_or_none()
        if quick_chapter_id is None:
            connection.execute(
                insert(chapters).values(
                    id=QUICK_WORKSPACE_CHAPTER_ID,
                    book_id=quick_book_id,
                    ordinal=1,
                    title="快速翻译",
                )
            )

        if connection.execute(select(queue_state.c.singleton_id)).scalar_one_or_none() is None:
            connection.execute(insert(queue_state).values(singleton_id=1))
