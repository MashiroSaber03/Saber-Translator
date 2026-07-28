"""Idempotent system records required by every v2 database."""

from __future__ import annotations

import json
import uuid

from sqlalchemy import Engine, insert, select

from src.backend_v2.storage.defaults import (
    DEFAULT_FONT_ID,
    DEFAULT_TEXT_STYLE,
    DEFAULT_WORKFLOW_PREFERENCES,
    FACTORY_PROMPTS,
)
from src.backend_v2.storage.schema import (
    app_settings,
    books,
    chapters,
    fonts,
    prompts,
    queue_state,
    translation_constraints,
)


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

        if connection.execute(
            select(translation_constraints.c.book_id).where(
                translation_constraints.c.book_id == quick_book_id
            )
        ).scalar_one_or_none() is None:
            connection.execute(
                insert(translation_constraints).values(
                    book_id=quick_book_id,
                    payload_json='{"glossary":[],"nonTranslate":[]}',
                )
            )

        default_domains = {
            "translation": {},
            "detection": {},
            "ocr": {},
            "inpainting": {},
            "rendering": {},
            "workflow_preferences": DEFAULT_WORKFLOW_PREFERENCES,
            "text_style_defaults": DEFAULT_TEXT_STYLE,
            "insight": {},
            "web_import": {},
        }
        existing_domains = set(
            connection.execute(select(app_settings.c.domain)).scalars()
        )
        for domain, payload in default_domains.items():
            if domain not in existing_domains:
                connection.execute(
                    insert(app_settings).values(
                        domain=domain,
                        payload_json=json.dumps(
                            payload,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    )
                )

        if connection.execute(
            select(fonts.c.id).where(fonts.c.builtin_key == "default")
        ).scalar_one_or_none() is None:
            connection.execute(
                insert(fonts).values(
                    id=DEFAULT_FONT_ID,
                    kind="builtin",
                    display_name="默认字体",
                    builtin_key="default",
                )
            )

        existing_factory_types = set(
            connection.execute(
                select(prompts.c.type).where(
                    prompts.c.is_factory_default.is_(True)
                )
            ).scalars()
        )
        for prompt_type, content in FACTORY_PROMPTS.items():
            if prompt_type not in existing_factory_types:
                connection.execute(
                    insert(prompts).values(
                        id=str(uuid.uuid4()),
                        type=prompt_type,
                        name="默认提示词",
                        content=content,
                        is_factory_default=True,
                    )
                )

        if connection.execute(select(queue_state.c.singleton_id)).scalar_one_or_none() is None:
            connection.execute(insert(queue_state).values(singleton_id=1))
