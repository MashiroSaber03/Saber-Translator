"""Idempotent system records required by every v2 database."""

from __future__ import annotations

import uuid

from sqlalchemy import Engine, case, insert, select, update

from src.backend_v2.serialization import canonical_json
from src.backend_v2.storage.builtin_fonts import discover_bundled_fonts
from src.backend_v2.content.translation_constraints import (
    empty_translation_constraints,
)
from src.backend_v2.storage.defaults import (
    DEFAULT_INSIGHT_SETTINGS,
    DEFAULT_TEXT_STYLE,
    DEFAULT_WEB_IMPORT_SETTINGS,
    DEFAULT_WORKFLOW_PREFERENCES,
    FACTORY_PROMPTS,
    TEXT_STYLE_DEFAULTS_SCHEMA_VERSION,
    TRANSLATION_SETTINGS_SCHEMA_VERSION,
    default_translation_settings,
)
from src.backend_v2.storage.schema import (
    app_settings,
    books,
    chapters,
    fonts,
    plugins,
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
                    payload_json=canonical_json(empty_translation_constraints()),
                )
            )

        default_domains = {
            "translation": default_translation_settings(),
            "workflow_preferences": DEFAULT_WORKFLOW_PREFERENCES,
            "text_style_defaults": DEFAULT_TEXT_STYLE,
            "insight": DEFAULT_INSIGHT_SETTINGS,
            "web_import": DEFAULT_WEB_IMPORT_SETTINGS,
        }
        default_schema_versions = {
            "translation": TRANSLATION_SETTINGS_SCHEMA_VERSION,
            "text_style_defaults": TEXT_STYLE_DEFAULTS_SCHEMA_VERSION,
        }
        existing_domains = set(
            connection.execute(select(app_settings.c.domain)).scalars()
        )
        for domain, payload in default_domains.items():
            if domain not in existing_domains:
                connection.execute(
                    insert(app_settings).values(
                        domain=domain,
                        payload_json=canonical_json(payload),
                        schema_version=default_schema_versions.get(domain, 1),
                    )
                )

        # Runtime enablement is a process-lifetime override.  The Launcher is
        # the sole initialization/seeding owner, so reset it once before API and
        # Worker are spawned; an API child restart must not rewrite this state.
        connection.execute(
            update(plugins).values(
                runtime_enabled=plugins.c.default_enabled,
                state=case(
                    (plugins.c.state == "error", "error"),
                    (
                        plugins.c.default_enabled.is_(True),
                        "enabled",
                    ),
                    else_="disabled",
                ),
            )
        )

        bundled_fonts = {
            font.builtin_key: font for font in discover_bundled_fonts()
        }
        existing_builtin_fonts = {
            str(row["builtin_key"]): row
            for row in connection.execute(
                select(
                    fonts.c.id,
                    fonts.c.builtin_key,
                    fonts.c.display_name,
                ).where(fonts.c.kind == "builtin")
            ).mappings()
        }
        unexpected_builtin_keys = existing_builtin_fonts.keys() - bundled_fonts.keys()
        if unexpected_builtin_keys:
            raise RuntimeError(
                "bundled font catalog contains unsupported keys: "
                f"{sorted(unexpected_builtin_keys)}"
            )
        for bundled_font in bundled_fonts.values():
            existing = existing_builtin_fonts.get(bundled_font.builtin_key)
            if existing is None:
                connection.execute(
                    insert(fonts).values(
                        id=bundled_font.id,
                        kind="builtin",
                        display_name=bundled_font.display_name,
                        builtin_key=bundled_font.builtin_key,
                    )
                )
                continue
            if str(existing["id"]) != bundled_font.id:
                raise RuntimeError(
                    "bundled font catalog id mismatch for "
                    f"{bundled_font.builtin_key}"
                )
            if str(existing["display_name"]) != bundled_font.display_name:
                raise RuntimeError(
                    "bundled font catalog display name mismatch for "
                    f"{bundled_font.builtin_key}"
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
