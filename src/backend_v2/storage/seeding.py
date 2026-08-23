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
    DEFAULT_CUSTOM_AI_PROFILES,
    DEFAULT_EXPORT_PREFERENCES,
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
    DEFAULT_ASSET_QUOTA_BYTES,
    DEFAULT_PUBLIC_USER_POLICY_JSON,
    DEFAULT_SCHEDULING_POLICY_JSON,
    app_settings,
    books,
    chapters,
    fonts,
    plugins,
    platform_config,
    prompts,
    queue_state,
    translation_constraints,
    users,
)
from src.backend_v2.auth.constants import LOCAL_USER_ID, LOCAL_USERNAME
from src.backend_v2.runtime_profile import PROFILE_NAMES


QUICK_WORKSPACE_BOOK_ID = "00000000-0000-0000-0000-000000000001"
QUICK_WORKSPACE_CHAPTER_ID = "00000000-0000-0000-0000-000000000002"


def seed_system_records(engine: Engine, *, profile_name: str = "local") -> None:
    if profile_name not in PROFILE_NAMES:
        raise ValueError(f"unsupported runtime profile: {profile_name!r}")
    with engine.begin() as connection:
        if profile_name == "local":
            if connection.execute(
                select(users.c.id).where(users.c.id == LOCAL_USER_ID)
            ).scalar_one_or_none() is None:
                connection.execute(
                    insert(users).values(
                        id=LOCAL_USER_ID,
                        username=LOCAL_USERNAME,
                        password_hash=None,
                        role="admin",
                        status="active",
                    )
                )
        if connection.execute(
            select(platform_config.c.singleton_id)
        ).scalar_one_or_none() is None:
            connection.execute(
                insert(platform_config).values(
                    singleton_id=1,
                    registration_requires_invite=True,
                    asset_quota_bytes=DEFAULT_ASSET_QUOTA_BYTES,
                    public_user_policy_json=DEFAULT_PUBLIC_USER_POLICY_JSON,
                    scheduler_policy_json=DEFAULT_SCHEDULING_POLICY_JSON,
                )
            )

        if profile_name == "local":
            seed_user_records_in_connection(connection, LOCAL_USER_ID)

            # Runtime enablement exists only in the local profile.  Reset it
            # once before the local API and Worker are spawned.
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

        _seed_shared_records(connection)

        if connection.execute(select(queue_state.c.singleton_id)).scalar_one_or_none() is None:
            connection.execute(insert(queue_state).values(singleton_id=1))


def seed_user_records_in_connection(connection: object, user_id: str) -> None:
        quick_book_id = connection.execute(
            select(books.c.id).where(
                books.c.owner_user_id == user_id,
                books.c.kind == "quick_workspace",
            )
        ).scalar_one_or_none()
        if quick_book_id is None:
            quick_book_id = (
                QUICK_WORKSPACE_BOOK_ID
                if user_id == LOCAL_USER_ID
                else str(uuid.uuid4())
            )
            connection.execute(
                insert(books).values(
                    id=quick_book_id,
                    owner_user_id=user_id,
                    kind="quick_workspace",
                    title="快速翻译",
                )
            )

        quick_chapter_id = connection.execute(
            select(chapters.c.id)
            .where(chapters.c.book_id == quick_book_id)
            .order_by(chapters.c.ordinal)
            .limit(1)
        ).scalar_one_or_none()
        if quick_chapter_id is None:
            quick_chapter_id = (
                QUICK_WORKSPACE_CHAPTER_ID
                if user_id == LOCAL_USER_ID
                else str(uuid.uuid4())
            )
            connection.execute(
                insert(chapters).values(
                    id=quick_chapter_id,
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
            "export_preferences": DEFAULT_EXPORT_PREFERENCES,
            "custom_ai_profiles": DEFAULT_CUSTOM_AI_PROFILES,
            "text_style_defaults": DEFAULT_TEXT_STYLE,
            "insight": DEFAULT_INSIGHT_SETTINGS,
            "web_import": DEFAULT_WEB_IMPORT_SETTINGS,
        }
        default_schema_versions = {
            "translation": TRANSLATION_SETTINGS_SCHEMA_VERSION,
            "text_style_defaults": TEXT_STYLE_DEFAULTS_SCHEMA_VERSION,
        }
        existing_domains = set(
            connection.execute(
                select(app_settings.c.domain).where(
                    app_settings.c.owner_user_id == user_id
                )
            ).scalars()
        )
        for domain, payload in default_domains.items():
            if domain not in existing_domains:
                connection.execute(
                    insert(app_settings).values(
                        owner_user_id=user_id,
                        domain=domain,
                        payload_json=canonical_json(payload),
                        schema_version=default_schema_versions.get(domain, 1),
                    )
                )
        existing_factory_types = set(
            connection.execute(
                select(prompts.c.type).where(
                    prompts.c.owner_user_id == user_id,
                    prompts.c.is_factory_default.is_(True),
                )
            ).scalars()
        )
        for prompt_type, content in FACTORY_PROMPTS.items():
            if prompt_type not in existing_factory_types:
                connection.execute(
                    insert(prompts).values(
                        id=str(uuid.uuid4()),
                        owner_user_id=user_id,
                        type=prompt_type,
                        name="默认提示词",
                        content=content,
                        is_factory_default=True,
                    )
                )


def _seed_shared_records(connection: object) -> None:
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
