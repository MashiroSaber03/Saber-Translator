"""Public-profile resource ownership checks for route identifiers."""

from __future__ import annotations

from flask import Flask, Response, jsonify, request
from sqlalchemy import Engine, literal, or_, select, union_all

from src.backend_v2.auth.ownership import effective_owner_id
from src.backend_v2.runtime_profile import RuntimeProfile
from src.backend_v2.storage.schema import (
    analysis_runs,
    assets,
    books,
    chapters,
    continuation_character_forms,
    continuation_characters,
    continuation_pages,
    continuation_projects,
    credentials,
    fonts,
    job_batches,
    jobs,
    notes,
    operations,
    pages,
    prompts,
    studio_chat_sessions,
    studio_documents,
    studio_messages,
    tags,
)


def _owner_query(argument_name: str, resource_id: str):
    if argument_name == "font_id":
        owner_id = effective_owner_id()
        return select(literal(owner_id)).where(
            fonts.c.id == resource_id,
            or_(fonts.c.kind == "builtin", fonts.c.owner_user_id == owner_id),
        )
    direct = {
        "book_id": books,
        "asset_id": assets,
        "tag_id": tags,
        "job_id": jobs,
        "batch_id": job_batches,
        "operation_id": operations,
        "document_id": studio_documents,
        "run_id": analysis_runs,
        "note_id": notes,
        "project_id": continuation_projects,
        "credential_id": credentials,
        "prompt_id": prompts,
    }
    table = direct.get(argument_name)
    if table is not None:
        return select(table.c.owner_user_id).where(table.c.id == resource_id)
    if argument_name == "chapter_id":
        return (
            select(books.c.owner_user_id)
            .select_from(chapters.join(books, books.c.id == chapters.c.book_id))
            .where(chapters.c.id == resource_id)
        )
    if argument_name == "page_id":
        content_owner = (
            select(books.c.owner_user_id)
            .select_from(
                pages.join(chapters, chapters.c.id == pages.c.chapter_id).join(
                    books, books.c.id == chapters.c.book_id
                )
            )
            .where(pages.c.id == resource_id)
        )
        continuation_owner = (
            select(continuation_projects.c.owner_user_id)
            .select_from(
                continuation_pages.join(
                    continuation_projects,
                    continuation_projects.c.id == continuation_pages.c.project_id,
                )
            )
            .where(continuation_pages.c.id == resource_id)
        )
        return union_all(content_owner, continuation_owner)
    if argument_name == "character_id":
        return (
            select(continuation_projects.c.owner_user_id)
            .select_from(
                continuation_characters.join(
                    continuation_projects,
                    continuation_projects.c.id == continuation_characters.c.project_id,
                )
            )
            .where(continuation_characters.c.id == resource_id)
        )
    if argument_name == "form_id":
        return (
            select(continuation_projects.c.owner_user_id)
            .select_from(
                continuation_character_forms.join(
                    continuation_characters,
                    continuation_characters.c.id
                    == continuation_character_forms.c.character_id,
                ).join(
                    continuation_projects,
                    continuation_projects.c.id == continuation_characters.c.project_id,
                )
            )
            .where(continuation_character_forms.c.id == resource_id)
        )
    if argument_name == "session_id":
        return (
            select(studio_documents.c.owner_user_id)
            .select_from(
                studio_chat_sessions.join(
                    studio_documents,
                    studio_documents.c.id == studio_chat_sessions.c.document_id,
                )
            )
            .where(studio_chat_sessions.c.id == resource_id)
        )
    if argument_name == "message_id":
        return (
            select(studio_documents.c.owner_user_id)
            .select_from(
                studio_messages.join(
                    studio_chat_sessions,
                    studio_chat_sessions.c.id == studio_messages.c.session_id,
                ).join(
                    studio_documents,
                    studio_documents.c.id == studio_chat_sessions.c.document_id,
                )
            )
            .where(studio_messages.c.id == resource_id)
        )
    return None


def install_route_ownership(
    app: Flask, *, engine: Engine, profile: RuntimeProfile
) -> None:
    """Return a uniform 404 when a public user addresses another user's resource."""

    if not profile.requires_auth:
        raise ValueError("route ownership middleware requires the public profile")

    @app.before_request
    def authorize_route_identifiers() -> tuple[Response, int] | None:
        if not request.path.startswith("/api/v2"):
            return None
        view_args = request.view_args or {}
        if not view_args:
            return None
        owner_id = effective_owner_id()
        with engine.connect() as connection:
            for argument_name, resource_id in view_args.items():
                if not isinstance(resource_id, str):
                    continue
                statement = _owner_query(argument_name, resource_id)
                if statement is None:
                    continue
                owners = set(connection.execute(statement).scalars())
                if owner_id not in owners:
                    return jsonify(
                        {"error": {"code": "not_found", "message": "资源不存在"}}
                    ), 404
        return None
