"""Canonical Character Studio document boundary."""

from __future__ import annotations

from datetime import datetime
import json
from typing import Any, Mapping

from src.backend_v2.serialization import canonical_json as _json
from src.backend_v2.timestamps import iso_utc
from src.backend_v2.studio.pure import (
    create_empty_document,
    validate_current_document,
)


class StudioDocumentInvalid(ValueError):
    pass


def new_document(book_id: str, *, title: str) -> dict[str, Any]:
    return normalize_document(
        book_id=book_id,
        title=title,
        document=create_empty_document(book_id, title=title),
    )


def normalize_document(
    *,
    book_id: str,
    title: str | None,
    document: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        return validate_current_document(
            document,
            book_id=book_id,
            title=title,
        )
    except ValueError as exc:
        raise StudioDocumentInvalid(str(exc)) from exc


def to_storage(document: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    book_id = document.get("bookId")
    title_value = document.get("title")
    if not isinstance(book_id, str) or not book_id:
        raise StudioDocumentInvalid("document bookId is required")
    if not isinstance(title_value, str) or not title_value:
        raise StudioDocumentInvalid("document title is required")
    try:
        canonical = validate_current_document(
            document,
            book_id=book_id,
            title=title_value,
        )
    except ValueError as exc:
        raise StudioDocumentInvalid(str(exc)) from exc
    title = canonical.pop("title")
    origin = dict(canonical["origin"])
    origin_type = origin["type"]
    if origin_type not in {"analysis", "manual", "imported"}:
        raise StudioDocumentInvalid("origin.type is invalid")
    identity = dict(canonical["identity"])
    del identity["name"]
    meta = dict(canonical["meta"])
    status = dict(canonical["status"])
    export_artifacts = dict(canonical["exportArtifacts"])
    last_review = export_artifacts.get("last_review")
    last_diagnostics = status.get("last_diagnostics")
    return title, {
        "origin_type": origin_type,
        "source_character": (
            origin["source_character"]
        ),
        "tags_json": _json(meta["tags"]),
        "is_favorite": status["is_favorite"],
        "identity_json": _json(identity),
        "core_messages_json": _json(canonical["coreMessages"]),
        "lorebook_json": _json(canonical["lorebook"]),
        "regex_scripts_json": _json(canonical["regexScripts"]),
        "state_tasks_json": _json(canonical["stateTasks"]),
        "frozen_sections_json": _json(
            status["frozen_sections"]
        ),
        "last_review_json": (
            _json(last_review)
            if last_review is not None
            else None
        ),
        "last_diagnostics_json": (
            _json(last_diagnostics)
            if last_diagnostics is not None
            else None
        ),
        "last_validated_at": _parse_datetime(status["last_validated_at"]),
    }


def from_storage(row: Mapping[str, Any]) -> dict[str, Any]:
    title = row["title"]
    if not isinstance(title, str) or not title:
        raise StudioDocumentInvalid("stored Studio document title is invalid")
    document_id = row["id"]
    book_id = row["book_id"]
    if not isinstance(document_id, str) or not document_id:
        raise StudioDocumentInvalid("stored Studio document id is invalid")
    if not isinstance(book_id, str) or not book_id:
        raise StudioDocumentInvalid("stored Studio document bookId is invalid")
    avatar_id = row["avatar_asset_id"]
    if avatar_id is not None and (
        not isinstance(avatar_id, str) or not avatar_id
    ):
        raise StudioDocumentInvalid(
            "stored Studio document avatarAssetId is invalid"
        )
    identity = _load_object(row["identity_json"])
    identity["name"] = title
    last_review = _load_nullable(row["last_review_json"])
    last_diagnostics = _load_nullable(row["last_diagnostics_json"])
    created_at = iso_utc(row["created_at"])
    updated_at = iso_utc(row["updated_at"])
    raw = {
        "id": document_id,
        "bookId": book_id,
        "title": title,
        "origin": {
            "type": row["origin_type"],
            "source_character": row["source_character"],
        },
        "status": {
            "is_favorite": row["is_favorite"],
            "frozen_sections": _load_list(
                row["frozen_sections_json"]
            ),
            "last_diagnostics": last_diagnostics,
            "last_validated_at": iso_utc(
                row["last_validated_at"]
            ),
        },
        "meta": {
            "title": title,
            "tags": _load_list(row["tags_json"]),
        },
        "identity": identity,
        "coreMessages": _load_object(row["core_messages_json"]),
        "lorebook": _load_object(row["lorebook_json"]),
        "regexScripts": _load_list(row["regex_scripts_json"]),
        "stateTasks": _load_list(row["state_tasks_json"]),
        "exportArtifacts": (
            {"last_review": last_review}
            if last_review is not None
            else {}
        ),
        "revision": row["revision"],
        "avatarAssetId": avatar_id,
        "avatarUrl": (
            f"/api/v2/assets/{avatar_id}"
            if avatar_id is not None
            else None
        ),
        "createdAt": created_at,
        "updatedAt": updated_at,
    }
    try:
        canonical = validate_current_document(
            raw,
            book_id=book_id,
            title=title,
        )
    except ValueError as exc:
        raise StudioDocumentInvalid(str(exc)) from exc
    return {
        "id": document_id,
        **canonical,
        "revision": row["revision"],
        "avatarAssetId": avatar_id,
        "avatarUrl": (
            f"/api/v2/assets/{avatar_id}"
            if avatar_id is not None
            else None
        ),
        "createdAt": created_at,
        "updatedAt": updated_at,
    }


def _load_nullable(value: object) -> object | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise StudioDocumentInvalid(
            "stored Studio document JSON column is invalid"
        )
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise StudioDocumentInvalid(
            "stored Studio document JSON is invalid"
        ) from exc


def _load_object(value: object) -> dict[str, Any]:
    loaded = _load_nullable(value)
    if not isinstance(loaded, Mapping):
        raise StudioDocumentInvalid(
            "stored Studio document object JSON is invalid"
        )
    return dict(loaded)


def _load_list(value: object) -> list[Any]:
    loaded = _load_nullable(value)
    if not isinstance(loaded, list):
        raise StudioDocumentInvalid(
            "stored Studio document array JSON is invalid"
        )
    return list(loaded)


def _parse_datetime(value: object) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise StudioDocumentInvalid("Studio document timestamp is invalid")
    rendered = value
    if rendered.endswith("Z"):
        rendered = rendered[:-1] + "+00:00"
    parsed = datetime.fromisoformat(rendered)
    return parsed.replace(tzinfo=None)
