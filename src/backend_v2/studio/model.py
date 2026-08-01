"""Canonical Character Studio document boundary."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from typing import Any, Mapping

from src.backend_v2.serialization import canonical_json as _json
from src.backend_v2.timestamps import iso_utc
from src.backend_v2.studio.pure import (
    create_empty_document,
    ensure_document_shape,
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
    raw = deepcopy(dict(document))
    raw_identity = raw.get("identity")
    identity_name = (
        str(raw_identity.get("name", "")).strip()
        if isinstance(raw_identity, Mapping)
        else ""
    )
    meta = raw.get("meta")
    meta_title = (
        str(meta.get("title", "")).strip()
        if isinstance(meta, Mapping)
        else ""
    )
    explicit_title = str(title or "").strip()
    names = {
        value for value in (explicit_title, identity_name, meta_title) if value
    }
    if len(names) > 1:
        raise StudioDocumentInvalid(
            "title, meta.title and identity.name must agree"
        )
    canonical_title = next(iter(names), "新角色")
    shaped = ensure_document_shape(raw, book_id=book_id)
    shaped["identity"]["name"] = canonical_title
    shaped["meta"]["title"] = canonical_title
    shaped["title"] = canonical_title
    return shaped


def to_storage(document: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    canonical = deepcopy(dict(document))
    title = str(canonical.pop("title", "")).strip()
    if not title:
        raise StudioDocumentInvalid("document title is required")
    origin = _object(canonical.get("origin"))
    origin_type = str(origin.get("type", "manual") or "manual")
    if origin_type not in {"analysis", "manual", "imported"}:
        raise StudioDocumentInvalid("origin.type is invalid")
    identity = _object(canonical.get("identity"))
    identity.pop("name", None)
    meta = _object(canonical.get("meta"))
    status = _object(canonical.get("status"))
    export_artifacts = _object(canonical.get("exportArtifacts"))
    last_review = export_artifacts.get("last_review")
    last_diagnostics = status.get("last_diagnostics")
    return title, {
        "origin_type": origin_type,
        "source_character": (
            str(origin["source_character"])
            if origin.get("source_character") is not None
            else None
        ),
        "tags_json": _json(meta.get("tags", [])),
        "is_favorite": bool(status.get("is_favorite", False)),
        "identity_json": _json(identity),
        "core_messages_json": _json(
            _object(canonical.get("coreMessages"))
        ),
        "lorebook_json": _json(_object(canonical.get("lorebook"))),
        "regex_scripts_json": _json(
            canonical.get("regexScripts", [])
        ),
        "state_tasks_json": _json(canonical.get("stateTasks", [])),
        "frozen_sections_json": _json(
            status.get("frozen_sections", [])
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
        "last_validated_at": _parse_datetime(
            status.get("last_validated_at")
        ),
    }


def from_storage(row: Mapping[str, Any]) -> dict[str, Any]:
    title = str(row["title"])
    avatar_id = row.get("avatar_asset_id")
    identity = _load_object(row.get("identity_json"))
    identity["name"] = title
    last_review = _load_nullable(row.get("last_review_json"))
    last_diagnostics = _load_nullable(row.get("last_diagnostics_json"))
    return {
        "id": str(row["id"]),
        "bookId": str(row["book_id"]),
        "title": title,
        "origin": {
            "type": str(row["origin_type"]),
            "source_character": row.get("source_character"),
        },
        "status": {
            "is_favorite": bool(row.get("is_favorite")),
            "frozen_sections": _load_list(
                row.get("frozen_sections_json")
            ),
            "last_diagnostics": last_diagnostics,
            "last_validated_at": iso_utc(
                row.get("last_validated_at")
            ),
        },
        "meta": {
            "title": title,
            "tags": _load_list(row.get("tags_json")),
        },
        "identity": identity,
        "coreMessages": _load_object(row.get("core_messages_json")),
        "lorebook": _load_object(row.get("lorebook_json")),
        "regexScripts": _load_list(row.get("regex_scripts_json")),
        "stateTasks": _load_list(row.get("state_tasks_json")),
        "exportArtifacts": (
            {"last_review": last_review}
            if last_review is not None
            else {}
        ),
        "revision": int(row["revision"]),
        "avatarAssetId": avatar_id,
        "avatarUrl": (
            f"/api/v2/assets/{avatar_id}" if avatar_id else None
        ),
        "createdAt": iso_utc(row.get("created_at")),
        "updatedAt": iso_utc(row.get("updated_at")),
    }


def _object(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _load_nullable(value: object) -> object | None:
    if value is None:
        return None
    return json.loads(str(value))


def _load_object(value: object) -> dict[str, Any]:
    loaded = _load_nullable(value)
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def _load_list(value: object) -> list[Any]:
    loaded = _load_nullable(value)
    return list(loaded) if isinstance(loaded, list) else []


def _parse_datetime(value: object) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    rendered = str(value)
    if rendered.endswith("Z"):
        rendered = rendered[:-1] + "+00:00"
    parsed = datetime.fromisoformat(rendered)
    return parsed.replace(tzinfo=None)
