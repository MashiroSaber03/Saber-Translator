"""Canonical Character Studio document boundary."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

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
    shaped.pop("grounding", None)
    shaped.pop("chatPreset", None)
    avatar = shaped.get("avatar")
    if isinstance(avatar, dict):
        avatar.pop("asset_path", None)
    shaped["title"] = canonical_title
    return shaped


def to_storage(document: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    canonical = deepcopy(dict(document))
    title = str(canonical.pop("title", "")).strip()
    if not title:
        raise StudioDocumentInvalid("document title is required")
    canonical.pop("id", None)
    canonical.pop("bookId", None)
    canonical.pop("revision", None)
    canonical.pop("generation", None)
    canonical.pop("createdAt", None)
    canonical.pop("updatedAt", None)
    canonical.pop("grounding", None)
    canonical.pop("chatPreset", None)
    identity = canonical.setdefault("identity", {})
    if isinstance(identity, dict):
        identity.pop("name", None)
    meta = canonical.setdefault("meta", {})
    if isinstance(meta, dict):
        meta.pop("title", None)
    avatar = canonical.get("avatar")
    if isinstance(avatar, dict):
        avatar.pop("asset_path", None)
    return title, canonical


def from_storage(row: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    document = deepcopy(dict(payload))
    title = str(row["title"])
    document["id"] = str(row["id"])
    document["bookId"] = str(row["book_id"])
    document["title"] = title
    document.setdefault("identity", {})["name"] = title
    document.setdefault("meta", {})["title"] = title
    document["revision"] = int(row["revision"])
    document["generation"] = int(row["generation"])
    document["avatarAssetId"] = row.get("avatar_asset_id")
    document["createdAt"] = _iso(row.get("created_at"))
    document["updatedAt"] = _iso(row.get("updated_at"))
    document.pop("grounding", None)
    document.pop("chatPreset", None)
    return document


def _iso(value: object) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat() + "Z"
    return str(value)
