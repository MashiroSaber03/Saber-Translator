"""Short, streamed Studio import/export and asset conversions."""

from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any, BinaryIO, Mapping

from sqlalchemy import Engine, select, update

from src.backend_v2.content.image_import import (
    ImageImportService,
    ImportSafetyLimits,
)
from src.backend_v2.content.repository import ContentRepository
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.schema import assets
from src.backend_v2.timestamps import utcnow
from src.backend_v2.studio.media import read_card_png, write_card_png
from src.backend_v2.studio.pure import (
    build_export_bundle,
    import_document_payload,
)
from src.backend_v2.studio.repository import StudioRepository


class StudioIOService:
    def __init__(
        self,
        *,
        data_root: Path,
        engine: Engine,
        repository: StudioRepository,
        limits: ImportSafetyLimits = ImportSafetyLimits(),
    ) -> None:
        self.engine = engine
        self.repository = repository
        self.limits = limits
        self.storage = AssetStorageService(data_root, engine)
        self.images = ImageImportService(
            data_root=data_root,
            repository=ContentRepository(engine),
            storage=self.storage,
            limits=limits,
        )

    def publish_image(
        self,
        upload: BinaryIO,
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        payload = self._read_limited(upload)
        request_identity = _binary_request_identity(payload)
        scope = "POST:uploadStudioAsset"
        replay = self.repository.replay_short_command(
            scope=scope,
            key=idempotency_key,
            request=request_identity,
        )
        if replay is not None:
            return replay
        result: dict[str, Any] = {}

        def bind(connection, asset_id: str) -> None:
            row = connection.execute(
                select(assets).where(assets.c.id == asset_id)
            ).mappings().one()
            response, replayed = (
                self.repository.execute_bound_short_command(
                    connection,
                    scope=scope,
                    key=idempotency_key,
                    request=request_identity,
                    http_status=201,
                    resource_type="asset",
                    mutation=lambda: (
                        _asset_row_dto(row),
                        asset_id,
                    ),
                )
            )
            if replayed:
                connection.execute(
                    update(assets)
                    .where(assets.c.id == asset_id)
                    .values(gc_marked_at=utcnow())
                )
            result.update(response)

        self.images.publish_standalone_source(
            BytesIO(payload),
            bind=bind,
        )
        return result

    def import_document(
        self,
        *,
        book_id: str,
        upload: BinaryIO,
        filename: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        payload = self._read_limited(upload)
        request_identity = {
            "bookId": book_id,
            "filename": filename,
            **_binary_request_identity(payload),
        }
        scope = f"POST:importStudioDocument:{book_id}"
        replay = self.repository.replay_short_command(
            scope=scope,
            key=idempotency_key,
            request=request_identity,
        )
        if replay is not None:
            return replay
        suffix = Path(filename or "").suffix.lower()
        if suffix == ".json":
            decoded = json.loads(payload.decode("utf-8-sig"))
            if not isinstance(decoded, dict):
                raise ValueError("Studio JSON import must contain an object")
            document = import_document_payload(book_id, decoded)
            return self.repository.create_document(
                book_id=book_id,
                title=str(document["identity"]["name"]),
                document=document,
                kind="imported",
                idempotency_key=idempotency_key,
                idempotency_request=request_identity,
                idempotency_scope=scope,
            )
        card = read_card_png(payload) if suffix == ".png" else {}
        if card:
            document = import_document_payload(book_id, card)
            title = str(document["identity"]["name"])
        else:
            title = Path(filename or "导入角色").stem or "导入角色"
            document = None
        avatar = self.images.publish_standalone_source(BytesIO(payload))
        try:
            created = self.repository.create_document(
                book_id=book_id,
                title=title,
                document=document,
                kind="imported",
                avatar_asset_id=avatar.id,
                idempotency_key=idempotency_key,
                idempotency_request=request_identity,
                idempotency_scope=scope,
            )
        except Exception:
            self._mark_assets_for_gc([avatar.id])
            raise
        if created.get("avatarAssetId") != avatar.id:
            self._mark_assets_for_gc([avatar.id])
        return created

    def export_png(self, document: Mapping[str, Any]) -> bytes:
        bundle = build_export_bundle(document)
        return write_card_png(
            bundle["v3"],
            base_image_path=self._asset_path(
                document.get("avatarAssetId")
            ),
        )

    def export_session(self, session_id: str) -> dict[str, Any]:
        session = deepcopy(self.repository.get_session(session_id))
        session["schema"] = "saber-studio-chat-v2"
        for message in session.get("messages", []):
            exported: list[dict[str, Any]] = []
            for index, attachment in enumerate(
                message.pop("attachments", []),
                start=1,
            ):
                asset_id = str(attachment.get("assetId", ""))
                path = self._asset_path(asset_id)
                if path is None:
                    continue
                exported.append(
                    {
                        "filename": f"attachment-{index}{path.suffix}",
                        "mime_type": attachment.get(
                            "mimeType",
                            "application/octet-stream",
                        ),
                        "blob_base64": base64.b64encode(
                            path.read_bytes()
                        ).decode("ascii"),
                    }
                )
            message["attachments"] = exported
        for key in (
            "sessionId",
            "documentId",
            "revision",
            "generation",
            "archived",
        ):
            session.pop(key, None)
        return session

    def import_session(
        self,
        *,
        document_id: str,
        base_index_revision: int,
        payload: Mapping[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        request_identity = {
            "documentId": document_id,
            "baseIndexRevision": base_index_revision,
            "session": deepcopy(dict(payload)),
        }
        scope = f"POST:importStudioSession:{document_id}"
        replay = self.repository.replay_short_command(
            scope=scope,
            key=idempotency_key,
            request=request_identity,
        )
        if replay is not None:
            return replay
        imported = deepcopy(dict(payload))
        messages = imported.get("messages", [])
        if not isinstance(messages, list):
            raise ValueError("session messages must be an array")
        imported_assets = self._restore_session_attachments(messages)
        try:
            result = self.repository.import_session(
                document_id=document_id,
                base_index_revision=base_index_revision,
                payload=imported,
                idempotency_key=idempotency_key,
                idempotency_request=request_identity,
            )
        except Exception:
            self._mark_assets_for_gc(imported_assets)
            raise
        referenced = {
            str(attachment.get("assetId"))
            for message in result.get("messages", [])
            for attachment in message.get("attachments", [])
            if isinstance(attachment, Mapping)
            and attachment.get("assetId")
        }
        self._mark_assets_for_gc(
            [
                asset_id
                for asset_id in imported_assets
                if asset_id not in referenced
            ]
        )
        return result

    def _restore_session_attachments(
        self,
        messages: list[Any],
    ) -> list[str]:
        imported_assets: list[str] = []
        try:
            for message in messages:
                if not isinstance(message, dict):
                    raise ValueError("each session message must be an object")
                restored_ids: list[str] = []
                attachments = message.pop("attachments", [])
                if not isinstance(attachments, list):
                    raise ValueError("message attachments must be an array")
                for attachment in attachments:
                    if not isinstance(attachment, Mapping):
                        raise ValueError("each attachment must be an object")
                    encoded = str(attachment.get("blob_base64", "") or "")
                    if not encoded:
                        continue
                    try:
                        binary = base64.b64decode(encoded, validate=True)
                    except ValueError as exc:
                        raise ValueError(
                            "attachment base64 is invalid"
                        ) from exc
                    if len(binary) > self.limits.max_image_bytes:
                        raise ValueError(
                            "attachment exceeds the configured single-file byte limit"
                        )
                    asset_id = self.images.publish_standalone_source(
                        BytesIO(binary)
                    ).id
                    restored_ids.append(asset_id)
                    imported_assets.append(asset_id)
                message["assetIds"] = restored_ids
        except Exception:
            self._mark_assets_for_gc(imported_assets)
            raise
        return imported_assets

    def _asset_path(self, asset_id: object) -> Path | None:
        if not isinstance(asset_id, str) or not asset_id:
            return None
        with self.engine.connect() as connection:
            relative_path = connection.execute(
                select(assets.c.relative_path).where(
                    assets.c.id == asset_id,
                    assets.c.integrity_status == "ok",
                    assets.c.gc_marked_at.is_(None),
                )
            ).scalar_one_or_none()
        if relative_path is None:
            return None
        path = self.storage.resolve_relative_path(str(relative_path))
        return path if path.is_file() else None

    def _mark_assets_for_gc(self, asset_ids: list[str]) -> None:
        if not asset_ids:
            return
        with self.engine.begin() as connection:
            connection.execute(
                update(assets)
                .where(assets.c.id.in_(tuple(set(asset_ids))))
                .values(gc_marked_at=utcnow())
            )

    def _read_limited(self, upload: BinaryIO) -> bytes:
        output = bytearray()
        while True:
            chunk = upload.read(self.limits.stream_chunk_bytes)
            if not chunk:
                break
            output.extend(chunk)
            if len(output) > self.limits.max_image_bytes:
                raise ValueError(
                    "file exceeds the configured single-file byte limit"
                )
        if not output:
            raise ValueError("uploaded file is empty")
        return bytes(output)


def _asset_row_dto(row: Mapping[str, Any]) -> dict[str, Any]:
    asset_id = str(row["id"])
    return {
        "assetId": asset_id,
        "assetUrl": f"/api/v2/assets/{asset_id}",
        "mimeType": str(row["mime_type"]),
        "byteSize": int(row["byte_size"]),
        "width": row["width"],
        "height": row["height"],
    }


def _binary_request_identity(payload: bytes) -> dict[str, Any]:
    return {
        "checksum": hashlib.sha256(payload).hexdigest(),
        "byteSize": len(payload),
    }
