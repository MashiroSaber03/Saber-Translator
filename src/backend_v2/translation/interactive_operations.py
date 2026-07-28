"""Worker-side saved OCR, color, and detection operations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
import uuid

from PIL import Image
from sqlalchemy import Engine, delete, insert, select, update
from sqlalchemy.engine import Connection

from src.backend_v2.operations.repository import (
    OperationFence,
    OperationRepository,
    RenderRequestRepository,
)
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.schema import assets, bubbles, page_assets, pages
from src.backend_v2.translation.pipeline import (
    LegacyTranslationAlgorithms,
    TranslationAlgorithms,
    TranslationPipelineService,
    _rgb_hex,
)


class InteractivePageOperationService:
    def __init__(
        self,
        *,
        data_root: Path,
        engine: Engine,
        repository: OperationRepository,
        algorithms: TranslationAlgorithms | None = None,
    ) -> None:
        self.engine = engine
        self.repository = repository
        self.storage = AssetStorageService(data_root, engine)
        self.algorithms = algorithms or LegacyTranslationAlgorithms()
        self.renders = RenderRequestRepository(engine)

    def handle(
        self,
        fence: OperationFence,
        operation: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        kind = str(operation["kind"])
        if kind == "bubble_ocr":
            result = self._bubble_ocr(fence, operation)
        elif kind == "bubble_color":
            result = self._bubble_color(fence, operation)
        elif kind == "page_detect":
            result = self._page_detect(fence, operation)
        else:
            raise ValueError(f"unsupported Worker page operation: {kind}")
        return {**result, "__already_published__": True}

    def _bubble_ocr(
        self,
        fence: OperationFence,
        operation: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        page, rows = self._snapshot(str(operation["pageId"]))
        index = self._bubble_index(rows, str(operation["bubbleId"]))
        image = self._open_input(operation, "source")
        try:
            result = self.algorithms.ocr(
                image,
                [dict(rows[index]["payload"])],
                self._payload(operation),
            )
        finally:
            image.close()
        texts = list(result.get("texts", []))
        details = list(result.get("results", []))
        if len(texts) != 1:
            raise RuntimeError("single-bubble OCR returned an invalid result count")
        payload = dict(rows[index]["payload"])
        payload["originalText"] = str(texts[0])
        payload["ocrResult"] = details[0] if details else None
        new_revision = int(page["document_revision"]) + 1

        def publish(connection: Connection, _row: Mapping[str, Any]) -> None:
            self._update_one_bubble(
                connection,
                page_id=str(operation["pageId"]),
                bubble_id=str(operation["bubbleId"]),
                base_revision=int(page["document_revision"]),
                new_revision=new_revision,
                payload=payload,
                render_changed=False,
            )

        response = {
            "bubbleId": operation["bubbleId"],
            "originalText": texts[0],
            "documentRevision": new_revision,
        }
        self.repository.complete(fence, result=response, publisher=publish)
        return response

    def _bubble_color(
        self,
        fence: OperationFence,
        operation: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        page, rows = self._snapshot(str(operation["pageId"]))
        index = self._bubble_index(rows, str(operation["bubbleId"]))
        image = self._open_input(operation, "source")
        try:
            colors = self.algorithms.colors(
                image, [dict(rows[index]["payload"])]
            )
        finally:
            image.close()
        if len(colors) != 1:
            raise RuntimeError("single-bubble color returned an invalid result count")
        color = colors[0]
        payload = dict(rows[index]["payload"])
        payload["autoFgColor"] = color.get("fg_color")
        payload["autoBgColor"] = color.get("bg_color")
        payload["colorConfidence"] = float(color.get("confidence", 0))
        style_defaults = json.loads(page["page_style_defaults_json"] or "{}")
        uses_auto_color = bool(style_defaults.get("useAutoTextColor", False))
        old_text_color = payload.get("textColor")
        if uses_auto_color and color.get("fg_color") is not None:
            payload["textColor"] = _rgb_hex(color["fg_color"])
        if uses_auto_color and color.get("bg_color") is not None:
            payload["fillColor"] = _rgb_hex(color["bg_color"])
        changes_render = payload.get("textColor") != old_text_color
        new_revision = int(page["document_revision"]) + 1

        def publish(connection: Connection, _row: Mapping[str, Any]) -> None:
            self._update_one_bubble(
                connection,
                page_id=str(operation["pageId"]),
                bubble_id=str(operation["bubbleId"]),
                base_revision=int(page["document_revision"]),
                new_revision=new_revision,
                payload=payload,
                render_changed=changes_render,
            )
            if changes_render and str(payload.get("translatedText", "")).strip():
                self.renders.upsert(
                    connection,
                    page_id=str(operation["pageId"]),
                    requested_revision=new_revision,
                    existing_chain=True,
                )

        response = {
            "bubbleId": operation["bubbleId"],
            "color": dict(color),
            "documentRevision": new_revision,
        }
        self.repository.complete(fence, result=response, publisher=publish)
        return response

    def _page_detect(
        self,
        fence: OperationFence,
        operation: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        page, _rows = self._snapshot(str(operation["pageId"]))
        image = self._open_input(operation, "source")
        try:
            detected = self.algorithms.detect(image, self._payload(operation))
        finally:
            image.close()
        coords = list(detected.get("coords", []))
        polygons = list(detected.get("polygons", []))
        angles = list(detected.get("angles", []))
        directions = list(detected.get("auto_directions", []))
        textlines = list(detected.get("textlines_per_bubble", []))
        style = json.loads(page["page_style_defaults_json"] or "{}")
        payloads = [
            TranslationPipelineService._new_bubble_payload(
                coords=value,
                polygon=polygons[index] if index < len(polygons) else [],
                angle=angles[index] if index < len(angles) else 0,
                auto_direction=(
                    directions[index] if index < len(directions) else "vertical"
                ),
                textlines=textlines[index] if index < len(textlines) else [],
                style=style,
            )
            for index, value in enumerate(coords)
        ]
        new_revision = int(page["document_revision"]) + 1

        def publish(connection: Connection, _row: Mapping[str, Any]) -> None:
            current = connection.execute(
                select(pages.c.document_revision).where(
                    pages.c.id == operation["pageId"]
                )
            ).scalar_one_or_none()
            if current != page["document_revision"]:
                raise RuntimeError("page revision changed")
            connection.execute(
                delete(bubbles).where(bubbles.c.page_id == operation["pageId"])
            )
            if payloads:
                connection.execute(
                    insert(bubbles),
                    [
                        {
                            "id": str(uuid.uuid4()),
                            "page_id": operation["pageId"],
                            "ordinal": index,
                            "payload_json": json.dumps(
                                payload,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            "payload_schema_version": 1,
                            "updated_revision": new_revision,
                        }
                        for index, payload in enumerate(payloads, start=1)
                    ],
                )
            connection.execute(
                update(pages)
                .where(
                    pages.c.id == operation["pageId"],
                    pages.c.document_revision == page["document_revision"],
                )
                .values(
                    document_revision=new_revision,
                    detection_state="processed",
                    render_status="stale",
                )
            )
            if connection.execute(
                select(page_assets.c.asset_id).where(
                    page_assets.c.page_id == operation["pageId"],
                    page_assets.c.role == "translated",
                )
            ).scalar_one_or_none() is not None:
                self.renders.upsert(
                    connection,
                    page_id=str(operation["pageId"]),
                    requested_revision=new_revision,
                    existing_chain=True,
                )

        response = {
            "bubbleCount": len(payloads),
            "documentRevision": new_revision,
        }
        self.repository.complete(fence, result=response, publisher=publish)
        return response

    def _snapshot(self, page_id: str):
        with self.engine.connect() as connection:
            page = connection.execute(
                select(pages).where(pages.c.id == page_id)
            ).mappings().one()
            rows = [
                {
                    "id": row["id"],
                    "payload": json.loads(row["payload_json"]),
                }
                for row in connection.execute(
                    select(bubbles)
                    .where(bubbles.c.page_id == page_id)
                    .order_by(bubbles.c.ordinal)
                ).mappings()
            ]
        return page, rows

    def _open_input(
        self,
        operation: Mapping[str, Any],
        role: str,
    ) -> Image.Image:
        inputs = operation.get("inputs")
        if not isinstance(inputs, Mapping) or role not in inputs:
            raise RuntimeError(f"operation has no frozen {role} input")
        with self.engine.connect() as connection:
            relative_path = connection.execute(
                select(assets.c.relative_path).where(
                    assets.c.id == inputs[role]
                )
            ).scalar_one()
        opened = Image.open(self.storage.resolve_relative_path(relative_path))
        converted = opened.convert("RGB")
        opened.close()
        return converted

    @staticmethod
    def _bubble_index(rows: list[dict[str, Any]], bubble_id: str) -> int:
        for index, row in enumerate(rows):
            if row["id"] == bubble_id:
                return index
        raise RuntimeError("operation bubble no longer exists")

    @staticmethod
    def _payload(operation: Mapping[str, Any]) -> dict[str, Any]:
        request = operation.get("request")
        if not isinstance(request, Mapping):
            return {}
        payload = request.get("payload")
        return dict(payload) if isinstance(payload, Mapping) else {}

    @staticmethod
    def _update_one_bubble(
        connection: Connection,
        *,
        page_id: str,
        bubble_id: str,
        base_revision: int,
        new_revision: int,
        payload: Mapping[str, Any],
        render_changed: bool,
    ) -> None:
        page = connection.execute(
            select(
                pages.c.render_status,
                pages.c.rendered_revision,
            ).where(
                pages.c.id == page_id,
                pages.c.document_revision == base_revision,
            )
        ).mappings().one_or_none()
        if page is None:
            raise RuntimeError("page revision changed")
        values: dict[str, object] = {"document_revision": new_revision}
        if render_changed:
            values["render_status"] = "stale"
        elif (
            page["render_status"] == "ready"
            and page["rendered_revision"] == base_revision
        ):
            values["rendered_revision"] = new_revision
            connection.execute(
                update(page_assets)
                .where(
                    page_assets.c.page_id == page_id,
                    page_assets.c.role.in_(
                        ("translated", "thumbnail_translated")
                    ),
                    page_assets.c.input_document_revision == base_revision,
                )
                .values(input_document_revision=new_revision)
            )
        changed = connection.execute(
            update(pages)
            .where(
                pages.c.id == page_id,
                pages.c.document_revision == base_revision,
            )
            .values(**values)
        )
        if changed.rowcount != 1:
            raise RuntimeError("page revision changed")
        bubble_changed = connection.execute(
            update(bubbles)
            .where(bubbles.c.id == bubble_id, bubbles.c.page_id == page_id)
            .values(
                payload_json=json.dumps(
                    dict(payload),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                updated_revision=new_revision,
            )
        )
        if bubble_changed.rowcount != 1:
            raise RuntimeError("bubble no longer exists")
