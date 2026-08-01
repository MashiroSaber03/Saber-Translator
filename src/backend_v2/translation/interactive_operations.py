"""Saved editor operations executed outside the browser lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
import uuid

from PIL import Image
from sqlalchemy import Engine, delete, insert, select, update
from sqlalchemy.engine import Connection

from src.backend_v2.content.page_style import rgb_to_hex, validate_page_style
from src.backend_v2.serialization import canonical_json
from src.backend_v2.operations.repository import (
    OperationFence,
    OperationRepository,
    RenderRequestRepository,
)
from src.backend_v2.rendering.service import publish_png_asset
from src.backend_v2.storage.assets import AssetRecord, AssetStorageService
from src.backend_v2.storage.platform_repositories import SettingsRepository
from src.backend_v2.storage.schema import (
    bubbles,
    page_assets,
    pages,
)
from src.backend_v2.translation.pipeline import (
    CoreTranslationAlgorithms,
    TranslationAlgorithms,
    TranslationPipelineService,
    _preserve_detected_text,
)


class InteractivePageOperationService:
    def __init__(
        self,
        *,
        data_root: Path,
        engine: Engine,
        repository: OperationRepository,
        algorithms: TranslationAlgorithms | None = None,
        plugin_runtime: Any | None = None,
    ) -> None:
        self.engine = engine
        self.repository = repository
        self.storage = AssetStorageService(data_root, engine)
        self.credentials = SettingsRepository(engine)
        self.algorithms = algorithms or CoreTranslationAlgorithms()
        self.renders = RenderRequestRepository(engine)
        self.plugin_runtime = plugin_runtime

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
        elif kind == "bubble_translate":
            result = self._bubble_translate(fence, operation)
        else:
            raise ValueError(f"unsupported page operation: {kind}")
        return {**result, "__already_published__": True}

    def _bubble_ocr(
        self,
        fence: OperationFence,
        operation: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        page, rows = self._snapshot(str(operation["pageId"]))
        index = self._bubble_index(rows, str(operation["bubbleId"]))
        page_id = str(operation["pageId"])
        source_asset_id = self._input_asset_id(operation, "source")
        raw_config = self._payload(operation)
        before = self._atomic_hook(
            fence,
            phase="before",
            scope="ocr",
            page_id=page_id,
            data={
                "pageId": page_id,
                "sourceAssetId": source_asset_id,
                "bubbles": [dict(rows[index]["payload"])],
                "ocrConfig": raw_config,
            },
        )
        image = self._open_asset(str(before["sourceAssetId"]), "RGB")
        try:
            result = self.algorithms.ocr(
                image,
                [dict(value) for value in before["bubbles"]],
                self._with_credential(before["ocrConfig"]),
            )
        finally:
            image.close()
        texts = list(result.get("texts", []))
        details = list(result.get("results", []))
        after = self._atomic_hook(
            fence,
            phase="after",
            scope="ocr",
            page_id=page_id,
            data={
                "pageId": page_id,
                "originalTexts": [str(value) for value in texts],
                "ocrResults": details,
            },
        )
        texts = list(after["originalTexts"])
        details = list(after["ocrResults"])
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

    def _bubble_translate(
        self,
        fence: OperationFence,
        operation: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        page, rows = self._snapshot(str(operation["pageId"]))
        index = self._bubble_index(rows, str(operation["bubbleId"]))
        payload = dict(rows[index]["payload"])
        original_text = str(payload.get("originalText", "")).strip()
        if not original_text:
            raise ValueError("bubble has no original text")
        config = self._with_credential(self._payload(operation))
        translated = self.algorithms.translate(
            [original_text],
            config,
            mode="single",
        )
        values = list(translated.get("translated", []))
        textbox = list(translated.get("textbox", []))
        if len(values) != 1:
            raise RuntimeError(
                "single-bubble translation returned an invalid result count"
            )
        payload["translatedText"] = str(values[0])
        if textbox:
            payload["textboxText"] = str(textbox[0])
        new_revision = int(page["document_revision"]) + 1
        has_drawable_text = any(
            str(
                payload.get("translatedText", "")
                if row_index == index
                else row["payload"].get("translatedText", "")
            ).strip()
            for row_index, row in enumerate(rows)
        )

        def publish(connection: Connection, _row: Mapping[str, Any]) -> None:
            has_translated = connection.execute(
                select(page_assets.c.asset_id).where(
                    page_assets.c.page_id == operation["pageId"],
                    page_assets.c.role == "translated",
                )
            ).scalar_one_or_none()
            needs_render = has_translated is not None or has_drawable_text
            self._update_one_bubble(
                connection,
                page_id=str(operation["pageId"]),
                bubble_id=str(operation["bubbleId"]),
                base_revision=int(page["document_revision"]),
                new_revision=new_revision,
                payload=payload,
                render_changed=needs_render,
            )
            if needs_render:
                self.renders.upsert(
                    connection,
                    page_id=str(operation["pageId"]),
                    requested_revision=new_revision,
                    existing_chain=True,
                )

        response = {
            "bubbleId": operation["bubbleId"],
            "translatedText": str(values[0]),
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
        page_id = str(operation["pageId"])
        source_asset_id = self._input_asset_id(operation, "source")
        before = self._atomic_hook(
            fence,
            phase="before",
            scope="color",
            page_id=page_id,
            data={
                "pageId": page_id,
                "sourceAssetId": source_asset_id,
                "bubbles": [dict(rows[index]["payload"])],
            },
        )
        image = self._open_asset(str(before["sourceAssetId"]), "RGB")
        try:
            colors = self.algorithms.colors(
                image,
                [dict(value) for value in before["bubbles"]],
            )
        finally:
            image.close()
        after = self._atomic_hook(
            fence,
            phase="after",
            scope="color",
            page_id=page_id,
            data={
                "pageId": page_id,
                "colors": [
                    {
                        "fgColor": (
                            list(value["fg_color"])
                            if value.get("fg_color") is not None
                            else None
                        ),
                        "bgColor": (
                            list(value["bg_color"])
                            if value.get("bg_color") is not None
                            else None
                        ),
                        "confidence": float(
                            value.get("confidence", 0)
                        ),
                    }
                    for value in colors
                ],
            },
        )
        colors = [
            {
                "fg_color": value.get("fgColor"),
                "bg_color": value.get("bgColor"),
                "confidence": value.get("confidence", 0),
            }
            for value in after["colors"]
        ]
        if len(colors) != 1:
            raise RuntimeError("single-bubble color returned an invalid result count")
        color = colors[0]
        payload = dict(rows[index]["payload"])
        payload["autoFgColor"] = color.get("fg_color")
        payload["autoBgColor"] = color.get("bg_color")
        payload["colorConfidence"] = float(color.get("confidence", 0))
        style_defaults = validate_page_style(
            json.loads(page["page_style_defaults_json"]),
            partial=False,
        )
        uses_auto_color = bool(style_defaults["useAutoTextColor"])
        old_text_color = payload.get("textColor")
        if uses_auto_color and color.get("fg_color") is not None:
            payload["textColor"] = rgb_to_hex(color["fg_color"])
        if uses_auto_color and color.get("bg_color") is not None:
            payload["fillColor"] = rgb_to_hex(color["bg_color"])
        changes_render = payload.get("textColor") != old_text_color
        new_revision = int(page["document_revision"]) + 1
        has_drawable_text = any(
            str(
                payload.get("translatedText", "")
                if row_index == index
                else row["payload"].get("translatedText", "")
            ).strip()
            for row_index, row in enumerate(rows)
        )

        def publish(connection: Connection, _row: Mapping[str, Any]) -> None:
            has_translated = connection.execute(
                select(page_assets.c.asset_id).where(
                    page_assets.c.page_id == operation["pageId"],
                    page_assets.c.role == "translated",
                )
            ).scalar_one_or_none()
            needs_render = bool(
                changes_render
                and (has_translated is not None or has_drawable_text)
            )
            self._update_one_bubble(
                connection,
                page_id=str(operation["pageId"]),
                bubble_id=str(operation["bubbleId"]),
                base_revision=int(page["document_revision"]),
                new_revision=new_revision,
                payload=payload,
                render_changed=needs_render,
            )
            if needs_render:
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
        page_id = str(operation["pageId"])
        source_asset_id = self._input_asset_id(operation, "source")
        before = self._atomic_hook(
            fence,
            phase="before",
            scope="detect",
            page_id=page_id,
            data={
                "pageId": page_id,
                "sourceAssetId": source_asset_id,
                "detectorConfig": self._payload(operation),
            },
        )
        image = self._open_asset(str(before["sourceAssetId"]), "RGB")
        try:
            detected = self.algorithms.detect(
                image,
                dict(before["detectorConfig"]),
            )
        finally:
            image.close()
        coords = list(detected.get("coords", []))
        polygons = list(detected.get("polygons", []))
        angles = list(detected.get("angles", []))
        directions = list(detected.get("auto_directions", []))
        textlines = list(detected.get("textlines_per_bubble", []))
        style = validate_page_style(
            json.loads(page["page_style_defaults_json"]),
            partial=False,
        )
        payloads = _preserve_detected_text([
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
        ], tuple(row["payload"] for row in _rows))
        mask_record: AssetRecord | None = None
        raw_mask = detected.get("raw_mask")
        if isinstance(raw_mask, Image.Image):
            mask_record = publish_png_asset(self.storage, raw_mask, mode="L")
            raw_mask.close()
        elif raw_mask is not None:
            mask_image = Image.fromarray(raw_mask)
            try:
                mask_record = publish_png_asset(
                    self.storage,
                    mask_image,
                    mode="L",
                )
            finally:
                mask_image.close()
        after = self._atomic_hook(
            fence,
            phase="after",
            scope="detect",
            page_id=page_id,
            data={
                "pageId": page_id,
                "bubbles": payloads,
                "textMaskAssetId": (
                    mask_record.id if mask_record is not None else None
                ),
            },
        )
        payloads = [dict(value) for value in after["bubbles"]]
        mask_asset_id = after.get("textMaskAssetId")
        mask_record = (
            self._asset_record(str(mask_asset_id))
            if mask_asset_id is not None
            else None
        )
        new_revision = int(page["document_revision"]) + 1

        def publish(connection: Connection, _row: Mapping[str, Any]) -> None:
            current = connection.execute(
                select(pages.c.document_revision).where(
                    pages.c.id == operation["pageId"]
                )
            ).scalar_one_or_none()
            if current != page["document_revision"]:
                raise RuntimeError("page revision changed")
            has_translated_asset = (
                connection.execute(
                    select(page_assets.c.asset_id).where(
                        page_assets.c.page_id == operation["pageId"],
                        page_assets.c.role == "translated",
                    )
                ).scalar_one_or_none()
                is not None
            )
            has_drawable_text = any(
                str(payload.get("translatedText", "")).strip()
                for payload in payloads
            )
            needs_render = has_translated_asset or has_drawable_text
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
                            "payload_json": canonical_json(payload),
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
                    rendered_revision=(
                        page["rendered_revision"]
                        if needs_render
                        else None
                    ),
                    render_status=(
                        "stale" if needs_render else "not_rendered"
                    ),
                )
            )
            connection.execute(
                delete(page_assets).where(
                    page_assets.c.page_id == operation["pageId"],
                    page_assets.c.role == "text_mask",
                )
            )
            if mask_record is not None:
                connection.execute(
                    insert(page_assets).values(
                        page_id=operation["pageId"],
                        role="text_mask",
                        asset_id=mask_record.id,
                        input_source_revision=page["source_revision"],
                        input_document_revision=new_revision,
                        parent_asset_id=None,
                        producer_job_step_id=None,
                        producer_operation_id=fence.operation_id,
                        producer_render_request_id=None,
                    )
                )
            if needs_render:
                self.renders.upsert(
                    connection,
                    page_id=str(operation["pageId"]),
                    requested_revision=new_revision,
                    existing_chain=True,
                )

        response = {
            "bubbleCount": len(payloads),
            "documentRevision": new_revision,
            "textMaskAssetId": mask_record.id if mask_record else None,
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

    @staticmethod
    def _input_asset_id(
        operation: Mapping[str, Any],
        role: str,
    ) -> str:
        inputs = operation.get("inputs")
        if not isinstance(inputs, Mapping) or role not in inputs:
            raise RuntimeError(f"operation has no frozen {role} input")
        return str(inputs[role])

    def _open_asset(
        self,
        asset_id: str,
        mode: str,
    ) -> Image.Image:
        record = self._asset_record(asset_id)
        with Image.open(
            self.storage.resolve_relative_path(record.relative_path)
        ) as opened:
            return opened.convert(mode)

    def _asset_record(self, asset_id: str) -> AssetRecord:
        record = self.storage.get_record(asset_id)
        if record is None:
            raise RuntimeError("operation referenced an unknown asset")
        return record

    def _atomic_hook(
        self,
        fence: OperationFence,
        *,
        phase: str,
        scope: str,
        page_id: str,
        data: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.plugin_runtime is None:
            return dict(data)
        return self.plugin_runtime.run_atomic(
            fence,
            phase=phase,
            step=scope,
            page_id=page_id,
            data=data,
        )

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

    def _with_credential(
        self,
        section: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = dict(section)
        version_id = result.pop("credentialVersionId", None)
        if not version_id:
            return result
        try:
            secret = self.credentials.resolve_secret(str(version_id))
        except LookupError as exc:
            raise RuntimeError(
                "frozen credential version no longer exists"
            ) from exc
        result.update(secret)
        result["credential_version_id"] = str(version_id)
        return result

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
                payload_json=canonical_json(dict(payload)),
                updated_revision=new_revision,
            )
        )
        if bubble_changed.rowcount != 1:
            raise RuntimeError("bubble no longer exists")
