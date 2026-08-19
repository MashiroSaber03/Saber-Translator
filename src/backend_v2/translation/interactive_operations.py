"""Saved editor operations executed outside the browser lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
import uuid

from PIL import Image
from sqlalchemy import Engine, delete, insert, select, update
from sqlalchemy.engine import Connection

from src.backend_v2.content.page_style import (
    PAGE_STYLE_SCHEMA_VERSION,
    rgb_to_hex,
    validate_page_style,
)
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
    _payload_text,
    _preserve_detected_text,
    _require_non_empty_string,
    _require_result_mapping,
    _require_text_list,
    _validate_bubble_inputs,
    _validate_color_results,
    _validate_detected_payloads,
    _validate_detection_result,
    _validate_ocr_results,
)
from src.core.config_models import (
    BUBBLE_PAYLOAD_SCHEMA_VERSION,
    validate_bubble_payload,
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
        kind = operation.get("kind")
        if not isinstance(kind, str):
            raise ValueError("page operation kind is invalid")
        page_id = operation.get("pageId")
        if not isinstance(page_id, str) or not page_id:
            raise ValueError("page operation target is invalid")
        base_revision = operation.get("baseRevision")
        if (
            isinstance(base_revision, bool)
            or not isinstance(base_revision, int)
            or base_revision < 1
        ):
            raise ValueError("page operation base revision is invalid")
        bubble_id = operation.get("bubbleId")
        if kind in {"bubble_ocr", "bubble_color", "bubble_translate"}:
            if not isinstance(bubble_id, str) or not bubble_id:
                raise ValueError("bubble operation target is invalid")
        elif kind == "page_detect" and bubble_id is not None:
            raise ValueError("page detection cannot target a bubble")
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
        page_id = operation["pageId"]
        bubble_id = operation["bubbleId"]
        page, rows = self._snapshot(
            page_id,
            expected_revision=operation["baseRevision"],
        )
        index = self._bubble_index(rows, bubble_id)
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
        input_bubbles = _validate_bubble_inputs(
            before["bubbles"],
            expected_count=1,
            label="single-bubble OCR input",
        )
        source_asset_id = _require_non_empty_string(
            before.get("sourceAssetId"),
            label="single-bubble OCR source asset",
        )
        image = self._open_asset(source_asset_id, "RGB")
        try:
            result = self.algorithms.ocr(
                image,
                input_bubbles,
                self._with_credential(before.get("ocrConfig")),
            )
        finally:
            image.close()
        result = _require_result_mapping(result, label="single-bubble OCR result")
        texts = _require_text_list(
            result.get("texts"),
            label="single-bubble OCR texts",
        )
        details = _validate_ocr_results(
            result.get("results"),
            label="single-bubble OCR details",
        )
        if len(texts) != 1 or len(details) != 1:
            raise RuntimeError("single-bubble OCR returned an invalid result count")
        after = self._atomic_hook(
            fence,
            phase="after",
            scope="ocr",
            page_id=page_id,
            data={
                "pageId": page_id,
                "originalTexts": texts,
                "ocrResults": details,
            },
        )
        texts = _require_text_list(
            after.get("originalTexts"),
            label="single-bubble OCR plugin texts",
        )
        details = _validate_ocr_results(
            after.get("ocrResults"),
            label="single-bubble OCR plugin details",
        )
        if len(texts) != 1 or len(details) != 1:
            raise RuntimeError("single-bubble OCR returned an invalid result count")
        payload = dict(rows[index]["payload"])
        payload["originalText"] = texts[0]
        payload["ocrResult"] = details[0]
        new_revision = int(page["document_revision"]) + 1

        def publish(connection: Connection, _row: Mapping[str, Any]) -> None:
            self._update_one_bubble(
                connection,
                page_id=page_id,
                bubble_id=bubble_id,
                base_revision=int(page["document_revision"]),
                new_revision=new_revision,
                payload=payload,
                render_changed=False,
                expected_bubble_count=len(rows),
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
        page_id = operation["pageId"]
        bubble_id = operation["bubbleId"]
        page, rows = self._snapshot(
            page_id,
            expected_revision=operation["baseRevision"],
        )
        index = self._bubble_index(rows, bubble_id)
        payload = dict(rows[index]["payload"])
        original_text = payload["originalText"].strip()
        if not original_text:
            raise ValueError("bubble has no original text")
        config = self._with_credential(self._payload(operation))
        translated = self.algorithms.translate(
            [original_text],
            config,
            mode="single",
        )
        translated = _require_result_mapping(
            translated,
            label="single-bubble translation result",
        )
        values = _require_text_list(
            translated.get("translated"),
            label="single-bubble translated texts",
        )
        textbox = _require_text_list(
            translated.get("textbox"),
            label="single-bubble textbox texts",
        )
        if len(values) != 1:
            raise RuntimeError(
                "single-bubble translation returned an invalid result count"
            )
        if len(textbox) not in {0, 1}:
            raise RuntimeError(
                "single-bubble textbox translation returned an invalid result count"
            )
        payload["translatedText"] = values[0]
        payload["textboxText"] = textbox[0] if textbox else ""
        new_revision = int(page["document_revision"]) + 1
        has_drawable_text = any(
            _payload_text(
                payload if row_index == index else row["payload"],
                "translatedText",
                label="persisted translated text",
            ).strip()
            for row_index, row in enumerate(rows)
        )

        def publish(connection: Connection, _row: Mapping[str, Any]) -> None:
            has_translated = connection.execute(
                select(page_assets.c.asset_id).where(
                    page_assets.c.page_id == page_id,
                    page_assets.c.role == "translated",
                )
            ).scalar_one_or_none()
            needs_render = has_translated is not None or has_drawable_text
            self._update_one_bubble(
                connection,
                page_id=page_id,
                bubble_id=bubble_id,
                base_revision=int(page["document_revision"]),
                new_revision=new_revision,
                payload=payload,
                render_changed=needs_render,
                expected_bubble_count=len(rows),
            )
            if needs_render:
                self.renders.upsert(
                    connection,
                    page_id=page_id,
                    requested_revision=new_revision,
                    existing_chain=True,
                )

        response = {
            "bubbleId": operation["bubbleId"],
            "translatedText": values[0],
            "documentRevision": new_revision,
        }
        self.repository.complete(fence, result=response, publisher=publish)
        return response

    def _bubble_color(
        self,
        fence: OperationFence,
        operation: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        page_id = operation["pageId"]
        bubble_id = operation["bubbleId"]
        page, rows = self._snapshot(
            page_id,
            expected_revision=operation["baseRevision"],
        )
        index = self._bubble_index(rows, bubble_id)
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
        input_bubbles = _validate_bubble_inputs(
            before["bubbles"],
            expected_count=1,
            label="single-bubble color input",
        )
        source_asset_id = _require_non_empty_string(
            before.get("sourceAssetId"),
            label="single-bubble color source asset",
        )
        image = self._open_asset(source_asset_id, "RGB")
        try:
            colors = self.algorithms.colors(
                image,
                input_bubbles,
            )
        finally:
            image.close()
        colors = _validate_color_results(
            colors,
            label="single-bubble color result",
            plugin_fields=False,
        )
        if len(colors) != 1:
            raise RuntimeError("single-bubble color returned an invalid result count")
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
                        "confidence": value["confidence"],
                    }
                    for value in colors
                ],
            },
        )
        colors = _validate_color_results(
            after.get("colors"),
            label="single-bubble color plugin result",
            plugin_fields=True,
        )
        if len(colors) != 1:
            raise RuntimeError("single-bubble color returned an invalid result count")
        color = colors[0]
        payload = dict(rows[index]["payload"])
        payload["autoFgColor"] = color.get("fg_color")
        payload["autoBgColor"] = color.get("bg_color")
        payload["colorConfidence"] = color["confidence"]
        style_defaults = validate_page_style(
            json.loads(page["page_style_defaults_json"]),
            partial=False,
        )
        uses_auto_color = bool(style_defaults["useAutoTextColor"])
        old_text_color = payload["textColor"]
        if uses_auto_color and color.get("fg_color") is not None:
            payload["textColor"] = rgb_to_hex(color["fg_color"])
        if uses_auto_color and color.get("bg_color") is not None:
            payload["fillColor"] = rgb_to_hex(color["bg_color"])
        changes_render = payload["textColor"] != old_text_color
        new_revision = int(page["document_revision"]) + 1
        has_drawable_text = any(
            _payload_text(
                payload if row_index == index else row["payload"],
                "translatedText",
                label="persisted translated text",
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
                page_id=page_id,
                bubble_id=bubble_id,
                base_revision=int(page["document_revision"]),
                new_revision=new_revision,
                payload=payload,
                render_changed=needs_render,
                expected_bubble_count=len(rows),
            )
            if needs_render:
                self.renders.upsert(
                    connection,
                    page_id=page_id,
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
        page_id = operation["pageId"]
        page, _rows = self._snapshot(
            page_id,
            expected_revision=operation["baseRevision"],
        )
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
        source_asset_id = _require_non_empty_string(
            before.get("sourceAssetId"),
            label="page detection source asset",
        )
        detector_config = before.get("detectorConfig")
        if not isinstance(detector_config, Mapping):
            raise ValueError("page detection configuration is invalid")
        image = self._open_asset(source_asset_id, "RGB")
        source_size = image.size
        try:
            detected = self.algorithms.detect(
                image,
                dict(detector_config),
            )
        finally:
            image.close()
        (
            coords,
            polygons,
            angles,
            directions,
            textlines,
            raw_mask,
        ) = _validate_detection_result(detected)
        style = validate_page_style(
            json.loads(page["page_style_defaults_json"]),
            partial=False,
        )
        payloads = _preserve_detected_text([
            TranslationPipelineService._new_bubble_payload(
                coords=value,
                polygon=polygons[index],
                angle=angles[index],
                auto_direction=directions[index],
                textlines=textlines[index],
                style=style,
            )
            for index, value in enumerate(coords)
        ], tuple(row["payload"] for row in _rows))
        mask_record: AssetRecord | None = None
        if isinstance(raw_mask, Image.Image):
            try:
                if raw_mask.size != source_size:
                    raise ValueError(
                        "page detection mask size does not match source image"
                    )
                mask_record = publish_png_asset(self.storage, raw_mask, mode="L")
            finally:
                raw_mask.close()
        elif raw_mask is not None:
            mask_image = Image.fromarray(raw_mask)
            try:
                if mask_image.size != source_size:
                    raise ValueError(
                        "page detection mask size does not match source image"
                    )
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
        payloads = _validate_detected_payloads(after.get("bubbles"))
        mask_asset_id = after.get("textMaskAssetId")
        if mask_asset_id is not None and (
            not isinstance(mask_asset_id, str) or not mask_asset_id
        ):
            raise ValueError("page detection mask asset is invalid")
        mask_record = (
            self._asset_record(mask_asset_id)
            if mask_asset_id is not None
            else None
        )
        if mask_record is not None:
            stored_mask = self._open_asset(mask_record.id, "L")
            try:
                if stored_mask.size != source_size:
                    raise ValueError(
                        "page detection mask asset size does not match source image"
                    )
            finally:
                stored_mask.close()
        new_revision = int(page["document_revision"]) + 1

        def publish(connection: Connection, _row: Mapping[str, Any]) -> None:
            current = connection.execute(
                select(pages.c.document_revision).where(
                    pages.c.id == page_id
                )
            ).scalar_one_or_none()
            if current != page["document_revision"]:
                raise RuntimeError("page revision changed")
            has_translated_asset = (
                connection.execute(
                    select(page_assets.c.asset_id).where(
                        page_assets.c.page_id == page_id,
                        page_assets.c.role == "translated",
                    )
                ).scalar_one_or_none()
                is not None
            )
            has_drawable_text = any(
                _payload_text(
                    payload,
                    "translatedText",
                    label="detected translated text",
                ).strip()
                for payload in payloads
            )
            needs_render = has_translated_asset or has_drawable_text
            connection.execute(
                delete(bubbles).where(bubbles.c.page_id == page_id)
            )
            if payloads:
                connection.execute(
                    insert(bubbles),
                    [
                        {
                            "id": str(uuid.uuid4()),
                            "page_id": page_id,
                            "ordinal": index,
                            "font_id": page["default_font_id"],
                            "payload_json": canonical_json(payload),
                            "payload_schema_version": BUBBLE_PAYLOAD_SCHEMA_VERSION,
                            "updated_revision": new_revision,
                        }
                        for index, payload in enumerate(payloads, start=1)
                    ],
                )
            changed = connection.execute(
                update(pages)
                .where(
                    pages.c.id == page_id,
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
            if changed.rowcount != 1:
                raise RuntimeError("page revision changed")
            connection.execute(
                delete(page_assets).where(
                    page_assets.c.page_id == page_id,
                    page_assets.c.role == "text_mask",
                )
            )
            if mask_record is not None:
                connection.execute(
                    insert(page_assets).values(
                        page_id=page_id,
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
                    page_id=page_id,
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

    def _snapshot(self, page_id: str, *, expected_revision: int):
        with self.engine.connect() as connection:
            page = connection.execute(
                select(pages).where(pages.c.id == page_id)
            ).mappings().one_or_none()
            if page is None:
                raise RuntimeError("operation target page no longer exists")
            if page["document_revision"] != expected_revision:
                raise RuntimeError("page revision changed")
            if page["page_style_schema_version"] != PAGE_STYLE_SCHEMA_VERSION:
                raise RuntimeError("page style schema version is not current")
            rows = []
            for row in connection.execute(
                select(bubbles)
                .where(bubbles.c.page_id == page_id)
                .order_by(bubbles.c.ordinal)
            ).mappings():
                if row["payload_schema_version"] != BUBBLE_PAYLOAD_SCHEMA_VERSION:
                    raise RuntimeError(
                        "bubble payload schema version is not current"
                    )
                if row["updated_revision"] != page["document_revision"]:
                    raise RuntimeError(
                        "bubble revision does not match page document"
                    )
                try:
                    payload = validate_bubble_payload(
                        json.loads(row["payload_json"]),
                        render=False,
                    )
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "bubble payload does not match the current schema"
                    ) from exc
                rows.append({"id": row["id"], "payload": payload})
        return page, rows

    @staticmethod
    def _input_asset_id(
        operation: Mapping[str, Any],
        role: str,
    ) -> str:
        inputs = operation.get("inputs")
        if not isinstance(inputs, Mapping) or role not in inputs:
            raise RuntimeError(f"operation has no frozen {role} input")
        value = inputs[role]
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"operation has an invalid frozen {role} input")
        return value

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
            raise RuntimeError("operation request snapshot is invalid")
        payload = request.get("payload")
        if not isinstance(payload, Mapping):
            raise RuntimeError("operation payload snapshot is invalid")
        result = dict(payload)
        settings_snapshot = result.pop("settingsSnapshot", None)
        if settings_snapshot is not None and not isinstance(
            settings_snapshot,
            Mapping,
        ):
            raise RuntimeError("operation settings snapshot is invalid")
        return result

    def _with_credential(
        self,
        section: object,
    ) -> dict[str, Any]:
        if not isinstance(section, Mapping):
            raise RuntimeError("frozen provider configuration is invalid")
        result = dict(section)
        version_id = result.pop("credentialVersionId", None)
        if version_id is None:
            return result
        if not isinstance(version_id, str) or not version_id:
            raise RuntimeError("frozen credential version is invalid")
        try:
            secret = self.credentials.resolve_secret(version_id)
        except LookupError as exc:
            raise RuntimeError(
                "frozen credential version no longer exists"
            ) from exc
        result.update(secret)
        result["credential_version_id"] = version_id
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
        expected_bubble_count: int,
    ) -> None:
        payload = validate_bubble_payload(payload, render=False)
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
            pointer_changed = connection.execute(
                update(page_assets)
                .where(
                    page_assets.c.page_id == page_id,
                    page_assets.c.role == "translated",
                    page_assets.c.input_document_revision == base_revision,
                )
                .values(input_document_revision=new_revision)
            )
            if pointer_changed.rowcount != 1:
                raise RuntimeError("current translated asset is missing")
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
        revisions_changed = connection.execute(
            update(bubbles)
            .where(
                bubbles.c.page_id == page_id,
                bubbles.c.updated_revision == base_revision,
            )
            .values(updated_revision=new_revision)
        )
        if revisions_changed.rowcount != expected_bubble_count:
            raise RuntimeError("bubble revisions changed during publication")
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
