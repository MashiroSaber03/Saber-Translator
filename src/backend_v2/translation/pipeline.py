"""Durable per-page translation steps executed exclusively by the Worker."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Protocol
import uuid

from PIL import Image
from sqlalchemy import Engine, delete, insert, select, update
from sqlalchemy.engine import Connection

from src.backend_v2.jobs.repository import AttemptFence, JobConflict, JobQueueRepository
from src.backend_v2.storage.assets import AssetRecord, AssetStorageService
from src.backend_v2.storage.schema import (
    bubbles,
    credential_versions,
    job_step_asset_outputs,
    page_assets,
    pages,
)


@dataclass(frozen=True, slots=True)
class PageSnapshot:
    page_id: str
    source_revision: int
    document_revision: int
    render_status: str
    style_defaults: dict[str, Any]
    bubbles: tuple[dict[str, Any], ...]


class TranslationAlgorithms(Protocol):
    def detect(self, image: Image.Image, config: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def ocr(
        self,
        image: Image.Image,
        bubble_payloads: list[dict[str, Any]],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def colors(
        self,
        image: Image.Image,
        bubble_payloads: list[dict[str, Any]],
    ) -> list[Mapping[str, Any]]: ...

    def translate(
        self,
        texts: list[str],
        config: Mapping[str, Any],
        *,
        mode: str,
    ) -> Mapping[str, Any]: ...

    def repair(
        self,
        image: Image.Image,
        bubble_payloads: list[dict[str, Any]],
        config: Mapping[str, Any],
    ) -> Image.Image: ...

    def render(
        self,
        clean_image: Image.Image,
        bubble_payloads: list[dict[str, Any]],
        config: Mapping[str, Any],
    ) -> Image.Image: ...


class LegacyTranslationAlgorithms:
    """Direct adapters around existing core functions, without HTTP/Base64."""

    def detect(self, image: Image.Image, config: Mapping[str, Any]) -> Mapping[str, Any]:
        from src.core.detection import (
            get_bubble_detection_result_with_auto_directions,
        )

        allowed = {
            "detector_type",
            "expand_ratio",
            "expand_top",
            "expand_bottom",
            "expand_left",
            "expand_right",
            "enable_aux_yolo_detection",
            "aux_yolo_conf_threshold",
            "aux_yolo_overlap_threshold",
            "enable_saber_yolo_refine",
            "saber_yolo_refine_overlap_threshold",
            "min_text_block_area_percent",
        }
        kwargs = {key: value for key, value in config.items() if key in allowed}
        return get_bubble_detection_result_with_auto_directions(image, **kwargs)

    def ocr(
        self,
        image: Image.Image,
        bubble_payloads: list[dict[str, Any]],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        from src.core.ocr import recognize_ocr_results_in_bubbles
        from src.core.ocr_types import (
            extract_texts_from_ocr_results,
            ocr_results_to_dicts,
        )

        coords = [payload.get("coords", [0, 0, 0, 0]) for payload in bubble_payloads]
        textlines = [payload.get("textlines", []) for payload in bubble_payloads]
        allowed = {
            "source_language",
            "ocr_engine",
            "baidu_api_key",
            "baidu_secret_key",
            "baidu_version",
            "baidu_ocr_language",
            "ai_vision_provider",
            "ai_vision_api_key",
            "ai_vision_model_name",
            "ai_vision_ocr_prompt",
            "ai_vision_prompt_mode",
            "custom_ai_vision_base_url",
            "ai_vision_min_image_size",
            "ai_vision_openai_options",
            "enable_hybrid_ocr",
            "secondary_ocr_engine",
            "hybrid_ocr_threshold",
        }
        kwargs = {key: value for key, value in config.items() if key in allowed}
        kwargs["textlines_per_bubble"] = textlines
        results = recognize_ocr_results_in_bubbles(image, coords, **kwargs)
        return {
            "texts": extract_texts_from_ocr_results(results),
            "results": ocr_results_to_dicts(results),
        }

    def colors(
        self,
        image: Image.Image,
        bubble_payloads: list[dict[str, Any]],
    ) -> list[Mapping[str, Any]]:
        from src.core.color_extractor import extract_bubble_colors

        coords = [payload.get("coords", [0, 0, 0, 0]) for payload in bubble_payloads]
        textlines = [payload.get("textlines", []) for payload in bubble_payloads]
        return [
            result.to_dict()
            for result in extract_bubble_colors(image, coords, textlines)
        ]

    def translate(
        self,
        texts: list[str],
        config: Mapping[str, Any],
        *,
        mode: str,
    ) -> Mapping[str, Any]:
        from src.core.translation import translate_text_list

        provider = config.get("provider", config.get("model_provider", "siliconflow"))
        translated = translate_text_list(
            texts,
            target_language=str(config.get("target_language", "zh")),
            model_provider=str(provider),
            api_key=config.get("api_key"),
            model_name=config.get("model_name"),
            prompt_content=config.get("prompt_content"),
            custom_base_url=config.get("custom_base_url"),
        )
        return {"translated": translated, "textbox": [], "mode": mode}

    def repair(
        self,
        image: Image.Image,
        bubble_payloads: list[dict[str, Any]],
        config: Mapping[str, Any],
    ) -> Image.Image:
        from src.core.inpainting import inpaint_bubbles

        coords = [payload.get("coords", [0, 0, 0, 0]) for payload in bubble_payloads]
        polygons = [payload.get("polygon", []) for payload in bubble_payloads]
        repaired, _ = inpaint_bubbles(
            image,
            coords,
            method=str(config.get("method", "solid")),
            fill_color=str(config.get("fill_color", "#FFFFFF")),
            bubble_polygons=polygons,
            mask_dilate_size=int(config.get("mask_dilate_size", 0)),
            mask_box_expand_ratio=float(config.get("mask_box_expand_ratio", 0)),
            lama_model=str(config.get("lama_model", "lama_mpe")),
        )
        return repaired

    def render(
        self,
        clean_image: Image.Image,
        bubble_payloads: list[dict[str, Any]],
        config: Mapping[str, Any],
    ) -> Image.Image:
        from src.core.config_models import BubbleState
        from src.core.rendering import render_bubbles_unified

        states = [BubbleState.from_dict(payload) for payload in bubble_payloads]
        rendered = clean_image.copy()
        render_bubbles_unified(rendered, states)
        return rendered


class TranslationPipelineService:
    def __init__(
        self,
        *,
        data_root: Path,
        engine: Engine,
        jobs: JobQueueRepository,
        algorithms: TranslationAlgorithms | None = None,
    ) -> None:
        self.data_root = data_root
        self.engine = engine
        self.jobs = jobs
        self.storage = AssetStorageService(data_root, engine)
        self.algorithms = algorithms or LegacyTranslationAlgorithms()

    def handler(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Execute one atomic step and publish under the latest heartbeat fence."""

        page_id = step.get("pageId")
        if not isinstance(page_id, str):
            raise ValueError("translation step has no page")
        kind = str(step["stepKind"])
        if kind == "detect":
            result = self._detect(fence, step, page_id)
        elif kind == "ocr":
            result = self._ocr(fence, step, page_id)
        elif kind == "color":
            result = self._color(fence, step, page_id)
        elif kind == "auto_terms":
            result = self._checkpoint_only(fence, step, {"delta": []})
        elif kind in {"translate", "hq_translate", "proofread"}:
            result = self._translate(fence, step, page_id, kind)
        elif kind == "repair":
            result = self._repair(fence, step, page_id)
        elif kind == "render":
            result = self._render(fence, step, page_id)
        elif kind == "publish_clean":
            result = self._checkpoint_only(
                fence, step, {"published": "clean"}
            )
        else:
            raise ValueError(f"unsupported translation step: {kind}")
        return {**result, "__already_published__": True}

    def _detect(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
        page_id: str,
    ) -> Mapping[str, Any]:
        snapshot = self._snapshot(page_id)
        image = self._open_bound_image(fence, step, page_id, "source")
        try:
            result = self.algorithms.detect(
                image,
                self._config(step).get("detector", {}),
            )
        finally:
            image.close()
        coords = list(result.get("coords", []))
        polygons = list(result.get("polygons", []))
        angles = list(result.get("angles", []))
        directions = list(result.get("auto_directions", []))
        textlines = list(result.get("textlines_per_bubble", []))
        payloads = [
            self._new_bubble_payload(
                coords=value,
                polygon=polygons[index] if index < len(polygons) else [],
                angle=angles[index] if index < len(angles) else 0,
                auto_direction=(
                    directions[index] if index < len(directions) else "vertical"
                ),
                textlines=textlines[index] if index < len(textlines) else [],
                style=snapshot.style_defaults,
            )
            for index, value in enumerate(coords)
        ]
        mask_record: AssetRecord | None = None
        mask = result.get("raw_mask")
        if isinstance(mask, Image.Image):
            mask_record = self._publish_image(mask, mode="L")
        elif mask is not None:
            mask_record = self._publish_image(Image.fromarray(mask), mode="L")
        new_revision = snapshot.document_revision + 1

        def publish(connection: Connection) -> None:
            self._assert_revision(
                connection, page_id, snapshot.document_revision
            )
            connection.execute(delete(bubbles).where(bubbles.c.page_id == page_id))
            if payloads:
                connection.execute(
                    insert(bubbles),
                    [
                        {
                            "id": str(uuid.uuid4()),
                            "page_id": page_id,
                            "ordinal": index,
                            "payload_json": _json(payload),
                            "payload_schema_version": 1,
                            "updated_revision": new_revision,
                        }
                        for index, payload in enumerate(payloads, start=1)
                    ],
                )
            connection.execute(
                update(pages)
                .where(
                    pages.c.id == page_id,
                    pages.c.document_revision == snapshot.document_revision,
                )
                .values(
                    document_revision=new_revision,
                    detection_state="processed",
                    render_status="stale",
                )
            )
            if mask_record is not None:
                self._publish_pointer(
                    connection,
                    page_id=page_id,
                    role="text_mask",
                    asset=mask_record,
                    source_revision=snapshot.source_revision,
                    document_revision=new_revision,
                    step_id=str(step["stepId"]),
                )

        checkpoint = {
            "bubbleCount": len(payloads),
            "documentRevision": new_revision,
        }
        self.jobs.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint=checkpoint,
            publisher=publish,
        )
        return checkpoint

    def _ocr(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
        page_id: str,
    ) -> Mapping[str, Any]:
        snapshot = self._snapshot(page_id)
        image = self._open_bound_image(fence, step, page_id, "source")
        try:
            section = self._with_credential(self._config(step).get("ocr", {}))
            section.setdefault(
                "source_language",
                self._config(step).get("sourceLanguage", "japanese"),
            )
            result = self.algorithms.ocr(
                image,
                [dict(value) for value in snapshot.bubbles],
                section,
            )
        finally:
            image.close()
        texts = list(result.get("texts", []))
        details = list(result.get("results", []))
        updated = [dict(payload) for payload in snapshot.bubbles]
        if len(texts) != len(updated):
            raise JobConflict("OCR result count does not match persisted bubbles")
        for index, payload in enumerate(updated):
            payload["originalText"] = str(texts[index])
            payload["ocrResult"] = details[index] if index < len(details) else None
        return self._publish_bubble_update(
            fence,
            step,
            snapshot,
            updated,
            {"recognized": len(texts)},
        )

    def _color(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
        page_id: str,
    ) -> Mapping[str, Any]:
        snapshot = self._snapshot(page_id)
        image = self._open_bound_image(fence, step, page_id, "source")
        try:
            colors = self.algorithms.colors(
                image,
                [dict(value) for value in snapshot.bubbles],
            )
        finally:
            image.close()
        if len(colors) != len(snapshot.bubbles):
            raise JobConflict("color result count does not match persisted bubbles")
        updated = [dict(payload) for payload in snapshot.bubbles]
        for payload, color in zip(updated, colors):
            foreground = color.get("fg_color")
            background = color.get("bg_color")
            payload["autoFgColor"] = foreground
            payload["autoBgColor"] = background
            payload["colorConfidence"] = float(color.get("confidence", 0))
            if foreground is not None:
                payload["textColor"] = _rgb_hex(foreground)
            if background is not None:
                payload["fillColor"] = _rgb_hex(background)
        return self._publish_bubble_update(
            fence,
            step,
            snapshot,
            updated,
            {"colored": len(colors)},
        )

    def _translate(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
        page_id: str,
        mode: str,
    ) -> Mapping[str, Any]:
        snapshot = self._snapshot(page_id)
        texts = [str(payload.get("originalText", "")) for payload in snapshot.bubbles]
        section = self._with_credential(
            self._config(step).get("translation", {})
        )
        section.setdefault(
            "target_language",
            self._config(step).get("targetLanguage", "zh"),
        )
        result = self.algorithms.translate(texts, section, mode=mode)
        translated = list(result.get("translated", []))
        textbox = list(result.get("textbox", []))
        if len(translated) != len(snapshot.bubbles):
            raise JobConflict("translation result count does not match bubbles")
        updated = [dict(payload) for payload in snapshot.bubbles]
        for index, payload in enumerate(updated):
            payload["translatedText"] = str(translated[index])
            payload["textboxText"] = (
                str(textbox[index]) if index < len(textbox) else ""
            )
        return self._publish_bubble_update(
            fence,
            step,
            snapshot,
            updated,
            {"translated": len(translated), "mode": mode},
        )

    def _repair(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
        page_id: str,
    ) -> Mapping[str, Any]:
        snapshot = self._snapshot(page_id)
        image = self._open_bound_image(fence, step, page_id, "source")
        try:
            repaired = self.algorithms.repair(
                image,
                [dict(value) for value in snapshot.bubbles],
                self._config(step).get("inpainting", {}),
            )
        finally:
            image.close()
        record = self._publish_image(repaired)
        repaired.close()

        def publish(connection: Connection) -> None:
            self._assert_revision(
                connection, page_id, snapshot.document_revision
            )
            self._publish_pointer(
                connection,
                page_id=page_id,
                role="clean",
                asset=record,
                source_revision=snapshot.source_revision,
                document_revision=snapshot.document_revision,
                step_id=str(step["stepId"]),
            )

        checkpoint = {
            "cleanAssetId": record.id,
            "documentRevision": snapshot.document_revision,
        }
        self.jobs.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint=checkpoint,
            publisher=publish,
        )
        return checkpoint

    def _render(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
        page_id: str,
    ) -> Mapping[str, Any]:
        snapshot = self._snapshot(page_id)
        from src.backend_v2.rendering.fonts import (
            materialize_render_payloads,
        )

        with self.engine.connect() as connection:
            projected = materialize_render_payloads(
                connection,
                self.storage,
                page_id,
            )
        persisted_payloads = [
            (bubble_id, payload)
            for bubble_id, payload, _render_payload in projected
        ]
        render_payloads = [
            render_payload
            for _bubble_id, _payload, render_payload in projected
        ]
        try:
            clean = self._open_bound_image(fence, step, page_id, "clean")
        except JobConflict:
            clean = self._open_bound_image(fence, step, page_id, "source")
        try:
            rendered = self.algorithms.render(
                clean,
                render_payloads,
                self._config(step).get("render", {}),
            )
        finally:
            clean.close()
        translated = self._publish_image(rendered)
        thumbnail = self._publish_thumbnail(rendered)
        rendered.close()

        def publish(connection: Connection) -> None:
            self._assert_revision(
                connection, page_id, snapshot.document_revision
            )
            for bubble_id, payload in persisted_payloads:
                connection.execute(
                    update(bubbles)
                    .where(
                        bubbles.c.id == bubble_id,
                        bubbles.c.page_id == page_id,
                        bubbles.c.updated_revision
                        <= snapshot.document_revision,
                    )
                    .values(
                        payload_json=_json(payload),
                        updated_revision=snapshot.document_revision,
                    )
                )
            self._publish_pointer(
                connection,
                page_id=page_id,
                role="translated",
                asset=translated,
                source_revision=snapshot.source_revision,
                document_revision=snapshot.document_revision,
                step_id=str(step["stepId"]),
            )
            self._publish_pointer(
                connection,
                page_id=page_id,
                role="thumbnail_translated",
                asset=thumbnail,
                source_revision=snapshot.source_revision,
                document_revision=snapshot.document_revision,
                step_id=str(step["stepId"]),
                parent_asset_id=translated.id,
            )
            connection.execute(
                update(pages)
                .where(
                    pages.c.id == page_id,
                    pages.c.document_revision == snapshot.document_revision,
                )
                .values(
                    rendered_revision=snapshot.document_revision,
                    render_status="ready",
                )
            )

        checkpoint = {
            "translatedAssetId": translated.id,
            "thumbnailAssetId": thumbnail.id,
            "documentRevision": snapshot.document_revision,
        }
        self.jobs.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint=checkpoint,
            publisher=publish,
        )
        return checkpoint

    def _publish_bubble_update(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
        snapshot: PageSnapshot,
        payloads: list[dict[str, Any]],
        checkpoint: dict[str, Any],
    ) -> Mapping[str, Any]:
        new_revision = snapshot.document_revision + 1

        def publish(connection: Connection) -> None:
            self._assert_revision(
                connection, snapshot.page_id, snapshot.document_revision
            )
            rows = list(
                connection.execute(
                    select(bubbles.c.id)
                    .where(bubbles.c.page_id == snapshot.page_id)
                    .order_by(bubbles.c.ordinal)
                ).scalars()
            )
            if len(rows) != len(payloads):
                raise JobConflict("bubble set changed before step publication")
            for bubble_id, payload in zip(rows, payloads):
                connection.execute(
                    update(bubbles)
                    .where(
                        bubbles.c.id == bubble_id,
                        bubbles.c.updated_revision
                        == snapshot.document_revision,
                    )
                    .values(
                        payload_json=_json(payload),
                        updated_revision=new_revision,
                    )
                )
            changed = connection.execute(
                update(pages)
                .where(
                    pages.c.id == snapshot.page_id,
                    pages.c.document_revision == snapshot.document_revision,
                )
                .values(
                    document_revision=new_revision,
                    render_status="stale",
                )
            )
            if changed.rowcount != 1:
                raise JobConflict("page revision changed during step publication")

        checkpoint = {**checkpoint, "documentRevision": new_revision}
        self.jobs.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint=checkpoint,
            publisher=publish,
        )
        return checkpoint

    def _checkpoint_only(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
        checkpoint: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.jobs.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint=checkpoint,
        )
        return checkpoint

    def _snapshot(self, page_id: str) -> PageSnapshot:
        with self.engine.connect() as connection:
            page = connection.execute(
                select(pages).where(pages.c.id == page_id)
            ).mappings().one_or_none()
            if page is None:
                raise JobConflict("job target page no longer exists")
            rows = connection.execute(
                select(bubbles.c.payload_json)
                .where(bubbles.c.page_id == page_id)
                .order_by(bubbles.c.ordinal)
            ).scalars()
            return PageSnapshot(
                page_id=page_id,
                source_revision=int(page["source_revision"]),
                document_revision=int(page["document_revision"]),
                render_status=str(page["render_status"]),
                style_defaults=json.loads(page["page_style_defaults_json"] or "{}"),
                bubbles=tuple(json.loads(value) for value in rows),
            )

    def _open_bound_image(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
        page_id: str,
        role: str,
    ) -> Image.Image:
        bound = self.jobs.bind_item_inputs(
            fence,
            item_id=str(step["itemId"]),
            page_id=page_id,
            roles=(role,),
        )[role]
        path = self.storage.resolve_relative_path(str(bound["relative_path"]))
        image = Image.open(path)
        if image.mode != "RGB":
            converted = image.convert("RGB")
            image.close()
            image = converted
        else:
            image.load()
        return image

    def _config(self, step: Mapping[str, Any]) -> dict[str, Any]:
        value = step.get("config", {})
        return dict(value) if isinstance(value, Mapping) else {}

    def _with_credential(self, section: object) -> dict[str, Any]:
        result = dict(section) if isinstance(section, Mapping) else {}
        version_id = result.pop("credentialVersionId", None)
        if version_id:
            with self.engine.connect() as connection:
                value = connection.execute(
                    select(credential_versions.c.secret_json).where(
                        credential_versions.c.id == version_id
                    )
                ).scalar_one_or_none()
            if value is None:
                raise JobConflict("frozen credential version no longer exists")
            secret = json.loads(value)
            if isinstance(secret, dict):
                result.update(secret)
        return result

    def _publish_image(
        self,
        image: Image.Image,
        *,
        mode: str = "RGB",
    ) -> AssetRecord:
        converted = image if image.mode == mode else image.convert(mode)
        output = BytesIO()
        converted.save(output, format="PNG")
        if converted is not image:
            converted.close()
        return self.storage.publish_bytes(
            output.getvalue(),
            extension="png",
            mime_type="image/png",
            width=image.width,
            height=image.height,
        )

    def _publish_thumbnail(self, image: Image.Image) -> AssetRecord:
        thumbnail = image.copy()
        if thumbnail.height / max(thumbnail.width, 1) > 4:
            if thumbnail.width > 320:
                height = max(1, round(thumbnail.height * 320 / thumbnail.width))
                thumbnail = thumbnail.resize((320, height), Image.Resampling.LANCZOS)
            if thumbnail.height > 1280:
                thumbnail = thumbnail.crop((0, 0, thumbnail.width, 1280))
        else:
            thumbnail.thumbnail((320, 320), Image.Resampling.LANCZOS)
        output = BytesIO()
        thumbnail.save(output, format="WEBP", quality=80, method=4)
        width, height = thumbnail.size
        thumbnail.close()
        return self.storage.publish_bytes(
            output.getvalue(),
            extension="webp",
            mime_type="image/webp",
            width=width,
            height=height,
        )

    @staticmethod
    def _assert_revision(
        connection: Connection,
        page_id: str,
        expected: int,
    ) -> None:
        current = connection.execute(
            select(pages.c.document_revision).where(pages.c.id == page_id)
        ).scalar_one_or_none()
        if current != expected:
            raise JobConflict("page document revision changed")

    @staticmethod
    def _publish_pointer(
        connection: Connection,
        *,
        page_id: str,
        role: str,
        asset: AssetRecord,
        source_revision: int,
        document_revision: int,
        step_id: str,
        parent_asset_id: str | None = None,
    ) -> None:
        existing = connection.execute(
            select(page_assets.c.asset_id).where(
                page_assets.c.page_id == page_id,
                page_assets.c.role == role,
            )
        ).scalar_one_or_none()
        values = {
            "asset_id": asset.id,
            "input_source_revision": source_revision,
            "input_document_revision": document_revision,
            "parent_asset_id": parent_asset_id,
            "producer_job_step_id": step_id,
            "producer_operation_id": None,
            "producer_render_request_id": None,
        }
        if existing is None:
            connection.execute(
                insert(page_assets).values(page_id=page_id, role=role, **values)
            )
        else:
            connection.execute(
                update(page_assets)
                .where(
                    page_assets.c.page_id == page_id,
                    page_assets.c.role == role,
                )
                .values(**values)
            )
        connection.execute(
            insert(job_step_asset_outputs)
            .values(job_step_id=step_id, role=role, asset_id=asset.id)
            .prefix_with("OR REPLACE")
        )

    @staticmethod
    def _new_bubble_payload(
        *,
        coords: object,
        polygon: object,
        angle: object,
        auto_direction: object,
        textlines: object,
        style: Mapping[str, Any],
    ) -> dict[str, Any]:
        direction = str(auto_direction)
        if direction == "v":
            direction = "vertical"
        elif direction == "h":
            direction = "horizontal"
        if direction not in {"vertical", "horizontal"}:
            direction = "vertical"
        defaults = {
            "originalText": "",
            "translatedText": "",
            "textboxText": "",
            "coords": list(coords) if isinstance(coords, (list, tuple)) else [0, 0, 0, 0],
            "polygon": polygon if isinstance(polygon, list) else [],
            "fontSize": 25,
            "fontFamily": "",
            "textDirection": direction,
            "autoTextDirection": direction,
            "textColor": "#000000",
            "fillColor": "#FFFFFF",
            "rotationAngle": float(angle or 0),
            "position": {"x": 0, "y": 0},
            "strokeEnabled": False,
            "strokeColor": "#FFFFFF",
            "strokeWidth": 2,
            "lineSpacing": 1.0,
            "textAlign": "center",
            "inpaintMethod": "solid",
            "autoFgColor": None,
            "autoBgColor": None,
            "colorConfidence": 0,
            "textlines": textlines if isinstance(textlines, list) else [],
            "ocrResult": None,
        }
        for key in defaults.keys() & style.keys():
            defaults[key] = style[key]
        return defaults


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _rgb_hex(value: object) -> str:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return "#000000"
    red, green, blue = (max(0, min(255, int(part))) for part in value[:3])
    return f"#{red:02X}{green:02X}{blue:02X}"
