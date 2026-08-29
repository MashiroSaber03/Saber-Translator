"""Unified solid/LaMA/restore repair creation and execution."""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping

from PIL import Image, ImageDraw, UnidentifiedImageError
from sqlalchemy import Engine, insert, select, update
from sqlalchemy.engine import Connection

from src.backend_v2.auth.ownership import effective_owner_id
from src.backend_v2.operations.repository import (
    OperationConflict,
    OperationFence,
    OperationRepository,
    RenderRequestRepository,
)
from src.backend_v2.settings.resolver import SettingsResolver
from src.backend_v2.storage.assets import AssetRecord, AssetStorageService
from src.backend_v2.storage.schema import (
    assets,
    books,
    bubbles,
    chapters,
    page_assets,
    pages,
)
from src.core.bubble_geometry import rotated_box_polygon
from src.core.config_models import validate_bubble_payload
from src.shared.user_logging import log_result


class PageRepairService:
    def __init__(
        self,
        *,
        data_root: Path,
        engine: Engine,
        repository: OperationRepository,
        plugin_runtime: Any | None = None,
        method_validator: Callable[[str], None] | None = None,
        settings_transformer: (
            Callable[[Mapping[str, Any]], dict[str, Any]] | None
        ) = None,
    ) -> None:
        self.engine = engine
        self.repository = repository
        self.storage = AssetStorageService(data_root, engine)
        self.renders = RenderRequestRepository(engine)
        self.settings = SettingsResolver(engine)
        self.plugin_runtime = plugin_runtime
        self.method_validator = method_validator
        self.settings_transformer = settings_transformer

    def create_for_bubble(
        self,
        *,
        page_id: str,
        bubble_id: str,
        base_revision: int,
        idempotency_key: str,
    ) -> tuple[dict[str, object], bool]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    pages.c.document_revision,
                    assets.c.width,
                    assets.c.height,
                    bubbles.c.payload_json,
                    bubbles.c.updated_revision,
                )
                .join(bubbles, bubbles.c.page_id == pages.c.id)
                .join(chapters, chapters.c.id == pages.c.chapter_id)
                .join(books, books.c.id == chapters.c.book_id)
                .join(
                    page_assets,
                    (page_assets.c.page_id == pages.c.id)
                    & (page_assets.c.role == "source"),
                )
                .join(assets, assets.c.id == page_assets.c.asset_id)
                .where(
                    pages.c.id == page_id,
                    bubbles.c.id == bubble_id,
                    books.c.owner_user_id == effective_owner_id(),
                )
            ).mappings().one_or_none()
        if row is None:
            raise ValueError("page or bubble not found")
        if row["updated_revision"] != row["document_revision"]:
            raise ValueError("bubble revision does not match page document")
        payload = validate_bubble_payload(
            json.loads(row["payload_json"]),
            render=False,
        )
        width, height = int(row["width"]), int(row["height"])
        method = payload["inpaintMethod"]
        fill_color = payload["fillColor"] if method == "solid" else None
        base_revision, method, fill_color = (
            self.repository.validate_page_repair_identity(
                base_revision=base_revision,
                method=method,
                fill_color=fill_color,
            )
        )
        if self.method_validator is not None:
            self.method_validator(method)
        mask = Image.new("L", (width, height), 0)
        try:
            draw = ImageDraw.Draw(mask)
            polygon = rotated_box_polygon(
                payload["coords"],
                payload["rotationAngle"],
            )
            draw.polygon(
                [tuple(point) for point in polygon],
                fill=255,
            )
            if mask.getbbox() is None:
                raise ValueError("bubble repair mask is empty")
            mask_payload = self._mask_payload(mask)
        finally:
            mask.close()
        mask_checksum = hashlib.sha256(mask_payload).hexdigest()
        replay = self.repository.find_page_repair_replay(
            page_id=page_id,
            base_revision=base_revision,
            method=method,
            fill_color=fill_color,
            mask_checksum=mask_checksum,
            idempotency_key=idempotency_key,
        )
        if replay is not None:
            return replay, True
        if row["document_revision"] != base_revision:
            raise OperationConflict("page document revision changed")
        repair_settings = (
            self.settings.resolve_page_repair(page_id=page_id)
            if method in {"lama_mpe", "litelama"}
            else {"disableResize": False, "settingsSnapshot": {}}
        )
        if self.settings_transformer is not None:
            repair_settings = self.settings_transformer(repair_settings)
        mask_asset = self._publish_mask_payload(
            mask_payload,
            width=width,
            height=height,
        )
        disable_resize = repair_settings["disableResize"]
        if not isinstance(disable_resize, bool):
            raise RuntimeError("repair disableResize must be boolean")
        return self.repository.create_page_repair(
            page_id=page_id,
            base_revision=base_revision,
            method=method,
            fill_color=fill_color,
            disable_resize=disable_resize,
            settings_snapshot=repair_settings["settingsSnapshot"],
            mask_asset_id=mask_asset.id,
            mask_checksum=mask_checksum,
            idempotency_key=idempotency_key,
        )

    def create_for_mask(
        self,
        *,
        page_id: str,
        upload: BinaryIO,
        base_revision: int,
        method: str,
        fill_color: str | None,
        idempotency_key: str,
    ) -> tuple[dict[str, object], bool]:
        base_revision, method, fill_color = (
            self.repository.validate_page_repair_identity(
                base_revision=base_revision,
                method=method,
                fill_color=fill_color,
            )
        )
        if self.method_validator is not None:
            self.method_validator(method)
        try:
            with Image.open(upload) as opened:
                if (
                    opened.format != "PNG"
                    or opened.mode != "L"
                    or getattr(opened, "n_frames", 1) != 1
                ):
                    raise ValueError(
                        "repair mask must be a single-frame 8-bit grayscale PNG"
                    )
                mask = opened.copy()
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(
                "repair mask must be a single-frame 8-bit grayscale PNG"
            ) from exc
        try:
            colors = mask.getcolors(maxcolors=3)
            if (
                colors is None
                or any(value not in {0, 255} for _count, value in colors)
                or not any(value == 255 for _count, value in colors)
            ):
                raise ValueError(
                    "repair mask must be binary (0=keep, 255=repair) and non-empty"
                )
            with self.engine.connect() as connection:
                page = connection.execute(
                    select(
                        assets.c.width,
                        assets.c.height,
                        pages.c.document_revision,
                    )
                    .join(page_assets, page_assets.c.asset_id == assets.c.id)
                    .join(pages, pages.c.id == page_assets.c.page_id)
                    .join(chapters, chapters.c.id == pages.c.chapter_id)
                    .join(books, books.c.id == chapters.c.book_id)
                    .where(
                        page_assets.c.page_id == page_id,
                        page_assets.c.role == "source",
                        books.c.owner_user_id == effective_owner_id(),
                    )
                ).one_or_none()
            if page is None or mask.size != (int(page[0]), int(page[1])):
                raise ValueError(
                    "repair mask dimensions must match the source image"
                )
            mask_payload = self._mask_payload(mask)
        finally:
            mask.close()
        mask_checksum = hashlib.sha256(mask_payload).hexdigest()
        replay = self.repository.find_page_repair_replay(
            page_id=page_id,
            base_revision=base_revision,
            method=method,
            fill_color=fill_color,
            mask_checksum=mask_checksum,
            idempotency_key=idempotency_key,
        )
        if replay is not None:
            return replay, True
        if int(page[2]) != base_revision:
            raise OperationConflict("page document revision changed")
        repair_settings = (
            self.settings.resolve_page_repair(page_id=page_id)
            if method in {"lama_mpe", "litelama"}
            else {"disableResize": False, "settingsSnapshot": {}}
        )
        if self.settings_transformer is not None:
            repair_settings = self.settings_transformer(repair_settings)
        mask_asset = self._publish_mask_payload(
            mask_payload,
            width=int(page[0]),
            height=int(page[1]),
        )
        disable_resize = repair_settings["disableResize"]
        if not isinstance(disable_resize, bool):
            raise RuntimeError("repair disableResize must be boolean")
        return self.repository.create_page_repair(
            page_id=page_id,
            base_revision=base_revision,
            method=method,
            fill_color=fill_color,
            disable_resize=disable_resize,
            settings_snapshot=repair_settings["settingsSnapshot"],
            mask_asset_id=mask_asset.id,
            mask_checksum=mask_checksum,
            idempotency_key=idempotency_key,
        )

    def handle(
        self,
        fence: OperationFence,
        operation: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        request = operation.get("request")
        if not isinstance(request, Mapping):
            raise RuntimeError("repair operation request is invalid")
        method = request.get("method")
        fill_color = request.get("fillColor")
        revision = operation.get("baseRevision")
        revision, method, fill_color = (
            self.repository.validate_page_repair_identity(
                base_revision=revision,
                method=method,
                fill_color=fill_color,
            )
        )
        frozen_method = method
        if method in {"lama_mpe", "litelama"} and not isinstance(
            request.get("disableResize"),
            bool,
        ):
            raise RuntimeError("repair disableResize must be boolean")
        inputs = operation.get("inputs")
        if not isinstance(inputs, Mapping):
            raise RuntimeError("repair operation inputs are missing")
        page_id = self._required_text(operation, "pageId")
        with self.engine.connect() as connection:
            bubble_payloads: list[dict[str, Any]] = []
            for row in connection.execute(
                select(
                    bubbles.c.payload_json,
                    bubbles.c.updated_revision,
                )
                .where(bubbles.c.page_id == page_id)
                .order_by(bubbles.c.ordinal)
            ).mappings():
                if row["updated_revision"] != revision:
                    raise RuntimeError(
                        "bubble revision does not match page document"
                    )
                bubble_payload = validate_bubble_payload(
                    json.loads(row["payload_json"]),
                    render=False,
                )
                bubble_payloads.append(bubble_payload)
        source_asset_id = self._required_text(inputs, "source")
        input_asset_id = self._required_text(
            inputs,
            "parent_clean" if "parent_clean" in inputs else "source",
        )
        repair_mask_asset_id = self._required_text(inputs, "repair_mask")
        before = self._atomic_hook(
            fence,
            phase="before",
            page_id=page_id,
            data={
                "pageId": page_id,
                "sourceAssetId": source_asset_id,
                "inputAssetId": input_asset_id,
                "textMaskAssetId": repair_mask_asset_id,
                "bubbles": bubble_payloads,
                "method": method,
                "fillColor": fill_color,
            },
        )
        method = before.get("method")
        fill_color = before.get("fillColor")
        revision, method, fill_color = (
            self.repository.validate_page_repair_identity(
                base_revision=revision,
                method=method,
                fill_color=fill_color,
            )
        )
        if method != frozen_method:
            raise RuntimeError("repair plugin cannot change the frozen method")
        with ExitStack() as opened_assets:
            source = opened_assets.enter_context(
                self._open_asset(
                    self._required_text(before, "sourceAssetId"),
                    "RGB",
                )
            )
            parent = opened_assets.enter_context(
                self._open_asset(
                    self._required_text(before, "inputAssetId"),
                    "RGB",
                )
            )
            mask = opened_assets.enter_context(
                self._open_asset(
                    self._required_text(before, "textMaskAssetId"),
                    "L",
                )
            )
            if parent.size != source.size or mask.size != source.size:
                raise RuntimeError("repair assets must have identical dimensions")
            expected_size = source.size
            if method == "solid":
                repaired = parent.copy()
                fill = Image.new(
                    "RGB",
                    repaired.size,
                    fill_color,
                )
                try:
                    repaired.paste(fill, mask=mask)
                finally:
                    fill.close()
            elif method == "restore_source":
                repaired = parent.copy()
                repaired.paste(source, mask=mask)
            elif method in {"lama_mpe", "litelama"}:
                import numpy as np
                from src.core.inpainting import inpaint_bubbles

                repaired = inpaint_bubbles(
                    parent,
                    [(0, 0, parent.width, parent.height)],
                    method="lama",
                    user_mask=np.array(mask),
                    lama_model=method,
                    disable_resize=request["disableResize"],
                )
            else:
                raise RuntimeError(f"unsupported repair method: {method}")
            try:
                record = self._publish_png(repaired)
            finally:
                repaired.close()

        after = self._atomic_hook(
            fence,
            phase="after",
            page_id=page_id,
            data={
                "pageId": page_id,
                "cleanAssetId": record.id,
                "documentRevision": revision,
            },
        )
        record = self._asset_record(
            self._required_text(after, "cleanAssetId")
        )
        if (
            record.mime_type != "image/png"
            or (record.width, record.height) != expected_size
        ):
            raise RuntimeError("repair output asset does not match the source image")
        with self._open_asset(record.id, "RGB") as published:
            if published.size != expected_size:
                raise RuntimeError(
                    "repair output image does not match the source image"
                )

        def publish(connection: Connection, _row: Mapping[str, Any]) -> None:
            source_revision = int(
                connection.execute(
                    select(pages.c.source_revision).where(pages.c.id == page_id)
                ).scalar_one()
            )
            existing = connection.execute(
                select(page_assets.c.asset_id).where(
                    page_assets.c.page_id == page_id,
                    page_assets.c.role == "clean",
                )
            ).scalar_one_or_none()
            values = {
                "asset_id": record.id,
                "input_source_revision": source_revision,
                "input_document_revision": revision,
                "parent_asset_id": None,
                "producer_job_step_id": None,
                "producer_operation_id": fence.operation_id,
                "producer_render_request_id": None,
            }
            if existing is None:
                connection.execute(
                    insert(page_assets).values(
                        page_id=page_id,
                        role="clean",
                        **values,
                    )
                )
            else:
                connection.execute(
                    update(page_assets)
                    .where(
                        page_assets.c.page_id == page_id,
                        page_assets.c.role == "clean",
                    )
                    .values(**values)
                )
            connection.execute(
                update(pages)
                .where(
                    pages.c.id == page_id,
                    pages.c.document_revision == revision,
                )
                .values(render_status="stale")
            )
            self.renders.upsert(
                connection,
                page_id=page_id,
                requested_revision=revision,
                existing_chain=True,
            )

        result = {
            "cleanAssetId": record.id,
            "documentRevision": revision,
            "method": method,
        }
        self.repository.complete(fence, result=result, publisher=publish)
        method_labels = {
            "solid": "纯色填充",
            "restore_source": "恢复原图",
            "lama_mpe": "LaMA MPE",
            "litelama": "LiteLaMA",
        }
        log_result(
            "当前页文字修复完成",
            (
                f"方式：{method_labels.get(method, method)}",
                f"输出尺寸：{expected_size[0]}×{expected_size[1]}",
            ),
        )
        return {**result, "__already_published__": True}

    @staticmethod
    def _required_text(value: Mapping[str, Any], field: str) -> str:
        result = value.get(field)
        if not isinstance(result, str) or not result:
            raise RuntimeError(f"repair {field} must be a non-empty string")
        return result

    @staticmethod
    def _mask_payload(mask: Image.Image) -> bytes:
        with BytesIO() as output:
            mask.save(output, format="PNG")
            return output.getvalue()

    def _publish_mask_payload(
        self,
        payload: bytes,
        *,
        width: int,
        height: int,
    ) -> AssetRecord:
        return self.storage.publish_bytes(
            payload,
            extension="png",
            mime_type="image/png",
            width=width,
            height=height,
        )

    def _publish_png(self, image: Image.Image) -> AssetRecord:
        with BytesIO() as output:
            image.save(output, format="PNG")
            return self.storage.publish_bytes(
                output.getvalue(),
                extension="png",
                mime_type="image/png",
                width=image.width,
                height=image.height,
            )

    def _asset_record(self, asset_id: str) -> AssetRecord:
        record = self.storage.get_record(asset_id)
        if record is None:
            raise RuntimeError("plugin referenced an unknown asset")
        return record

    def _open_asset(self, asset_id: str, mode: str) -> Image.Image:
        with self.engine.connect() as connection:
            relative = connection.execute(
                select(assets.c.relative_path).where(assets.c.id == asset_id)
            ).scalar_one()
        with Image.open(self.storage.resolve_relative_path(relative)) as opened:
            return opened.convert(mode)

    def _atomic_hook(
        self,
        fence: OperationFence,
        *,
        phase: str,
        page_id: str,
        data: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.plugin_runtime is None:
            return dict(data)
        return self.plugin_runtime.run_atomic(
            fence,
            phase=phase,
            step="inpaint",
            page_id=page_id,
            data=data,
        )
