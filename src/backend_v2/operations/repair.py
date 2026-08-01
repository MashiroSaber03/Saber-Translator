"""Unified solid/LaMA/restore repair creation and execution."""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any, BinaryIO, Mapping

from PIL import Image, ImageDraw
from sqlalchemy import Engine, insert, select, update
from sqlalchemy.engine import Connection

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
    bubbles,
    page_assets,
    pages,
)


class PageRepairService:
    def __init__(
        self,
        *,
        data_root: Path,
        engine: Engine,
        repository: OperationRepository,
        plugin_runtime: Any | None = None,
    ) -> None:
        self.engine = engine
        self.repository = repository
        self.storage = AssetStorageService(data_root, engine)
        self.renders = RenderRequestRepository(engine)
        self.settings = SettingsResolver(engine)
        self.plugin_runtime = plugin_runtime

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
                )
                .join(bubbles, bubbles.c.page_id == pages.c.id)
                .join(
                    page_assets,
                    (page_assets.c.page_id == pages.c.id)
                    & (page_assets.c.role == "source"),
                )
                .join(assets, assets.c.id == page_assets.c.asset_id)
                .where(
                    pages.c.id == page_id,
                    bubbles.c.id == bubble_id,
                )
            ).mappings().one_or_none()
        if row is None:
            raise ValueError("page or bubble not found")
        payload = json.loads(row["payload_json"])
        width, height = int(row["width"]), int(row["height"])
        method = str(payload.get("inpaintMethod", "solid"))
        fill_color = payload.get("fillColor")
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        polygon = payload.get("polygon")
        if isinstance(polygon, list) and len(polygon) >= 3:
            draw.polygon(
                [
                    (int(point[0]), int(point[1]))
                    for point in polygon
                    if isinstance(point, (list, tuple)) and len(point) >= 2
                ],
                fill=255,
            )
        else:
            coords = payload.get("coords", [0, 0, 0, 0])
            draw.rectangle(tuple(int(value) for value in coords[:4]), fill=255)
        mask_payload = self._mask_payload(mask)
        mask.close()
        mask_checksum = hashlib.sha256(mask_payload).hexdigest()
        replay = self.repository.find_page_repair_replay(
            page_id=page_id,
            base_revision=base_revision,
            method=method,
            fill_color=str(fill_color) if fill_color is not None else None,
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
        mask_asset = self._publish_mask_payload(
            mask_payload,
            width=width,
            height=height,
        )
        return self.repository.create_page_repair(
            page_id=page_id,
            base_revision=base_revision,
            method=method,
            fill_color=str(fill_color) if fill_color is not None else None,
            disable_resize=bool(repair_settings["disableResize"]),
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
        raw = upload.read(64 * 1024 * 1024 + 1)
        if not raw or len(raw) > 64 * 1024 * 1024:
            raise ValueError("repair mask is empty or exceeds 64 MiB")
        with Image.open(BytesIO(raw)) as opened:
            if (
                opened.format != "PNG"
                or opened.mode != "L"
                or getattr(opened, "n_frames", 1) != 1
            ):
                raise ValueError(
                    "repair mask must be a single-frame 8-bit grayscale PNG"
                )
            mask = opened.copy()
        colors = mask.getcolors(maxcolors=3)
        if (
            colors is None
            or any(value not in {0, 255} for _count, value in colors)
            or not any(value == 255 for _count, value in colors)
        ):
            mask.close()
            raise ValueError(
                "repair mask must be binary (0=keep, 255=repair) and non-empty"
            )
        with self.engine.connect() as connection:
            dimensions = connection.execute(
                select(assets.c.width, assets.c.height)
                .join(page_assets, page_assets.c.asset_id == assets.c.id)
                .where(
                    page_assets.c.page_id == page_id,
                    page_assets.c.role == "source",
                )
            ).one_or_none()
        if dimensions is None or mask.size != tuple(dimensions):
            mask.close()
            raise ValueError("repair mask dimensions must match the source image")
        mask_payload = self._mask_payload(mask)
        mask_checksum = hashlib.sha256(mask_payload).hexdigest()
        mask.close()
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
        repair_settings = (
            self.settings.resolve_page_repair(page_id=page_id)
            if method in {"lama_mpe", "litelama"}
            else {"disableResize": False, "settingsSnapshot": {}}
        )
        mask_asset = self._publish_mask_payload(
            mask_payload,
            width=int(dimensions[0]),
            height=int(dimensions[1]),
        )
        return self.repository.create_page_repair(
            page_id=page_id,
            base_revision=base_revision,
            method=method,
            fill_color=fill_color,
            disable_resize=bool(repair_settings["disableResize"]),
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
        method = str(request.get("method", ""))
        inputs = operation.get("inputs")
        if not isinstance(inputs, Mapping):
            raise RuntimeError("repair operation inputs are missing")
        page_id = str(operation["pageId"])
        with self.engine.connect() as connection:
            bubble_payloads = [
                json.loads(value)
                for value in connection.execute(
                    select(bubbles.c.payload_json)
                    .where(bubbles.c.page_id == page_id)
                    .order_by(bubbles.c.ordinal)
                ).scalars()
            ]
        source_asset_id = str(inputs["source"])
        input_asset_id = str(
            inputs.get("parent_clean") or inputs["source"]
        )
        before = self._atomic_hook(
            fence,
            phase="before",
            page_id=page_id,
            data={
                "pageId": page_id,
                "sourceAssetId": source_asset_id,
                "inputAssetId": input_asset_id,
                "textMaskAssetId": str(inputs["repair_mask"]),
                "bubbles": bubble_payloads,
                "method": method,
                "fillColor": request.get("fillColor"),
            },
        )
        method = str(before["method"])
        source = self._open_asset(str(before["sourceAssetId"]), "RGB")
        parent = (
            self._open_asset(str(before["inputAssetId"]), "RGB")
        )
        mask = self._open_asset(str(before["textMaskAssetId"]), "L")
        try:
            if method == "solid":
                repaired = parent.copy()
                fill = Image.new(
                    "RGB",
                    repaired.size,
                    str(before.get("fillColor") or "#FFFFFF"),
                )
                repaired.paste(fill, mask=mask)
                fill.close()
            elif method == "restore_source":
                repaired = parent.copy()
                repaired.paste(source, mask=mask)
            elif method in {"lama_mpe", "litelama"}:
                import numpy as np
                from src.core.inpainting import inpaint_bubbles

                repaired, clean_background = inpaint_bubbles(
                    parent,
                    [(0, 0, parent.width, parent.height)],
                    method="lama",
                    fill_color=str(before.get("fillColor") or "#FFFFFF"),
                    user_mask=np.array(mask),
                    lama_model=method,
                    disable_resize=bool(request["disableResize"]),
                )
                warning = None
                if not bool(getattr(repaired, "_lama_inpainted", False)):
                    warning = {
                        "code": "lama_fallback_to_solid",
                        "message": "LaMA failed; frozen fillColor was applied",
                    }
                if clean_background is not None:
                    clean_background.close()
            else:
                raise RuntimeError(f"unsupported repair method: {method}")
            record = self._publish_png(repaired)
            repaired.close()
        finally:
            source.close()
            parent.close()
            mask.close()

        revision = int(operation["baseRevision"])
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
        record = self._asset_record(str(after["cleanAssetId"]))

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
        if method in {"lama_mpe", "litelama"} and warning is not None:
            result["warning"] = warning
        self.repository.complete(fence, result=result, publisher=publish)
        return {**result, "__already_published__": True}

    @staticmethod
    def _mask_payload(mask: Image.Image) -> bytes:
        output = BytesIO()
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
        output = BytesIO()
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
