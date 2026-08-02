"""Prepare immutable translated assets for a fenced render request."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image
from sqlalchemy import Engine, insert, select, update
from sqlalchemy.engine import Connection

from src.backend_v2.serialization import canonical_json
from src.backend_v2.operations.repository import RenderFence
from src.backend_v2.rendering.fonts import materialize_render_payloads
from src.backend_v2.storage.assets import AssetRecord, AssetStorageService
from src.backend_v2.storage.schema import (
    assets,
    bubbles,
    page_assets,
    pages,
)


def publish_png_asset(
    storage: AssetStorageService,
    image: Image.Image,
    *,
    mode: str | None = None,
) -> AssetRecord:
    converted = image if mode is None or image.mode == mode else image.convert(mode)
    try:
        with BytesIO() as output:
            converted.save(output, format="PNG")
            payload = output.getvalue()
    finally:
        if converted is not image:
            converted.close()
    return storage.publish_bytes(
        payload,
        extension="png",
        mime_type="image/png",
        width=image.width,
        height=image.height,
    )


class AuthoritativeRenderService:
    def __init__(self, *, data_root: Path, engine: Engine) -> None:
        self.engine = engine
        self.storage = AssetStorageService(data_root, engine)

    def prepare(self, fence: RenderFence):
        from src.core.config_models import BubbleState
        from src.core.rendering import render_bubbles_unified

        with self.engine.connect() as connection:
            page = connection.execute(
                select(
                    pages.c.source_revision,
                    pages.c.document_revision,
                    pages.c.default_font_id,
                    pages.c.page_style_defaults_json,
                ).where(pages.c.id == fence.page_id)
            ).mappings().one_or_none()
            if page is None or page["document_revision"] != fence.rendering_revision:
                raise RuntimeError("render target revision is no longer current")
            asset = connection.execute(
                select(assets.c.relative_path)
                .join(page_assets, page_assets.c.asset_id == assets.c.id)
                .where(
                    page_assets.c.page_id == fence.page_id,
                    page_assets.c.role.in_(("clean", "source")),
                )
                .order_by(
                    (page_assets.c.role == "clean").desc()
                )
                .limit(1)
            ).scalar_one_or_none()
            projected = materialize_render_payloads(
                connection,
                self.storage,
                fence.page_id,
            )
            persisted_payloads = [
                (bubble_id, persisted)
                for bubble_id, persisted, _render_payload in projected
            ]
            payloads = [
                render_payload
                for _bubble_id, _persisted, render_payload in projected
            ]
        if asset is None:
            raise RuntimeError("page has no renderable source asset")
        path = self.storage.resolve_relative_path(str(asset))
        with Image.open(path) as opened:
            base = opened.convert("RGB")
        try:
            states = [BubbleState.from_dict(payload) for payload in payloads]
            rendered = base.copy()
            if states:
                render_bubbles_unified(rendered, states)
            translated = publish_png_asset(self.storage, rendered)
        finally:
            base.close()
            if "rendered" in locals():
                rendered.close()

        def publish(connection: Connection) -> None:
            for bubble_id, payload in persisted_payloads:
                connection.execute(
                    update(bubbles)
                    .where(
                        bubbles.c.id == bubble_id,
                        bubbles.c.page_id == fence.page_id,
                        bubbles.c.updated_revision <= fence.rendering_revision,
                    )
                    .values(
                        payload_json=canonical_json(payload),
                        updated_revision=fence.rendering_revision,
                    )
                )
            self._set_translated_pointer(
                connection,
                fence=fence,
                asset=translated,
                source_revision=int(page["source_revision"]),
            )

        return publish

    @staticmethod
    def _set_translated_pointer(
        connection: Connection,
        *,
        fence: RenderFence,
        asset: AssetRecord,
        source_revision: int,
    ) -> None:
        existing = connection.execute(
            select(page_assets.c.asset_id).where(
                page_assets.c.page_id == fence.page_id,
                page_assets.c.role == "translated",
            )
        ).scalar_one_or_none()
        values = {
            "asset_id": asset.id,
            "input_source_revision": source_revision,
            "input_document_revision": fence.rendering_revision,
            "parent_asset_id": None,
            "producer_job_step_id": None,
            "producer_operation_id": None,
            "producer_render_request_id": fence.render_request_id,
        }
        if existing is None:
            connection.execute(
                insert(page_assets).values(
                    page_id=fence.page_id,
                    role="translated",
                    **values,
                )
            )
        else:
            connection.execute(
                update(page_assets)
                .where(
                    page_assets.c.page_id == fence.page_id,
                    page_assets.c.role == "translated",
                )
                .values(**values)
            )
