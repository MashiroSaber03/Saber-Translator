"""Prepare immutable translated assets for a fenced render request."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

from PIL import Image
from sqlalchemy import Engine, insert, select, update
from sqlalchemy.engine import Connection

from src.backend_v2.operations.repository import RenderFence
from src.backend_v2.storage.assets import AssetRecord, AssetStorageService
from src.backend_v2.storage.schema import (
    assets,
    bubbles,
    page_assets,
    pages,
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
                select(pages.c.source_revision, pages.c.document_revision).where(
                    pages.c.id == fence.page_id
                )
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
            payloads = [
                json.loads(value)
                for value in connection.execute(
                    select(bubbles.c.payload_json)
                    .where(bubbles.c.page_id == fence.page_id)
                    .order_by(bubbles.c.ordinal)
                ).scalars()
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
            translated = self._publish_png(rendered)
            thumbnail = self._publish_thumbnail(rendered)
        finally:
            base.close()
            if "rendered" in locals():
                rendered.close()

        def publish(connection: Connection) -> None:
            self._set_pointer(
                connection,
                fence=fence,
                role="translated",
                asset=translated,
                source_revision=int(page["source_revision"]),
                parent_asset_id=None,
            )
            self._set_pointer(
                connection,
                fence=fence,
                role="thumbnail_translated",
                asset=thumbnail,
                source_revision=int(page["source_revision"]),
                parent_asset_id=translated.id,
            )

        return publish

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
    def _set_pointer(
        connection: Connection,
        *,
        fence: RenderFence,
        role: str,
        asset: AssetRecord,
        source_revision: int,
        parent_asset_id: str | None,
    ) -> None:
        existing = connection.execute(
            select(page_assets.c.asset_id).where(
                page_assets.c.page_id == fence.page_id,
                page_assets.c.role == role,
            )
        ).scalar_one_or_none()
        values = {
            "asset_id": asset.id,
            "input_source_revision": source_revision,
            "input_document_revision": fence.rendering_revision,
            "parent_asset_id": parent_asset_id,
            "producer_job_step_id": None,
            "producer_operation_id": None,
            "producer_render_request_id": fence.render_request_id,
        }
        if existing is None:
            connection.execute(
                insert(page_assets).values(
                    page_id=fence.page_id,
                    role=role,
                    **values,
                )
            )
        else:
            connection.execute(
                update(page_assets)
                .where(
                    page_assets.c.page_id == fence.page_id,
                    page_assets.c.role == role,
                )
                .values(**values)
            )
