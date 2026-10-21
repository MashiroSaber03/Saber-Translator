"""Prepare immutable translated assets for a fenced render request."""

from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from pathlib import Path

from PIL import Image
from sqlalchemy import Engine, select
from sqlalchemy.engine import Connection

from src.backend_v2.operations.repository import RenderFence
from src.backend_v2.editing.native import NativeEditor
from src.backend_v2.storage.assets import AssetRecord, AssetStorageService
from src.backend_v2.storage.database import read_transaction
from src.backend_v2.storage.schema import pages


def publish_png_asset(
    storage: AssetStorageService,
    image: Image.Image,
    *,
    mode: str | None = None,
    bind: Callable[[Connection, str], None] | None = None,
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
        bind=bind,
    )


class AuthoritativeRenderService:
    def __init__(self, *, data_root: Path, engine: Engine) -> None:
        self.engine = engine
        self.storage = AssetStorageService(data_root, engine)

    def prepare(self, fence: RenderFence):
        with read_transaction(self.engine) as connection:
            page = connection.execute(
                select(
                    pages.c.source_revision,
                    pages.c.document_revision,
                ).where(pages.c.id == fence.page_id)
            ).mappings().one_or_none()
            if page is None:
                raise RuntimeError("render target page no longer exists")
            if page["document_revision"] != fence.rendering_revision:
                # The render repository owns revision coalescing.  Let its
                # atomic completion check requeue the newest revision without
                # rendering or reporting an expected superseded attempt.
                return lambda _connection: None
            native = NativeEditor(self.storage)
            result = native.prepare(connection, fence.page_id, render=True)

        def publish(connection: Connection) -> None:
            native.publish(connection, fence.page_id, fence.rendering_revision, result, render_request_id=fence.render_request_id)

        return publish
