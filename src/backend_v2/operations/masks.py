"""Page mask previews and non-rendering manual-mask persistence."""

from contextlib import ExitStack
from io import BytesIO
import base64
import json

import numpy as np
from PIL import Image, UnidentifiedImageError
from sqlalchemy import select

from src.backend_v2.auth.ownership import effective_owner_id
from src.backend_v2.content.repository import ContentRepository
from src.backend_v2.operations.repository import OperationNotFound
from src.backend_v2.settings.resolver import SettingsResolver
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.database import read_transaction
from src.backend_v2.storage.schema import assets, books, bubbles, chapters, page_assets, pages
from src.core.bubble_geometry import rotated_box_polygon
from src.core.repair_mask import apply_user_mask, build_automatic_mask_layers


def png_bytes(mask: Image.Image) -> bytes:
    with BytesIO() as stream:
        mask.save(stream, format="PNG")
        return stream.getvalue()


def read_user_mask(upload) -> Image.Image:
    try:
        with Image.open(upload) as image:
            if image.format != "PNG" or image.mode != "L" or getattr(image, "n_frames", 1) != 1:
                raise ValueError("掩膜必须是单帧、8位灰度 PNG")
            colors = image.getcolors(maxcolors=4)
            if colors is None or any(value not in {0, 127, 255} for _, value in colors):
                raise ValueError("掩膜像素值无效")
            return image.copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("掩膜必须是单帧、8位灰度 PNG") from exc


class PageMaskService:
    def __init__(self, *, data_root, engine, settings_transformer=None):
        self.engine = engine
        self.storage = AssetStorageService(data_root, engine)
        self.content = ContentRepository(engine)
        self.settings = SettingsResolver(engine)
        self.settings_transformer = settings_transformer

    def snapshot(self, page_id):
        with read_transaction(self.engine) as connection:
            page = connection.execute(select(
                pages.c.document_revision, pages.c.source_revision,
                assets.c.width, assets.c.height,
            ).select_from(pages).join(chapters, chapters.c.id == pages.c.chapter_id)
                .join(books, books.c.id == chapters.c.book_id)
                .join(page_assets, (page_assets.c.page_id == pages.c.id) & (page_assets.c.role == "source"))
                .join(assets, assets.c.id == page_assets.c.asset_id)
                .where(pages.c.id == page_id, books.c.owner_user_id == effective_owner_id())
            ).mappings().one_or_none()
            if page is None:
                raise OperationNotFound("page not found")
            asset_ids = dict(connection.execute(select(page_assets.c.role, page_assets.c.asset_id)
                .where(page_assets.c.page_id == page_id)).all())
            payloads = [json.loads(value) for value in connection.execute(select(bubbles.c.payload_json)
                .where(bubbles.c.page_id == page_id).order_by(bubbles.c.ordinal)).scalars()]
        return dict(page), asset_ids, payloads

    def open_mask(self, asset_id, size):
        if asset_id is None:
            return Image.new("L", size, 127)
        record = self.storage.get_record(asset_id)
        if record is None:
            raise ValueError("掩膜资产不存在")
        with Image.open(self.storage.resolve_relative_path(record.relative_path)) as image:
            if image.size != size:
                raise ValueError("掩膜尺寸与原图不一致")
            return image.convert("L")

    def preview(self, *, page_id, method):
        page, asset_ids, payloads = self.snapshot(page_id)
        size = (int(page["width"]), int(page["height"]))
        settings = self.settings.resolve_page_repair(page_id=page_id)
        if self.settings_transformer:
            settings = self.settings_transformer(settings)
        with ExitStack() as opened:
            precise = (opened.enter_context(self.open_mask(asset_ids["text_mask"], size))
                       if "text_mask" in asset_ids else None)
            user = opened.enter_context(self.open_mask(asset_ids.get("user_mask"), size))
            coords = [value["coords"] for value in payloads]
            polygons = [rotated_box_polygon(value["coords"], value["rotationAngle"]) for value in payloads]
            precise_array = np.array(precise) if precise is not None else None
            box, automatic = build_automatic_mask_layers(
                (size[1], size[0]), coords, polygons,
                precise_mask=precise_array,
                mask_dilate_size=settings["maskDilateSize"],
                mask_box_expand_ratio=settings["maskBoxExpandRatio"],
                method="solid" if method == "solid" else "lama",
                regional_inpainting=settings["regionalInpainting"],
            )
            manual = np.array(user)
            final = apply_user_mask(automatic, manual)
            def encoded(array):
                with Image.fromarray(array) as image:
                    return base64.b64encode(png_bytes(image)).decode("ascii")
            return {
                "pageId": page_id, "sourceRevision": page["source_revision"], "documentRevision": page["document_revision"],
                "width": size[0], "height": size[1],
                "preciseMask": encoded(precise_array) if precise_array is not None else None,
                "boxMask": encoded(box),
                "automaticMask": encoded(automatic), "userMask": encoded(manual),
                "finalMask": encoded(final),
                "processing": {
                    "boxExpandRatio": settings["maskBoxExpandRatio"] if precise is not None else 0,
                    "dilateSize": settings["maskDilateSize"] if precise is not None else 0,
                    "closeGaps": precise is not None and method != "solid" and settings["regionalInpainting"],
                },
            }

    def save(self, *, page_id, upload, base_revision, idempotency_key):
        page, _, _ = self.snapshot(page_id)
        with read_user_mask(upload) as mask:
            if mask.size != (page["width"], page["height"]):
                raise ValueError("掩膜尺寸与原图不一致")
            record = self.publish(mask)
        return self.content.mutate_page_document(
            page_id=page_id, base_revision=base_revision, mutations=[],
            idempotency_key=idempotency_key, user_mask_asset_id=record.id,
        )

    def publish(self, mask):
        return self.storage.publish_stream(
            BytesIO(png_bytes(mask)), extension="png", mime_type="image/png",
            width=mask.width, height=mask.height,
        )
