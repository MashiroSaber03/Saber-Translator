"""Immutable asset lookup for streamed Flask media responses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import Engine, select, update

from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.schema import assets
from src.backend_v2.timestamps import utcnow


@dataclass(frozen=True, slots=True)
class MediaAsset:
    path: Path
    mime_type: str
    checksum: str
    created_at: object


class AssetMediaService:
    def __init__(
        self,
        *,
        engine: Engine,
        storage: AssetStorageService,
    ) -> None:
        self.engine = engine
        self.storage = storage

    def locate(self, asset_id: str) -> MediaAsset | None:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    assets.c.relative_path,
                    assets.c.mime_type,
                    assets.c.checksum,
                    assets.c.created_at,
                ).where(
                    assets.c.id == asset_id,
                    assets.c.integrity_status == "ok",
                    assets.c.gc_marked_at.is_(None),
                )
            ).mappings().one_or_none()
        if row is None:
            return None
        path = self.storage.resolve_relative_path(str(row["relative_path"]))
        if not path.is_file():
            with self.engine.begin() as connection:
                connection.execute(
                    update(assets)
                    .where(
                        assets.c.id == asset_id,
                        assets.c.integrity_status == "ok",
                    )
                    .values(integrity_status="missing", updated_at=utcnow())
                )
            return None
        return MediaAsset(
            path=path,
            mime_type=str(row["mime_type"]),
            checksum=str(row["checksum"]),
            created_at=row["created_at"],
        )
