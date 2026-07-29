"""Immutable object storage with crash journal, integrity scan, and two-pass GC."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
from typing import BinaryIO
import uuid

from sqlalchemy import Engine, delete, exists, func, insert, or_, select, update

from src.backend_v2.storage.schema import assets, metadata, object_commit_journal


_EXTENSION_PATTERN = re.compile(r"^[a-z0-9]{1,12}$")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass(frozen=True, slots=True)
class AssetRecord:
    id: str
    relative_path: str
    mime_type: str
    checksum: str
    byte_size: int
    width: int | None
    height: int | None


@dataclass(frozen=True, slots=True)
class IntegrityScanResult:
    checked: int
    missing: int
    restored: int


@dataclass(frozen=True, slots=True)
class GarbageCollectionResult:
    marked: int
    deleted_rows: int
    deleted_files: int


@dataclass(frozen=True, slots=True)
class OrphanObjectReconciliationResult:
    scanned: int
    deleted: int
    protected: int
    grace_retained: int


class AssetStorageService:
    def __init__(self, data_root: Path, engine: Engine) -> None:
        self.data_root = data_root.resolve()
        self.objects_root = self.data_root / "objects"
        self.staging_root = self.data_root / "temp" / "staging"
        self.engine = engine
        self.objects_root.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)

    def publish_bytes(
        self,
        payload: bytes,
        *,
        extension: str,
        mime_type: str,
        width: int | None = None,
        height: int | None = None,
        bind: Callable[[object, str], None] | None = None,
        failpoint: Callable[[str], None] | None = None,
    ) -> AssetRecord:
        from io import BytesIO

        return self.publish_stream(
            BytesIO(payload),
            extension=extension,
            mime_type=mime_type,
            width=width,
            height=height,
            bind=bind,
            failpoint=failpoint,
        )

    def publish_stream(
        self,
        source: BinaryIO,
        *,
        extension: str,
        mime_type: str,
        width: int | None = None,
        height: int | None = None,
        bind: Callable[[object, str], None] | None = None,
        failpoint: Callable[[str], None] | None = None,
    ) -> AssetRecord:
        canonical_extension = extension.lower().lstrip(".")
        if not _EXTENSION_PATTERN.fullmatch(canonical_extension):
            raise ValueError("asset extension must contain only 1-12 lowercase letters/digits")
        if not mime_type or "/" not in mime_type:
            raise ValueError("a concrete MIME type is required")
        if width is not None and width < 1 or height is not None and height < 1:
            raise ValueError("asset dimensions must be positive")

        asset_id = str(uuid.uuid4())
        relative_path = f"objects/{asset_id[:2]}/{asset_id}.{canonical_extension}"
        staging_relative_path = f"temp/staging/{asset_id}.part"
        staging_path = self.resolve_relative_path(staging_relative_path)
        final_path = self.resolve_relative_path(relative_path)
        staging_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.parent.mkdir(parents=True, exist_ok=True)

        digest = hashlib.sha256()
        byte_size = 0
        with staging_path.open("xb") as output:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                byte_size += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        self._hit(failpoint, "staging_fsynced")

        record = AssetRecord(
            id=asset_id,
            relative_path=relative_path,
            mime_type=mime_type,
            checksum=digest.hexdigest(),
            byte_size=byte_size,
            width=width,
            height=height,
        )
        with self.engine.begin() as connection:
            connection.execute(
                insert(object_commit_journal).values(
                    asset_id=asset_id,
                    staging_relative_path=staging_relative_path,
                    final_relative_path=relative_path,
                    state="staged",
                )
            )
        self._hit(failpoint, "journal_staged")

        os.replace(staging_path, final_path)
        self._fsync_directory(final_path.parent)
        self._hit(failpoint, "file_published")

        with self.engine.begin() as connection:
            connection.execute(
                update(object_commit_journal)
                .where(
                    object_commit_journal.c.asset_id == asset_id,
                    object_commit_journal.c.state == "staged",
                )
                .values(state="file_published")
            )
        self._hit(failpoint, "journal_file_published")

        with self.engine.begin() as connection:
            connection.execute(
                insert(assets).values(
                    id=record.id,
                    relative_path=record.relative_path,
                    mime_type=record.mime_type,
                    checksum=record.checksum,
                    byte_size=record.byte_size,
                    width=record.width,
                    height=record.height,
                )
            )
            if bind is not None:
                bind(connection, asset_id)
            self._hit(failpoint, "database_before_commit")
            connection.execute(
                delete(object_commit_journal).where(
                    object_commit_journal.c.asset_id == asset_id
                )
            )
        self._hit(failpoint, "database_committed")
        return record

    @staticmethod
    def _hit(failpoint: Callable[[str], None] | None, point: str) -> None:
        if failpoint is not None:
            failpoint(point)

    def resolve_relative_path(self, relative_path: str) -> Path:
        pure_path = PurePosixPath(relative_path)
        if pure_path.is_absolute() or ".." in pure_path.parts or not pure_path.parts:
            raise ValueError("stored asset paths must be normalized data-root-relative paths")
        candidate = (self.data_root / Path(*pure_path.parts)).resolve()
        try:
            candidate.relative_to(self.data_root)
        except ValueError as exc:
            raise ValueError("stored path escapes the v2 data root") from exc
        return candidate

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        if os.name == "nt":
            return
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def recover_journal(self, *, orphan_grace_seconds: int = 3600) -> int:
        now = _utcnow()
        recovered = 0
        with self.engine.connect() as connection:
            rows = list(connection.execute(select(object_commit_journal)).mappings())

        for row in rows:
            asset_id = str(row["asset_id"])
            staging_path = self.resolve_relative_path(str(row["staging_relative_path"]))
            final_path = self.resolve_relative_path(str(row["final_relative_path"]))
            created_at = row["created_at"]
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)
            expired = created_at <= now - timedelta(seconds=orphan_grace_seconds)
            with self.engine.connect() as connection:
                database_has_asset = (
                    connection.execute(
                        select(assets.c.id).where(assets.c.id == asset_id)
                    ).scalar_one_or_none()
                    is not None
                )

            if database_has_asset:
                with self.engine.begin() as connection:
                    if not final_path.exists():
                        connection.execute(
                            update(assets)
                            .where(assets.c.id == asset_id)
                            .values(integrity_status="missing", updated_at=now)
                        )
                    connection.execute(
                        delete(object_commit_journal).where(
                            object_commit_journal.c.asset_id == asset_id
                        )
                    )
                staging_path.unlink(missing_ok=True)
                recovered += 1
                continue

            if staging_path.exists() and not final_path.exists():
                final_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging_path, final_path)
                with self.engine.begin() as connection:
                    connection.execute(
                        update(object_commit_journal)
                        .where(object_commit_journal.c.asset_id == asset_id)
                        .values(state="file_published")
                    )
                    if expired:
                        connection.execute(
                            delete(object_commit_journal).where(
                                object_commit_journal.c.asset_id == asset_id
                            )
                        )
                if expired:
                    final_path.unlink(missing_ok=True)
                recovered += 1
                continue

            if expired:
                staging_path.unlink(missing_ok=True)
                final_path.unlink(missing_ok=True)
                with self.engine.begin() as connection:
                    connection.execute(
                        delete(object_commit_journal).where(
                            object_commit_journal.c.asset_id == asset_id
                        )
                    )
                recovered += 1
        referenced_staging = {
            str(row["staging_relative_path"])
            for row in rows
        }
        cutoff_timestamp = (
            now - timedelta(seconds=orphan_grace_seconds)
        ).timestamp()
        for staging_path in self.staging_root.glob("*.part"):
            relative_path = staging_path.relative_to(self.data_root).as_posix()
            if relative_path in referenced_staging:
                continue
            if (
                orphan_grace_seconds == 0
                or staging_path.stat().st_mtime <= cutoff_timestamp
            ):
                staging_path.unlink(missing_ok=True)
                recovered += 1
        return recovered

    def scan_integrity(self) -> IntegrityScanResult:
        checked = missing = restored = 0
        with self.engine.connect() as connection:
            rows = list(
                connection.execute(
                    select(
                        assets.c.id,
                        assets.c.relative_path,
                        assets.c.integrity_status,
                    )
                ).mappings()
            )
        with self.engine.begin() as connection:
            for row in rows:
                checked += 1
                exists = self.resolve_relative_path(str(row["relative_path"])).is_file()
                desired = "ok" if exists else "missing"
                if desired == "missing":
                    missing += 1
                if desired == "ok" and row["integrity_status"] == "missing":
                    restored += 1
                if desired != row["integrity_status"]:
                    connection.execute(
                        update(assets)
                        .where(assets.c.id == row["id"])
                        .values(integrity_status=desired, updated_at=_utcnow())
                    )
        return IntegrityScanResult(checked=checked, missing=missing, restored=restored)

    def reconcile_orphan_objects(
        self,
        *,
        grace_seconds: int = 3600,
        now: datetime | None = None,
    ) -> OrphanObjectReconciliationResult:
        """Remove old object files that have neither a DB row nor a journal.

        Asset-row GC and physical object reconciliation are deliberately
        separate: a crash after ``os.replace`` but before the asset transaction
        can leave a file that never had an ``assets`` row and therefore cannot
        be discovered by SQL-only GC.
        """

        if grace_seconds < 0:
            raise ValueError("grace_seconds must be nonnegative")
        current_time = now or _utcnow()
        cutoff_timestamp = (
            current_time - timedelta(seconds=grace_seconds)
        ).timestamp()
        with self.engine.connect() as connection:
            asset_paths = {
                str(value)
                for value in connection.execute(
                    select(assets.c.relative_path)
                ).scalars()
            }
            journal_paths = {
                str(value)
                for value in connection.execute(
                    select(object_commit_journal.c.final_relative_path)
                ).scalars()
            }
        protected_paths = asset_paths | journal_paths
        scanned = deleted = protected = grace_retained = 0
        for object_path in self.objects_root.rglob("*"):
            if not object_path.is_file():
                continue
            scanned += 1
            relative_path = object_path.relative_to(self.data_root).as_posix()
            if relative_path in protected_paths:
                protected += 1
                continue
            if (
                grace_seconds > 0
                and object_path.stat().st_mtime > cutoff_timestamp
            ):
                grace_retained += 1
                continue
            try:
                object_path.unlink()
            except OSError:
                grace_retained += 1
            else:
                deleted += 1
        return OrphanObjectReconciliationResult(
            scanned=scanned,
            deleted=deleted,
            protected=protected,
            grace_retained=grace_retained,
        )

    def collect_garbage(
        self,
        *,
        grace_seconds: int = 3600,
        now: datetime | None = None,
    ) -> GarbageCollectionResult:
        current_time = now or _utcnow()
        delete_candidates: list[tuple[str, str]] = []
        cutoff = current_time - timedelta(seconds=grace_seconds)
        referenced = self._asset_is_referenced()
        with self.engine.begin() as connection:
            connection.execute(
                update(assets)
                .where(
                    assets.c.gc_marked_at.is_not(None),
                    referenced,
                )
                .values(gc_marked_at=None, updated_at=current_time)
            )
            candidates = list(
                connection.execute(
                    select(
                        assets.c.id,
                        assets.c.relative_path,
                        assets.c.gc_marked_at,
                    ).where(
                        assets.c.gc_marked_at <= cutoff,
                        ~referenced,
                    )
                ).mappings()
            )
            marked = connection.execute(
                update(assets)
                .where(
                    assets.c.gc_marked_at.is_(None),
                    ~referenced,
                )
                .values(gc_marked_at=current_time, updated_at=current_time)
            ).rowcount
            for row in candidates:
                deleted = connection.execute(
                    delete(assets).where(
                        assets.c.id == row["id"],
                        assets.c.gc_marked_at == row["gc_marked_at"],
                        ~referenced,
                    )
                )
                if deleted.rowcount == 1:
                    delete_candidates.append(
                        (str(row["id"]), str(row["relative_path"]))
                    )

        deleted_files = 0
        for _asset_id, relative_path in delete_candidates:
            path = self.resolve_relative_path(relative_path)
            if path.exists():
                path.unlink()
                deleted_files += 1
        return GarbageCollectionResult(
            marked=int(marked or 0),
            deleted_rows=len(delete_candidates),
            deleted_files=deleted_files,
        )

    @staticmethod
    def _asset_is_referenced():
        references = []
        for table in metadata.tables.values():
            if table is assets or table is object_commit_journal:
                continue
            for column in table.columns:
                if any(
                    foreign_key.column.table is assets
                    for foreign_key in column.foreign_keys
                ):
                    references.append(
                        exists(
                            select(1)
                            .select_from(table)
                            .where(column == assets.c.id)
                        )
                    )
        return or_(*references)

    def _referenced_asset_ids(self) -> set[str]:
        referenced: set[str] = set()
        with self.engine.connect() as connection:
            for table in metadata.tables.values():
                if table is assets or table is object_commit_journal:
                    continue
                for column in table.columns:
                    if not any(
                        foreign_key.column.table is assets
                        for foreign_key in column.foreign_keys
                    ):
                        continue
                    referenced.update(
                        value
                        for value in connection.execute(
                            select(column).where(column.is_not(None))
                        ).scalars()
                        if value is not None
                    )
        return referenced
