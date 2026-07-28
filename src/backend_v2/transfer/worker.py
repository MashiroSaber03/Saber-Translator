"""Worker handlers for incremental container import and frozen-asset export."""

from __future__ import annotations

from datetime import timedelta
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
from typing import Any, Mapping
import uuid
import zipfile

from PIL import Image
from sqlalchemy import Engine, insert, select, update
from sqlalchemy.engine import Connection

from src.backend_v2.content.image_import import ImageImportService
from src.backend_v2.content.repository import (
    ContentRepository,
    _deduplicate_logical_path,
    normalize_logical_path,
)
from src.backend_v2.jobs.repository import (
    AttemptFence,
    JobQueueRepository,
    utcnow,
)
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.schema import (
    assets,
    app_settings,
    chapter_write_locks,
    chapters,
    job_artifacts,
    job_asset_inputs,
    job_items,
    job_steps,
    jobs,
    fonts,
    page_assets,
    pages,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


def _natural_path_key(value: object) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(value))
    )


class TransferWorkerService:
    def __init__(
        self,
        *,
        data_root: Path,
        engine: Engine,
        jobs_repository: JobQueueRepository,
    ) -> None:
        self.data_root = data_root.resolve()
        self.engine = engine
        self.jobs = jobs_repository
        self.storage = AssetStorageService(data_root, engine)
        self.importer = ImageImportService(
            data_root=data_root,
            repository=ContentRepository(engine),
            storage=self.storage,
        )

    def handler(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        kind = str(step["stepKind"])
        if kind == "container_scan":
            return self._scan_container(fence, step)
        if kind == "container_import_page":
            return self._import_container_page(fence, step)
        if kind == "container_cleanup":
            return self._cleanup_container(fence, step)
        if kind == "export_package":
            return self._export(fence, step)
        raise ValueError(f"unsupported transfer step: {kind}")

    def _scan_container(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        config = self._config(step)
        container = self._data_path(str(config["containerRelativePath"]))
        container_type = str(config["containerType"])
        if container_type in {"zip", "cbz"}:
            entries = self._scan_zip(container)
        elif container_type == "pdf":
            import fitz

            with fitz.open(container) as document:
                entries = [
                    {
                        "kind": "pdf",
                        "pageIndex": index,
                        "logicalPath": f"page_{index + 1:05d}.png",
                    }
                    for index in range(document.page_count)
                ]
        elif container_type in {"mobi", "azw", "azw3"}:
            entries = self._scan_mobi(container, fence.job_id)
        else:
            raise ValueError("unsupported container type")
        if not entries:
            raise ValueError("container contains no supported images")
        new_config = {**config, "entries": entries}
        if container_type in {"mobi", "azw", "azw3"}:
            new_config["extractedRelativePath"] = (
                Path("temp") / "container-import" / fence.job_id
            ).as_posix()
        now = utcnow()

        def publish(connection: Connection) -> None:
            next_ordinal = 2
            for _entry in entries:
                item_id = str(uuid.uuid4())
                connection.execute(
                    insert(job_items).values(
                        id=item_id,
                        job_id=fence.job_id,
                        ordinal=next_ordinal,
                        status="pending",
                        created_at=now,
                        updated_at=now,
                    )
                )
                connection.execute(
                    insert(job_steps).values(
                        id=str(uuid.uuid4()),
                        job_item_id=item_id,
                        ordinal=1,
                        kind="container_import_page",
                        status="pending",
                        checkpoint_schema_version=1,
                        created_at=now,
                        updated_at=now,
                    )
                )
                next_ordinal += 1
            cleanup_item_id = str(uuid.uuid4())
            connection.execute(
                insert(job_items).values(
                    id=cleanup_item_id,
                    job_id=fence.job_id,
                    ordinal=next_ordinal,
                    status="pending",
                    created_at=now,
                    updated_at=now,
                )
            )
            connection.execute(
                insert(job_steps).values(
                    id=str(uuid.uuid4()),
                    job_item_id=cleanup_item_id,
                    ordinal=1,
                    kind="container_cleanup",
                    status="pending",
                    checkpoint_schema_version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            connection.execute(
                update(jobs)
                .where(
                    jobs.c.id == fence.job_id,
                    jobs.c.attempt_id == fence.attempt_id,
                )
                .values(
                    config_json=json.dumps(
                        new_config,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    updated_at=now,
                )
            )

        checkpoint = {"entryCount": len(entries), "scanned": True}
        self.jobs.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint=checkpoint,
            publisher=publish,
        )
        return {**checkpoint, "__already_published__": True}

    def _import_container_page(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        config = self._config(step)
        entries = config.get("entries")
        if not isinstance(entries, list):
            raise RuntimeError("container scan checkpoint is missing")
        entry_index = int(step["itemOrdinal"]) - 2
        if entry_index < 0 or entry_index >= len(entries):
            raise RuntimeError("container item ordinal is invalid")
        entry = entries[entry_index]
        if not isinstance(entry, dict):
            raise RuntimeError("container entry is invalid")
        raw = self._read_entry(config, entry)
        source, thumbnail = self.importer.publish_replacement(BytesIO(raw))
        page_id = str(uuid.uuid4())
        logical_path = normalize_logical_path(str(entry["logicalPath"]))
        now = utcnow()
        chapter_id = str(
            self._job_target(fence.job_id)["chapter_id"]
        )

        def publish(connection: Connection) -> None:
            if connection.execute(
                select(chapter_write_locks.c.job_id).where(
                    chapter_write_locks.c.chapter_id == chapter_id,
                    chapter_write_locks.c.job_id == fence.job_id,
                    chapter_write_locks.c.owner_attempt_id == fence.attempt_id,
                )
            ).scalar_one_or_none() is None:
                raise RuntimeError("container import lost its chapter write lock")
            existing_paths = set(
                connection.execute(
                    select(pages.c.logical_source_path).where(
                        pages.c.chapter_id == chapter_id
                    )
                ).scalars()
            )
            final_path = _deduplicate_logical_path(
                logical_path,
                existing_paths,
            )
            ordinal = int(
                connection.execute(
                    select(pages.c.ordinal)
                    .where(pages.c.chapter_id == chapter_id)
                    .order_by(pages.c.ordinal.desc())
                    .limit(1)
                ).scalar_one_or_none()
                or 0
            ) + 1
            connection.execute(
                insert(pages).values(
                    id=page_id,
                    chapter_id=chapter_id,
                    ordinal=ordinal,
                    logical_source_path=final_path,
                    default_font_id=connection.execute(
                        select(fonts.c.id)
                        .where(fonts.c.kind == "builtin")
                        .order_by(fonts.c.created_at)
                        .limit(1)
                    ).scalar_one_or_none(),
                    page_style_defaults_json=(
                        connection.execute(
                            select(app_settings.c.payload_json).where(
                                app_settings.c.domain == "text_style_defaults"
                            )
                        ).scalar_one_or_none()
                        or "{}"
                    ),
                    created_at=now,
                    updated_at=now,
                )
            )
            connection.execute(
                insert(page_assets),
                [
                    {
                        "page_id": page_id,
                        "role": "source",
                        "asset_id": source.id,
                        "input_source_revision": 1,
                        "parent_asset_id": None,
                    },
                    {
                        "page_id": page_id,
                        "role": "thumbnail_source",
                        "asset_id": thumbnail.id,
                        "input_source_revision": 1,
                        "parent_asset_id": source.id,
                    },
                ],
            )
            connection.execute(
                update(job_items)
                .where(
                    job_items.c.id == step["itemId"],
                    job_items.c.job_id == fence.job_id,
                )
                .values(page_id=page_id, updated_at=now)
            )
            connection.execute(
                update(chapters)
                .where(chapters.c.id == chapter_id)
                .values(
                    page_order_revision=chapters.c.page_order_revision + 1,
                    updated_at=now,
                )
            )

        checkpoint = {
            "pageId": page_id,
            "logicalPath": logical_path,
            "sourceAssetId": source.id,
            "thumbnailAssetId": thumbnail.id,
        }
        self.jobs.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint=checkpoint,
            input_fingerprint=source.checksum,
            publisher=publish,
        )
        return {**checkpoint, "__already_published__": True}

    def _cleanup_container(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        config = self._config(step)
        checkpoint = {"cleaned": True}
        self.jobs.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint=checkpoint,
        )
        self._data_path(str(config["containerRelativePath"])).unlink(
            missing_ok=True
        )
        extracted = config.get("extractedRelativePath")
        if isinstance(extracted, str):
            directory = self._data_path(extracted)
            if directory.is_dir():
                shutil.rmtree(directory)
        return {**checkpoint, "__already_published__": True}

    def _export(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        config = self._config(step)
        entries = config.get("entries")
        export_format = str(config.get("format", ""))
        if not isinstance(entries, list) or export_format not in {"zip", "cbz", "pdf"}:
            raise RuntimeError("export snapshot is invalid")
        self._assert_export_bindings(fence.job_id, entries)
        output_dir = self.data_root / "temp" / "exports"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{fence.job_id}.{export_format}"
        failures: list[dict[str, object]] = []
        successful = 0
        if export_format in {"zip", "cbz"}:
            used_names: set[str] = set()
            with zipfile.ZipFile(
                output_path,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                allowZip64=True,
            ) as archive:
                for entry in entries:
                    try:
                        path = self._asset_path(str(entry["assetId"]))
                        member = self._export_member_name(entry, used_names)
                        archive.write(path, member)
                        successful += 1
                    except Exception as exc:
                        failures.append(
                            {"pageId": entry.get("pageId"), "message": str(exc)}
                        )
        else:
            import img2pdf

            paths: list[str] = []
            for entry in entries:
                try:
                    path = self._asset_path(str(entry["assetId"]))
                    with Image.open(path) as image:
                        image.verify()
                    paths.append(str(path))
                    successful += 1
                except Exception as exc:
                    failures.append(
                        {"pageId": entry.get("pageId"), "message": str(exc)}
                    )
            if paths:
                with output_path.open("wb") as output:
                    img2pdf.convert(*paths, outputstream=output)
        if successful == 0:
            output_path.unlink(missing_ok=True)
            raise RuntimeError("export could not read any frozen page asset")
        mime = (
            "application/pdf"
            if export_format == "pdf"
            else (
                "application/vnd.comicbook+zip"
                if export_format == "cbz"
                else "application/zip"
            )
        )
        with output_path.open("rb") as source:
            artifact = self.storage.publish_stream(
                source,
                extension=export_format,
                mime_type=mime,
            )
        output_path.unlink(missing_ok=True)
        expires = utcnow() + timedelta(hours=24)

        def publish(connection: Connection) -> None:
            connection.execute(
                insert(job_artifacts).values(
                    job_id=fence.job_id,
                    kind=export_format,
                    asset_id=artifact.id,
                    expires_at=expires,
                )
            )

        checkpoint = {
            "artifactAssetId": artifact.id,
            "artifactUrl": f"/api/v2/assets/{artifact.id}",
            "format": export_format,
            "successfulPages": successful,
            "failedPages": failures,
            "expiresAt": expires.isoformat(),
        }
        self.jobs.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint=checkpoint,
            publisher=publish,
        )
        return {**checkpoint, "__already_published__": True}

    def _scan_zip(self, path: Path) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > 10_000:
                raise ValueError("archive contains too many entries")
            total_size = 0
            for info in infos:
                pure = PurePosixPath(info.filename.replace("\\", "/"))
                if (
                    pure.is_absolute()
                    or ".." in pure.parts
                    or stat.S_ISLNK(info.external_attr >> 16)
                ):
                    raise ValueError("archive contains an unsafe path or symlink")
                total_size += info.file_size
                if total_size > 4 * 1024 * 1024 * 1024:
                    raise ValueError("archive expands beyond 4 GiB")
                if (
                    info.compress_size > 0
                    and info.file_size / info.compress_size > 1000
                ):
                    raise ValueError("archive entry compression ratio is unsafe")
                if (
                    not info.is_dir()
                    and pure.suffix.lower() in IMAGE_SUFFIXES
                ):
                    entries.append(
                        {
                            "kind": "zip",
                            "member": info.filename,
                            "logicalPath": pure.as_posix(),
                            "byteSize": info.file_size,
                        }
                    )
        entries.sort(key=lambda entry: _natural_path_key(entry["logicalPath"]))
        return entries

    def _scan_mobi(
        self,
        path: Path,
        job_id: str,
    ) -> list[dict[str, object]]:
        import mobi

        temporary_root, extracted_path = mobi.extract(str(path))
        destination = self.data_root / "temp" / "container-import" / job_id
        destination.mkdir(parents=True, exist_ok=True)
        entries: list[dict[str, object]] = []
        try:
            root = Path(extracted_path)
            for source in sorted(root.rglob("*")):
                if not source.is_file() or source.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                relative = source.relative_to(root)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                entries.append(
                    {
                        "kind": "file",
                        "relativePath": (
                            Path("temp")
                            / "container-import"
                            / job_id
                            / relative
                        ).as_posix(),
                        "logicalPath": relative.as_posix(),
                    }
                )
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)
        return entries

    def _read_entry(
        self,
        config: Mapping[str, Any],
        entry: Mapping[str, Any],
    ) -> bytes:
        kind = str(entry["kind"])
        container = self._data_path(str(config["containerRelativePath"]))
        if kind == "zip":
            with zipfile.ZipFile(container) as archive:
                info = archive.getinfo(str(entry["member"]))
                if info.file_size > 128 * 1024 * 1024:
                    raise ValueError("image exceeds the single-file byte limit")
                with archive.open(info) as source:
                    raw = source.read(128 * 1024 * 1024 + 1)
        elif kind == "pdf":
            import fitz

            with fitz.open(container) as document:
                page = document.load_page(int(entry["pageIndex"]))
                raw = page.get_pixmap(alpha=False).tobytes("png")
        elif kind == "file":
            raw = self._data_path(str(entry["relativePath"])).read_bytes()
        else:
            raise RuntimeError("unknown container entry kind")
        if not raw or len(raw) > 128 * 1024 * 1024:
            raise ValueError("image is empty or exceeds the single-file byte limit")
        return raw

    def _assert_export_bindings(
        self,
        job_id: str,
        entries: list[object],
    ) -> None:
        with self.engine.connect() as connection:
            bound = set(
                connection.execute(
                    select(job_asset_inputs.c.asset_id).where(
                        job_asset_inputs.c.job_id == job_id,
                        job_asset_inputs.c.binding_phase == "create",
                    )
                ).scalars()
            )
        requested = {
            str(entry["assetId"])
            for entry in entries
            if isinstance(entry, dict)
        }
        if bound != requested:
            raise RuntimeError("export asset bindings do not match the frozen snapshot")

    def _job_target(self, job_id: str):
        with self.engine.connect() as connection:
            row = connection.execute(
                select(jobs.c.chapter_id).where(jobs.c.id == job_id)
            ).mappings().one()
        return row

    @staticmethod
    def _config(step: Mapping[str, Any]) -> dict[str, Any]:
        config = step.get("config")
        if not isinstance(config, dict):
            raise RuntimeError("job configuration is invalid")
        return dict(config)

    def _data_path(self, relative: str) -> Path:
        return self.storage.resolve_relative_path(relative)

    def _asset_path(self, asset_id: str) -> Path:
        with self.engine.connect() as connection:
            relative = connection.execute(
                select(assets.c.relative_path).where(assets.c.id == asset_id)
            ).scalar_one()
        return self._data_path(str(relative))

    @staticmethod
    def _export_member_name(
        entry: Mapping[str, Any],
        used: set[str],
    ) -> str:
        logical = PurePosixPath(str(entry["logicalPath"]))
        prefix = (
            "translated"
            if entry.get("assetRole") == "translated"
            else "original"
        )
        extension = Path(str(entry["relativePath"])).suffix.lower() or ".png"
        parent = "" if str(logical.parent) == "." else f"{logical.parent.as_posix()}/"
        base = f"{parent}{prefix}_{logical.stem}{extension}"
        candidate = base
        counter = 2
        while candidate.lower() in used:
            candidate = (
                f"{parent}{prefix}_{logical.stem} ({counter}){extension}"
            )
            counter += 1
        used.add(candidate.lower())
        return candidate
