"""Worker handlers for incremental container import and frozen-asset export."""

from __future__ import annotations

from datetime import timedelta
from io import BytesIO
from pathlib import Path, PurePosixPath
import shutil
import stat
from typing import Any, Mapping
import uuid
import zipfile

from PIL import Image
from sqlalchemy import Engine, insert, select, update
from sqlalchemy.engine import Connection

from src.backend_v2.checksums import sha256_file
from src.backend_v2.serialization import canonical_json
from src.backend_v2.content.image_import import (
    ImageImportService,
    ImportSafetyLimits,
)
from src.backend_v2.content.page_style import (
    PAGE_STYLE_SCHEMA_VERSION,
    resolve_new_page_style,
)
from src.backend_v2.content.repository import (
    ContentRepository,
    deduplicate_logical_path,
    natural_sort_key,
    normalize_logical_path,
)
from src.backend_v2.insight.repository import (
    mark_book_insight_derived_stale,
)
from src.backend_v2.jobs.repository import (
    AttemptFence,
    JobQueueRepository,
)
from src.backend_v2.timestamps import utcnow
from src.backend_v2.storage.assets import AssetRecord, AssetStorageService
from src.backend_v2.storage.schema import (
    assets,
    chapter_write_locks,
    chapters,
    job_artifacts,
    job_asset_inputs,
    job_items,
    job_steps,
    jobs,
    page_assets,
    pages,
)
from src.backend_v2.transfer.commands import (
    validate_container_config,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


class TransferWorkerService:
    def __init__(
        self,
        *,
        data_root: Path,
        engine: Engine,
        jobs_repository: JobQueueRepository,
        limits: ImportSafetyLimits = ImportSafetyLimits(),
    ) -> None:
        self.data_root = data_root.resolve()
        self.engine = engine
        self.jobs = jobs_repository
        self.limits = limits
        self.storage = AssetStorageService(data_root, engine)
        self.importer = ImageImportService(
            data_root=data_root,
            repository=ContentRepository(engine),
            storage=self.storage,
            limits=limits,
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
        if sha256_file(container) != config.get("checksum"):
            raise ValueError("frozen container checksum changed")
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
        chapter_id = str(self._job_target(fence.job_id)["chapter_id"])
        with self.engine.connect() as connection:
            used_paths = set(
                connection.execute(
                    select(pages.c.logical_source_path).where(
                        pages.c.chapter_id == chapter_id
                    )
                ).scalars()
            )
        frozen_entries: list[dict[str, object]] = []
        for raw_entry in entries:
            logical_path = normalize_logical_path(
                str(raw_entry["logicalPath"])
            )
            final_path = deduplicate_logical_path(logical_path, used_paths)
            used_paths.add(final_path)
            frozen_entries.append({**raw_entry, "logicalPath": final_path})
        entries = frozen_entries
        new_config = {
            **config,
            "entries": entries,
            "entryItemOrdinalBase": 2,
        }
        if container_type in {"mobi", "azw", "azw3"}:
            new_config["extractedRelativePath"] = (
                Path("temp") / "container-import" / fence.job_id
            ).as_posix()
        validate_container_config(new_config)
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
            connection.execute(
                update(jobs)
                .where(
                    jobs.c.id == fence.job_id,
                    jobs.c.attempt_id == fence.attempt_id,
                )
                .values(
                    config_json=canonical_json(new_config),
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
        entry_index = int(step["itemOrdinal"]) - int(
            config.get("entryItemOrdinalBase", 2)
        )
        if entry_index < 0 or entry_index >= len(entries):
            raise RuntimeError("container item ordinal is invalid")
        entry = entries[entry_index]
        if not isinstance(entry, dict):
            raise RuntimeError("container entry is invalid")
        source, thumbnail = self._publish_entry(config, entry)
        page_id = str(uuid.uuid4())
        logical_path = normalize_logical_path(str(entry["logicalPath"]))
        now = utcnow()
        target = self._job_target(fence.job_id)
        chapter_id = str(target["chapter_id"])
        book_id = str(target["book_id"])

        def publish(connection: Connection) -> None:
            if connection.execute(
                select(chapter_write_locks.c.job_id).where(
                    chapter_write_locks.c.chapter_id == chapter_id,
                    chapter_write_locks.c.job_id == fence.job_id,
                    chapter_write_locks.c.owner_attempt_id == fence.attempt_id,
                )
            ).scalar_one_or_none() is None:
                raise RuntimeError("container import lost its chapter write lock")
            ordinal = int(
                connection.execute(
                    select(pages.c.ordinal)
                    .where(pages.c.chapter_id == chapter_id)
                    .order_by(pages.c.ordinal.desc())
                    .limit(1)
                ).scalar_one_or_none()
                or 0
            ) + 1
            default_font_id, style_defaults = resolve_new_page_style(connection)
            connection.execute(
                insert(pages).values(
                    id=page_id,
                    chapter_id=chapter_id,
                    ordinal=ordinal,
                    logical_source_path=logical_path,
                    default_font_id=default_font_id,
                    page_style_defaults_json=canonical_json(style_defaults),
                    page_style_schema_version=PAGE_STYLE_SCHEMA_VERSION,
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
            mark_book_insight_derived_stale(
                connection,
                book_id=book_id,
                now=now,
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
        asset_paths = self._bound_export_paths(fence.job_id, entries)
        output_dir = self.data_root / "temp" / "exports"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{fence.job_id}.{export_format}"
        successful = 0
        try:
            if export_format in {"zip", "cbz"}:
                used_names: set[str] = set()
                with zipfile.ZipFile(
                    output_path,
                    "w",
                    compression=zipfile.ZIP_DEFLATED,
                    allowZip64=True,
                ) as archive:
                    for entry in entries:
                        path = asset_paths[str(entry["assetId"])]
                        member = self._export_member_name(
                            entry,
                            used_names,
                            extension=path.suffix.lower() or ".png",
                        )
                        archive.write(path, member)
                        successful += 1
            else:
                import img2pdf

                paths: list[str] = []
                for entry in entries:
                    path = asset_paths[str(entry["assetId"])]
                    with Image.open(path) as image:
                        image.verify()
                    paths.append(str(path))
                    successful += 1
                with output_path.open("wb") as output:
                    img2pdf.convert(*paths, outputstream=output)
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
        finally:
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
            seen_members: set[str] = set()
            for info in infos:
                pure = PurePosixPath(info.filename.replace("\\", "/"))
                member_key = pure.as_posix()
                if member_key in seen_members:
                    raise ValueError("archive contains duplicate member names")
                seen_members.add(member_key)
                mode = info.external_attr >> 16
                file_type = stat.S_IFMT(mode)
                if (
                    pure.is_absolute()
                    or ".." in pure.parts
                    or stat.S_ISLNK(mode)
                    or file_type
                    not in {0, stat.S_IFREG, stat.S_IFDIR}
                ):
                    raise ValueError(
                        "archive contains an unsafe path or special file"
                    )
                if (
                    info.file_size > 0
                    and (
                        info.compress_size == 0
                        or info.file_size / info.compress_size
                        > self.limits.max_compression_ratio
                    )
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
        entries.sort(key=lambda entry: natural_sort_key(entry["logicalPath"]))
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
            resolved_root = root.resolve()
            for source in sorted(root.rglob("*")):
                if source.is_symlink():
                    raise ValueError("MOBI extraction contains a symbolic link")
                if not source.is_file() or source.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                resolved_source = source.resolve()
                try:
                    resolved_source.relative_to(resolved_root)
                except ValueError as exc:
                    raise ValueError(
                        "MOBI extraction escaped its temporary directory"
                    ) from exc
                relative = source.relative_to(root)
                byte_size = source.stat().st_size
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
                        "byteSize": byte_size,
                    }
                )
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)
        entries.sort(key=lambda entry: natural_sort_key(entry["logicalPath"]))
        return entries

    def _publish_entry(
        self,
        config: Mapping[str, Any],
        entry: Mapping[str, Any],
    ) -> tuple[AssetRecord, AssetRecord]:
        kind = str(entry["kind"])
        container = self._data_path(str(config["containerRelativePath"]))
        if kind == "zip":
            with zipfile.ZipFile(container) as archive:
                with archive.open(str(entry["member"])) as source:
                    return self.importer.publish_standalone_image(source)
        if kind == "pdf":
            import fitz

            with fitz.open(container) as document:
                page = document.load_page(int(entry["pageIndex"]))
                raw = page.get_pixmap(alpha=False).tobytes("png")
            return self.importer.publish_standalone_image(BytesIO(raw))
        if kind == "file":
            path = self._data_path(str(entry["relativePath"]))
            with path.open("rb") as source:
                return self.importer.publish_standalone_image(source)
        raise RuntimeError("unknown container entry kind")

    def _bound_export_paths(
        self,
        job_id: str,
        entries: list[object],
    ) -> dict[str, Path]:
        with self.engine.connect() as connection:
            rows = list(
                connection.execute(
                    select(
                        job_asset_inputs.c.asset_id,
                        assets.c.relative_path,
                    )
                    .join(
                        assets,
                        assets.c.id == job_asset_inputs.c.asset_id,
                    )
                    .where(
                        job_asset_inputs.c.job_id == job_id,
                        job_asset_inputs.c.binding_phase == "create",
                    )
                ).mappings()
            )
        bound = {
            str(row["asset_id"]): self._data_path(str(row["relative_path"]))
            for row in rows
        }
        requested = {
            str(entry["assetId"])
            for entry in entries
            if isinstance(entry, dict)
        }
        if set(bound) != requested:
            raise RuntimeError("export asset bindings do not match the frozen snapshot")
        return bound

    def _job_target(self, job_id: str):
        with self.engine.connect() as connection:
            row = connection.execute(
                select(jobs.c.book_id, jobs.c.chapter_id).where(
                    jobs.c.id == job_id
                )
            ).mappings().one()
        return row

    @staticmethod
    def _config(step: Mapping[str, Any]) -> dict[str, Any]:
        config = step.get("config")
        kind = step.get("stepKind")
        if kind not in {
            "container_scan",
            "container_import_page",
            "export_package",
        }:
            raise RuntimeError("transfer step kind is invalid")
        if not isinstance(config, Mapping):
            raise RuntimeError("transfer job configuration is invalid")
        # Commands validate the complete frozen snapshot when it is created and
        # container_scan validates the expanded snapshot before publishing it.
        # Rewalking every entry for every page would make large imports O(n²).
        return dict(config)

    def _data_path(self, relative: str) -> Path:
        return self.storage.resolve_relative_path(relative)

    @staticmethod
    def _export_member_name(
        entry: Mapping[str, Any],
        used: set[str],
        *,
        extension: str,
    ) -> str:
        logical = PurePosixPath(str(entry["logicalPath"]))
        role = str(entry.get("assetRole") or "")
        prefix = (
            "translated"
            if role == "translated"
            else "clean"
            if role == "clean"
            else "original"
        )
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
