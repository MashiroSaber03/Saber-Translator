"""Read-only consistency report for v2 structured and file storage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from sqlalchemy import Engine, select

from src.backend_v2.insight.derived import InsightVectorStore
from src.backend_v2.storage.schema import assets, object_commit_journal


@dataclass(frozen=True, slots=True)
class ConsistencyReport:
    foreign_key_violations: tuple[tuple[object, ...], ...]
    invalid_asset_paths: tuple[str, ...]
    missing_asset_files: tuple[str, ...]
    integrity_status_mismatches: tuple[str, ...]
    orphan_object_files: tuple[str, ...]
    missing_vector_collections: tuple[str, ...]
    orphan_vector_collections: tuple[str, ...]
    vector_check_error: str | None

    @property
    def ok(self) -> bool:
        return not any(
            (
                self.foreign_key_violations,
                self.invalid_asset_paths,
                self.missing_asset_files,
                self.integrity_status_mismatches,
                self.orphan_object_files,
                self.missing_vector_collections,
                self.orphan_vector_collections,
                self.vector_check_error,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, **asdict(self)}


class ConsistencyChecker:
    def __init__(
        self,
        *,
        data_root: Path,
        engine: Engine,
    ) -> None:
        self.data_root = data_root.resolve()
        self.objects_root = self.data_root / "objects"
        self.engine = engine

    def check(self, *, include_vectors: bool = True) -> ConsistencyReport:
        invalid_paths: list[str] = []
        missing_files: list[str] = []
        status_mismatches: list[str] = []
        with self.engine.connect() as connection:
            foreign_key_violations = tuple(
                tuple(row)
                for row in connection.exec_driver_sql(
                    "PRAGMA foreign_key_check"
                ).all()
            )
            asset_rows = list(
                connection.execute(
                    select(
                        assets.c.id,
                        assets.c.relative_path,
                        assets.c.integrity_status,
                    )
                ).mappings()
            )
            journal_paths = {
                str(value)
                for value in connection.execute(
                    select(object_commit_journal.c.final_relative_path)
                ).scalars()
            }

        recorded_paths: set[str] = set()
        for row in asset_rows:
            asset_id = str(row["id"])
            relative_path = str(row["relative_path"])
            recorded_paths.add(relative_path)
            resolved = self._resolve(relative_path)
            if resolved is None:
                invalid_paths.append(asset_id)
                continue
            exists = resolved.is_file()
            if not exists:
                missing_files.append(asset_id)
            expected_status = "ok" if exists else "missing"
            if str(row["integrity_status"]) != expected_status:
                status_mismatches.append(asset_id)

        orphan_files: list[str] = []
        if self.objects_root.is_dir():
            for path in self.objects_root.rglob("*"):
                if not path.is_file():
                    continue
                relative_path = path.relative_to(self.data_root).as_posix()
                if (
                    relative_path not in recorded_paths
                    and relative_path not in journal_paths
                ):
                    orphan_files.append(relative_path)

        missing_vectors: tuple[str, ...] = ()
        orphan_vectors: tuple[str, ...] = ()
        vector_error: str | None = None
        if include_vectors:
            try:
                inspection = InsightVectorStore(
                    self.data_root
                ).inspect_collections(self.engine)
            except Exception as exc:
                vector_error = f"{type(exc).__name__}: {exc}"
            else:
                missing_vectors = inspection.missing
                orphan_vectors = inspection.orphaned

        return ConsistencyReport(
            foreign_key_violations=foreign_key_violations,
            invalid_asset_paths=tuple(sorted(invalid_paths)),
            missing_asset_files=tuple(sorted(missing_files)),
            integrity_status_mismatches=tuple(sorted(status_mismatches)),
            orphan_object_files=tuple(sorted(orphan_files)),
            missing_vector_collections=missing_vectors,
            orphan_vector_collections=orphan_vectors,
            vector_check_error=vector_error,
        )

    def _resolve(self, relative_path: str) -> Path | None:
        pure = PurePosixPath(relative_path)
        if pure.is_absolute() or not pure.parts or ".." in pure.parts:
            return None
        resolved = (self.data_root / Path(*pure.parts)).resolve()
        try:
            resolved.relative_to(self.data_root)
        except ValueError:
            return None
        return resolved
