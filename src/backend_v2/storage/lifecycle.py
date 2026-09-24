"""Initialize fresh storage atomically; accept only the current storage contract."""

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import json

from sqlalchemy import insert

from src.version import STORAGE_VERSION
from src.storage_migrator.contracts import (
    StorageError, contract_sql, read_database, read_identity, validate_schema,
)
from src.storage_migrator.control import control_root, is_empty_root, reject_links, atomic_json
from src.backend_v2.storage.database import create_sqlite_engine, database_path_for
from src.backend_v2.runtime_profile import PROFILE_NAMES
from src.backend_v2.storage.schema import metadata, schema_metadata
from src.backend_v2.storage.seeding import seed_system_records


@dataclass(frozen=True, slots=True)
class StorageInitializationResult:
    database_path: Path
    created: bool


def schema_smoke_test(database_path: Path) -> None:
    with closing(read_database(database_path)) as connection:
        connection.execute("BEGIN")
        validate_schema(connection, contract_sql(STORAGE_VERSION))


def _clear_initialization(data_root: Path) -> None:
    control = control_root(data_root)
    staging = control / "initialization"
    reject_links(staging)
    if not staging.exists():
        return
    marker = staging / "owner.json"
    for path in staging.rglob("*"):
        reject_links(path)
    if any(staging.iterdir()) and (not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != {"root": str(data_root.resolve())}):
        raise StorageError("无法核验初始化临时目录，拒绝清理")
    if not staging.resolve().is_relative_to(control.resolve()):
        raise StorageError("初始化清理路径越界")
    for path in staging.iterdir():
        if path != marker:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    marker.unlink(missing_ok=True)
    staging.rmdir()


def initialize_database(data_root: Path, *, profile_name: str = "local") -> StorageInitializationResult:
    if profile_name not in PROFILE_NAMES:
        raise ValueError(f"unsupported runtime profile: {profile_name!r}")
    database_path = database_path_for(data_root)
    if database_path.exists():
        version, profile = read_identity(data_root)
        if profile != profile_name:
            raise StorageError(f"该数据目录属于 {profile} 模式，不能由 {profile_name} 模式使用")
        if version != STORAGE_VERSION:
            raise StorageError(f"存储版本 {version} 与所需存储版本 {STORAGE_VERSION} 不一致，请先运行存储转换器")
        schema_smoke_test(database_path)
        _clear_initialization(data_root)
        return StorageInitializationResult(database_path, False)
    if not is_empty_root(data_root):
        raise StorageError("数据库缺失但目录仍有旧数据；请选择新的数据目录")
    control = control_root(data_root)
    control.mkdir(parents=True, exist_ok=True)
    # All seed writes happen outside the published root. Even a killed process
    # cannot leave a half-created production database.
    staging = control / "initialization"
    _clear_initialization(data_root)
    marker = staging / "owner.json"
    staging.mkdir()
    atomic_json(marker, {"root": str(data_root.resolve())})
    candidate = database_path_for(staging)
    engine = create_sqlite_engine(candidate)
    try:
        metadata.create_all(engine)
        seed_system_records(engine, profile_name=profile_name)
        with engine.begin() as connection:
            connection.execute(insert(schema_metadata).values(
                singleton_id=1, runtime_profile=profile_name, storage_version=STORAGE_VERSION,
            ))
        with engine.connect() as connection:
            connection.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE)")
        engine.dispose()
        schema_smoke_test(candidate)
        data_root.mkdir(parents=True, exist_ok=True)
        os.replace(candidate, database_path)
    finally:
        engine.dispose()
        _clear_initialization(data_root)
    return StorageInitializationResult(database_path, True)
