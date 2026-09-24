"""Fixed migration steps between storage versions, independent of app releases."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.version import parse_version
from .contracts import StorageError
from .v3_5_0 import stop_unfinished, validate
from .v3_5_1 import convert as directory_fonts, validate as validate_directory_fonts


@dataclass(frozen=True)
class Migration:
    source: str
    target: str
    convert: Callable[[Path], None] | None = None
    validate: Callable[[Path], None] | None = None


MIGRATIONS: tuple[Migration, ...] = (
    Migration("3.5.0", "3.5.1", directory_fonts),
    Migration("3.5.1", "3.5.2"),
)
SOURCE_PREPARERS = {"3.5.0": stop_unfinished, "3.5.1": stop_unfinished, "3.5.2": stop_unfinished}
VERSION_VALIDATORS = {"3.5.0": validate, "3.5.1": validate_directory_fonts, "3.5.2": validate_directory_fonts}


def migration_chain(source: str, target: str, steps=MIGRATIONS) -> list[Migration]:
    if parse_version(source) > parse_version(target):
        raise StorageError(f"数据版本 {source} 高于目标存储版本 {target}，不支持降级")
    by_source = {}
    for step in steps:
        if step.source in by_source or parse_version(step.source) >= parse_version(step.target):
            raise StorageError("迁移注册表包含重复或非递增的版本")
        by_source[step.source] = step
    result = []
    while source != target:
        step = by_source.get(source)
        if step is None or parse_version(step.target) > parse_version(target):
            raise StorageError(f"缺少从 {source} 到 {target} 的完整迁移路径")
        result.append(step)
        source = step.target
    return result
