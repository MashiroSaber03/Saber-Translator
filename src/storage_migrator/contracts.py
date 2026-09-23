"""SQLite contracts shared by startup checks and offline conversion."""

from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import quote

from src.version import parse_version
from .paths import filesystem_path


class StorageError(RuntimeError):
    pass


def read_database(path: Path):
    # Path.as_uri() treats the Windows extended-path '?' as a URI authority.
    path = filesystem_path(path).resolve()
    return sqlite3.connect("file:" + quote(str(path), safe="/:") + "?mode=ro", uri=True)


def read_identity(root: Path) -> tuple[str, str]:
    try:
        with closing(read_database(root / "saber.sqlite3")) as db:
            rows = db.execute("SELECT storage_version, runtime_profile FROM schema_metadata WHERE singleton_id=1").fetchall()
        if len(rows) != 1:
            raise ValueError("missing metadata")
        version, profile = rows[0]
        parse_version(version)
        if profile not in {"local", "public"}:
            raise ValueError("invalid profile")
        return version, profile
    except (sqlite3.Error, ValueError, TypeError) as exc:
        raise StorageError("数据库缺少有效的运行模式信息或三段式存储版本；不支持无版本旧数据，请选择新的数据目录。原数据未删除。") from exc


def _sql_tokens(sql: str) -> tuple[str, ...]:
    # Ignore formatting/case of SQL syntax, never normalize string literals.
    return tuple(token if token.startswith("'") else token.lower() for token in re.findall(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|\w+|[^\w\s]", sql or ""))


def _checks(sql: str) -> list[tuple[str, ...]]:
    tokens = _sql_tokens(sql)
    result = []
    for index, token in enumerate(tokens):
        if token != "check" or tokens[index + 1:index + 2] != ("(",):
            continue
        depth = 0
        for end in range(index + 1, len(tokens)):
            depth += (tokens[end] == "(") - (tokens[end] == ")")
            if depth == 0:
                result.append(tokens[index + 2:end])
                break
    return sorted(result)


def schema_signature(db: sqlite3.Connection) -> dict:
    result = {}
    for name, sql in db.execute("SELECT name,sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall():
        quoted = '"' + name.replace('"', '""') + '"'
        columns = [tuple(row) for row in db.execute(f"PRAGMA table_xinfo({quoted})")]
        foreign_keys = sorted(tuple(row[2:]) for row in db.execute(f"PRAGMA foreign_key_list({quoted})"))
        indexes = []
        for _, idx, unique, origin, partial in db.execute(f"PRAGMA index_list({quoted})").fetchall():
            iq = '"' + idx.replace('"', '""') + '"'
            keys = tuple(tuple(row[1:]) for row in db.execute(f"PRAGMA index_xinfo({iq})"))
            expression = db.execute("SELECT sql FROM sqlite_master WHERE type='index' AND name=?", (idx,)).fetchone()
            indexes.append((idx if origin == "c" else origin, unique, partial, keys, _sql_tokens(expression[0]) if expression and expression[0] else ()))
        result[name] = (columns, foreign_keys, sorted(indexes, key=repr), _checks(sql), "autoincrement" in _sql_tokens(sql))
    return result


def contract_sql(version: str) -> str:
    parse_version(version)
    path = Path(__file__).parent / "schemas" / f"{version}.sql"
    if not path.is_file():
        raise StorageError(f"没有存储版本 {version} 的格式定义")
    return path.read_text(encoding="utf-8")


def validate_schema(db: sqlite3.Connection, sql: str) -> None:
    if db.execute("PRAGMA integrity_check").fetchone() != ("ok",):
        raise StorageError("SQLite integrity_check failed")
    if db.execute("PRAGMA foreign_key_check").fetchall():
        raise StorageError("SQLite foreign_key_check failed")
    extras = db.execute("SELECT type,name FROM sqlite_master WHERE type IN ('trigger','view')").fetchall()
    with closing(sqlite3.connect(":memory:")) as expected:
        expected.executescript(sql)
        wanted = schema_signature(expected)
        expected_extras = expected.execute("SELECT type,name FROM sqlite_master WHERE type IN ('trigger','view')").fetchall()
    if extras != expected_extras:
        raise StorageError("存储结构不匹配: unexpected triggers/views")
    actual = schema_signature(db)
    missing, unexpected = set(wanted) - set(actual), set(actual) - set(wanted)
    changed = sorted(name for name in set(actual) & set(wanted) if actual[name] != wanted[name])
    if missing or unexpected or changed:
        raise StorageError(f"存储结构不匹配: missing={sorted(missing)}, unexpected={sorted(unexpected)}, changed={changed}")


def file_hash(path: Path) -> str:
    with filesystem_path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def validate_data(root: Path, sql: str, *, files: bool = False) -> None:
    with closing(read_database(root / "saber.sqlite3")) as db:
        db.execute("BEGIN")
        validate_schema(db, sql)
        for table in schema_signature(db):
            quoted = '"' + table.replace('"', '""') + '"'
            columns = [row[1] for row in db.execute(f"PRAGMA table_info({quoted})") if row[1].endswith("_json") or row[1] == "runtime_log"]
            for column in columns:
                for (value,) in db.execute(f'SELECT "{column}" FROM {quoted} WHERE "{column}" IS NOT NULL'):
                    try:
                        if not isinstance(json.loads(value), (dict, list)):
                            raise ValueError("expected object or array")
                    except (ValueError, TypeError) as exc:
                        raise StorageError(f"无效 JSON: {table}.{column}") from exc
        if files and "assets" in schema_signature(db):
            for relative, checksum, size in db.execute("SELECT relative_path,checksum,byte_size FROM assets WHERE integrity_status='ok'"):
                path = root / relative
                if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink() or not path.is_file():
                    raise StorageError(f"资源文件缺失或路径无效: {relative}")
                if path.stat().st_size != size or file_hash(path) != checksum:
                    raise StorageError(f"资源文件内容校验失败: {relative}")
