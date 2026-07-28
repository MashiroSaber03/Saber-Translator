"""Safe immutable package I/O for plugin v3."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

from src.backend_v2.plugins.contract import (
    PluginContractError,
    PluginManifest,
    parse_manifest,
)


MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_EXPANDED_BYTES = 128 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 2_000
MANIFEST_NAME = "plugin.json"


@dataclass(frozen=True, slots=True)
class ParsedPluginArchive:
    manifest: PluginManifest
    archive_checksum: str
    members: tuple[tuple[str, bytes], ...]


def parse_archive(data: bytes) -> ParsedPluginArchive:
    if not data or len(data) > MAX_ARCHIVE_BYTES:
        raise PluginContractError("plugin archive byte size is invalid")
    try:
        archive = zipfile.ZipFile(BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise PluginContractError("plugin archive is not a valid ZIP") from exc
    with archive:
        infos = archive.infolist()
        if not infos or len(infos) > MAX_ARCHIVE_ENTRIES:
            raise PluginContractError("plugin archive entry count is invalid")
        members: list[tuple[str, bytes]] = []
        total = 0
        for info in infos:
            path = _safe_member_path(info)
            if info.is_dir():
                continue
            total += int(info.file_size)
            if total > MAX_EXPANDED_BYTES:
                raise PluginContractError(
                    "plugin archive expanded size is too large"
                )
            members.append((path.as_posix(), archive.read(info)))
    by_name = {name: content for name, content in members}
    raw_manifest = by_name.get(MANIFEST_NAME)
    if raw_manifest is None:
        raise PluginContractError(f"plugin archive is missing {MANIFEST_NAME}")
    try:
        decoded = json.loads(raw_manifest.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PluginContractError("plugin manifest is invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise PluginContractError("plugin manifest must be an object")
    manifest = parse_manifest(decoded)
    module_path = manifest.entrypoint.rsplit(":", 1)[0].replace("\\", "/")
    if module_path not in by_name:
        raise PluginContractError("plugin entrypoint module is missing")
    return ParsedPluginArchive(
        manifest=manifest,
        archive_checksum=hashlib.sha256(data).hexdigest(),
        members=tuple(sorted(members)),
    )


def extract_archive(
    archive: ParsedPluginArchive,
    destination: Path,
) -> str:
    destination.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256()
    for name, content in archive.members:
        target = destination.joinpath(*PurePosixPath(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def directory_checksum(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        value for value in root.rglob("*") if value.is_file()
    ):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def build_archive(root: Path) -> bytes:
    if not root.is_dir():
        raise FileNotFoundError("plugin version directory is missing")
    output = BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in sorted(
            value for value in root.rglob("*") if value.is_file()
        ):
            info = zipfile.ZipInfo(path.relative_to(root).as_posix())
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return output.getvalue()


def _safe_member_path(info: zipfile.ZipInfo) -> PurePosixPath:
    raw = info.filename.replace("\\", "/")
    path = PurePosixPath(raw)
    mode = info.external_attr >> 16
    if (
        not raw
        or path.is_absolute()
        or ".." in path.parts
        or any(not part or part == "." for part in path.parts)
        or stat.S_ISLNK(mode)
    ):
        raise PluginContractError("plugin archive contains an unsafe path")
    return path
