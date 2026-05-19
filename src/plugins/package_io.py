from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Set, Tuple

from src.shared.exceptions import PluginException


PACKAGE_MANIFEST_NAME = "manifest.json"
PACKAGE_VERSION = 1
IGNORED_DIR_NAMES = {"__pycache__", ".git", ".svn", ".hg", ".idea", ".vscode"}
IGNORED_FILE_SUFFIXES = {".pyc", ".pyo", ".pyd"}


@dataclass(frozen=True)
class PluginPackageFile:
    path: str
    sha256: str
    size: int


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize_rel_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip("/")
    if not normalized:
        raise PluginException("插件包内存在空路径")
    if normalized.startswith("../") or "/../" in normalized or normalized == "..":
        raise PluginException("插件包包含非法路径")
    return normalized


def _is_ignored_path(rel_path: str) -> bool:
    normalized = rel_path.replace("\\", "/").strip("/")
    if not normalized:
        return True
    parts = normalized.split("/")
    if any(part in IGNORED_DIR_NAMES for part in parts):
        return True
    lower_name = parts[-1].lower()
    if lower_name == ".ds_store":
        return True
    suffix = os.path.splitext(lower_name)[1]
    return suffix in IGNORED_FILE_SUFFIXES


def _iter_plugin_source_files(source_path: str) -> List[Tuple[str, str]]:
    source_root = os.path.abspath(source_path)
    collected: List[Tuple[str, str]] = []
    for root, dirs, files in os.walk(source_root):
        dirs[:] = [
            d for d in sorted(dirs)
            if d not in IGNORED_DIR_NAMES
        ]
        for filename in sorted(files):
            abs_path = os.path.join(root, filename)
            rel_path = os.path.relpath(abs_path, source_root).replace("\\", "/")
            if rel_path == PACKAGE_MANIFEST_NAME or _is_ignored_path(rel_path):
                continue
            collected.append((rel_path, abs_path))
    collected.sort(key=lambda item: item[0])
    return collected


def build_plugin_package(
    source_path: str,
    *,
    manifest_fields: Dict[str, Any],
) -> Tuple[bytes, Dict[str, Any]]:
    source_root = os.path.abspath(source_path)
    if not os.path.isdir(source_root):
        raise PluginException("插件源目录不存在")

    source_directory = os.path.basename(os.path.normpath(source_root))
    files: List[PluginPackageFile] = []
    collected = _iter_plugin_source_files(source_root)

    buffer = io.BytesIO()
    manifest: Dict[str, Any] = {
        "package_version": PACKAGE_VERSION,
        "source_directory": source_directory,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        **manifest_fields,
    }

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for rel_path, abs_path in collected:
            with open(abs_path, "rb") as handle:
                raw = handle.read()
            archive.writestr(f"{source_directory}/{rel_path}", raw)
            files.append(
                PluginPackageFile(
                    path=f"{source_directory}/{rel_path}",
                    sha256=_sha256_bytes(raw),
                    size=len(raw),
                )
            )

        manifest["files"] = [file.__dict__ for file in files]
        archive.writestr(
            PACKAGE_MANIFEST_NAME,
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        )

    return buffer.getvalue(), manifest


def load_plugin_package_manifest(package_bytes: bytes) -> Dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(package_bytes), "r") as archive:
        if PACKAGE_MANIFEST_NAME not in archive.namelist():
            raise PluginException("插件包缺少 manifest.json")
        try:
            manifest = json.loads(archive.read(PACKAGE_MANIFEST_NAME).decode("utf-8"))
        except Exception as exc:
            raise PluginException("插件包 manifest.json 无法解析") from exc
    if not isinstance(manifest, dict):
        raise PluginException("插件包 manifest 必须是对象")
    package_version = manifest.get("package_version")
    if package_version != PACKAGE_VERSION:
        raise PluginException(
            "不支持的插件包版本",
            details={"package_version": package_version, "supported_version": PACKAGE_VERSION},
        )
    return manifest


def _safe_join(base_dir: str, rel_path: str) -> str:
    normalized_rel = _normalize_rel_path(rel_path)
    base_abs = os.path.abspath(base_dir)
    target_abs = os.path.abspath(os.path.join(base_abs, normalized_rel))
    if target_abs != base_abs and not target_abs.startswith(base_abs + os.sep):
        raise PluginException("插件包包含非法路径")
    return target_abs


def extract_plugin_package(package_bytes: bytes, destination_dir: str) -> Dict[str, Any]:
    os.makedirs(destination_dir, exist_ok=True)
    manifest = load_plugin_package_manifest(package_bytes)
    source_directory = str(manifest.get("source_directory") or "").strip()
    if not source_directory or "/" in source_directory or "\\" in source_directory:
        raise PluginException("插件包 manifest 的 source_directory 非法")

    expected_paths: Set[str] = set()
    files = manifest.get("files") or []
    if not isinstance(files, list):
        raise PluginException("插件包 manifest.files 必须是数组")
    for record in files:
        if not isinstance(record, dict):
            raise PluginException("插件包文件清单格式错误")
        rel_path = str(record.get("path") or "").strip()
        if not rel_path:
            raise PluginException("插件包文件清单缺少 path")
        expected_paths.add(_normalize_rel_path(rel_path))

    with zipfile.ZipFile(io.BytesIO(package_bytes), "r") as archive:
        actual_paths = {
            _normalize_rel_path(member.filename)
            for member in archive.infolist()
            if not member.filename.endswith("/") and member.filename != PACKAGE_MANIFEST_NAME
        }
        if actual_paths != expected_paths:
            unexpected = sorted(actual_paths - expected_paths)
            missing = sorted(expected_paths - actual_paths)
            raise PluginException(
                "插件包文件清单与压缩包内容不一致",
                details={
                    "unexpected_files": unexpected,
                    "missing_files": missing,
                },
            )

        for member in archive.infolist():
            if member.filename.endswith("/"):
                continue
            target_path = _safe_join(destination_dir, member.filename)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with archive.open(member, "r") as source, open(target_path, "wb") as target:
                target.write(source.read())

    return manifest


def validate_plugin_package_contents(destination_dir: str, manifest: Dict[str, Any]) -> None:
    source_directory = str(manifest.get("source_directory") or "").strip()
    if not source_directory:
        raise PluginException("插件包 manifest 缺少 source_directory")

    files = manifest.get("files") or []
    if not isinstance(files, list):
        raise PluginException("插件包 manifest.files 必须是数组")

    for record in files:
        if not isinstance(record, dict):
            raise PluginException("插件包文件清单格式错误")
        rel_path = str(record.get("path") or "").strip()
        if not rel_path:
            raise PluginException("插件包文件清单缺少 path")
        abs_path = _safe_join(destination_dir, rel_path)
        if not os.path.isfile(abs_path):
            raise PluginException(f"插件包缺少文件: {rel_path}")
        expected_sha256 = str(record.get("sha256") or "").strip()
        if expected_sha256:
            with open(abs_path, "rb") as handle:
                actual_sha256 = _sha256_bytes(handle.read())
            if actual_sha256 != expected_sha256:
                raise PluginException(f"插件包文件校验失败: {rel_path}")
