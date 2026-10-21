"""Prepare the source checkout's pinned editor binaries before starting children."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

from src.backend_v2.paths import project_root


WEB_ROOT = Path("src/backend_v2/static/photocraft")
STAMP = Path("native/editor-runtime.json")
REPOSITORY = "MashiroSaber03/Saber-Translator"
NOTICE_FILES = (
    "LICENSE-MIT", "LICENSE-APACHE", "NOTICE", "ATTRIBUTION.md", "UPSTREAM.json",
    "assets/fonts/OFL-Inter.txt", "assets/fonts/OFL-JetBrainsMono.txt",
    "assets/icons/LICENSE-lucide.txt", "assets/icons/LICENSE-symmetry-axes.txt",
    "assets/icons/LICENSE-noun-magnetic-lasso.txt", "assets/dict/LICENSE-SCOWL.txt",
    "crates/ui-egui/src/i18n/LICENSE-translations.txt",
)


def platform_name() -> str:
    system, machine = platform.system(), platform.machine().lower()
    if system != "Windows" or machine not in {"amd64", "x86_64"}:
        raise RuntimeError("当前预编译编辑内核支持 Windows x64；其他平台请使用 scripts/build_editor.py 构建")
    return "windows-x86_64"


def runtime_files() -> tuple[Path, ...]:
    native = Path("native/saber-editor.exe" if os.name == "nt" else "native/saber-editor")
    return (native, *(WEB_ROOT / name for name in (
        "index.html", "host.js", "photocraft.js", "photocraft_bg.wasm",
        "photocraft.d.ts", "photocraft_bg.wasm.d.ts",
    )), *(Path("native/licenses") / name for name in NOTICE_FILES))


def checksum(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def runtime_stamp(root: Path, build: str) -> dict:
    return {"build": build, "files": {
        path.as_posix(): checksum(root / path) for path in runtime_files()
    }}


def runtime_ready(root: Path, build: str) -> bool:
    try:
        recorded = json.loads((root / STAMP).read_text("utf-8"))
        return isinstance(recorded, dict) and recorded.get("build") == build and recorded == runtime_stamp(root, build)
    except (OSError, ValueError):
        return False


def pinned_build(root: Path) -> str:
    build = json.loads((root / "editor-runtime.json").read_text("utf-8"))["build"]
    if not isinstance(build, str) or len(build) != 24 or any(c not in "0123456789abcdef" for c in build):
        raise ValueError("editor-runtime.json 的 build 必须是 24 位源码指纹")
    return build


def bundle_name(build: str) -> str:
    return f"saber-editor-{build}-{platform_name()}.zip"


def _install(root: Path, staged: Path) -> None:
    """Replace files under the startup lock; commit the stamp last.

    Windows can retain directory handles after a rename. Keep the directories
    in place and back up old files until the complete replacement succeeds.
    """
    backup = staged.parent / "previous"
    installed = []
    try:
        for relative in (*runtime_files(), STAMP):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file():
                (backup / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup / relative)
            (staged / relative).replace(target)
            installed.append(relative)
    except OSError:
        for relative in reversed(installed):
            if (backup / relative).is_file():
                (backup / relative).replace(root / relative)
            else:
                (root / relative).unlink()
        raise


def ensure_editor_runtime(progress=print, *, root: Path | None = None) -> None:
    if getattr(sys, "frozen", False):
        return  # Release packages already contain their matching runtime.
    root = (root or project_root()).resolve()
    build = pinned_build(root)
    if runtime_ready(root, build):
        return
    filename = bundle_name(build)
    url = f"https://github.com/{REPOSITORY}/releases/download/editor-{build}/{filename}"
    # Reuse the project's OS lock; different data directories share these binaries.
    from src.storage_migrator.control import DataRootAlreadyLocked, DataRootLock

    try:
        with DataRootLock(root / "native"):
            if runtime_ready(root, build):
                return
            progress("正在准备编辑内核，首次使用或内核更新时需要下载一次…")
            with TemporaryDirectory(prefix=".tmp-editor-", dir=root) as directory:
                temporary = Path(directory)
                archive_path = temporary / filename
                with urlopen(url + ".sha256", timeout=30) as response:
                    expected = response.read(1024).decode("ascii").split()[0]
                if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
                    raise ValueError("编辑内核校验文件无效")
                with urlopen(url, timeout=30) as response, archive_path.open("wb") as output:
                    total = int(response.headers.get("Content-Length", 0))
                    received, last_percent = 0, -1
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        received += len(chunk)
                        percent = received * 100 // total if total else 0
                        if total and percent // 10 != last_percent // 10:
                            progress(f"正在下载编辑内核：{min(percent, 100)}%")
                            last_percent = percent
                if checksum(archive_path) != expected:
                    raise ValueError("编辑内核下载不完整或校验失败")
                staged = temporary / "unpacked"
                allowed = {path.as_posix() for path in (*runtime_files(), STAMP)}
                with ZipFile(archive_path) as archive:
                    if (set(archive.namelist()) != allowed or len(archive.infolist()) != len(allowed)
                            or sum(item.file_size for item in archive.infolist()) > 512 * 1024 * 1024):
                        raise ValueError("编辑内核包的文件清单不正确")
                    archive.extractall(staged)  # Only the explicit relative paths above are accepted.
                if not runtime_ready(staged, build):
                    raise ValueError("编辑内核包与当前源码不匹配或内容损坏")
                progress("编辑内核校验通过，正在安装…")
                _install(root, staged)
            progress("编辑内核已就绪")
    except DataRootAlreadyLocked as error:
        raise RuntimeError("另一个实例正在准备编辑内核，请稍后重新启动") from error
    except (OSError, ValueError, IndexError, BadZipFile) as error:
        detail = ("当前源码对应的内核包尚未发布，请等待 GitHub Actions 的 Publish Editor Runtime 构建完成后重试"
                  if isinstance(error, HTTPError) and error.code == 404 else str(error))
        raise RuntimeError(
            f"编辑内核准备失败：{detail}\n请检查网络或项目目录写入权限后重启。"
            f"也可手动下载下列 ZIP 和 .sha256 校验文件，校验后将 ZIP 解压到项目根目录：\n{url}"
        ) from error


if __name__ == "__main__":
    try:
        ensure_editor_runtime()
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
