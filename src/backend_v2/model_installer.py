"""Explicit, bounded model installation; no inference imports or startup downloads."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Callable
import hashlib
import logging
from pathlib import Path
import re
import shutil
import threading
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
import zipfile

import requests

from src.shared.model_catalog import (
    MODELS_BY_ID, LocalModel, ModelFile, download_sources, download_urls,
    file_matches, installed_root, model_available, models_root,
)
from src.storage_migrator.control import DataRootAlreadyLocked, DataRootLock

LOGGER = logging.getLogger(__name__)


class ModelInstallBusy(RuntimeError):
    pass


class InstallCancelled(Exception):
    pass


def _partial_size(path: Path, expected_size: int) -> int:
    try:
        return min(path.stat().st_size, expected_size)
    except OSError:
        # A completed download can be renamed while the UI reads progress.
        return 0


def _partial_path(root: Path, archive: ModelFile) -> Path:
    return root / ".downloads" / f"{archive.sha256}-{archive.asset_name}.part"


class ModelInstaller:
    def __init__(self) -> None:
        self._mutex = threading.RLock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._task: dict = {"status": "idle", "modelIds": [], "results": {}}

    def start(self) -> None:
        """Match the API runtime service lifecycle without starting any work."""

    def close(self) -> None:
        with self._mutex:
            self._closed = True
            self._cancel.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=15)

    def snapshot(self) -> dict:
        from src.backend_v2.font_matching.valen_runtime import runtime_ready

        with self._mutex:
            task = deepcopy(self._task)
        root = models_root()
        return {
            "directory": str(root),
            "valenRuntimeReady": runtime_ready(root),
            "sources": [
                {"id": "auto", "label": "自动（优先魔搭，失败后使用 GitHub）", "configured": True},
                {"id": "modelscope", "label": "魔搭社区", "configured": "modelscope" in download_sources()},
                {"id": "github", "label": "GitHub", "configured": True},
            ],
            "models": [
                {
                    "id": model.model_id, "label": model.label, "description": model.description,
                    "sizeBytes": model.archive.size, "installed": model_available(model.model_id),
                    "directory": str(installed_root(model.model_id) / model.directory),
                    "partialBytes": _partial_size(_partial_path(root, model.archive), model.archive.size),
                    **({"optional": True} if model.repository else {}),
                }
                for model in MODELS_BY_ID.values()
            ],
            "task": task,
        }

    def install(self, model_ids: object, *, source: str = "auto", verify: bool = False) -> dict:
        if not isinstance(model_ids, list) or not model_ids:
            raise ValueError("请选择需要安装或校验的模型")
        if any(not isinstance(model_id, str) or model_id not in MODELS_BY_ID for model_id in model_ids):
            raise ValueError("包含未知模型")
        if source not in {"auto", "github", "modelscope"}:
            raise ValueError("未知模型下载源")
        if source == "modelscope" and source not in download_sources():
            raise ValueError("魔搭仓库尚未配置，请先使用 GitHub 下载源")
        selected = list(dict.fromkeys(model_ids))
        return self._start(
            {"modelIds": selected, "source": source, "verify": verify,
             "totalBytes": sum(MODELS_BY_ID[key].archive.size for key in selected)},
            lambda root: self._run(selected, root, source, verify),
        )

    def prepare_valen(self) -> dict:
        from src.backend_v2.font_matching.valen_runtime import prepare_runtime

        return self._start(
            {"modelIds": [], "phase": "preparing_runtime"},
            lambda root: prepare_runtime(root, self._check_cancelled),
        )

    def _start(self, task: dict, run: Callable[[Path], None]) -> dict:
        with self._mutex:
            if self._closed:
                raise ModelInstallBusy("服务正在关闭，请重新打开后重试")
            if self._thread is not None and self._thread.is_alive():
                raise ModelInstallBusy("已有模型安装或校验正在进行")
            root = models_root()
            lock = DataRootLock(root / ".downloads")
            try:
                lock.acquire()
            except DataRootAlreadyLocked as error:
                raise ModelInstallBusy("另一个程序实例正在安装模型，请稍后重试") from error
            self._cancel.clear()
            self._task = {
                "status": "running", "currentModelId": None, "currentFile": None,
                "downloadedBytes": 0, "totalBytes": 0, "results": {}, "error": None, **task,
            }
            self._thread = threading.Thread(
                target=self._run_task, args=(run, root, lock),
                name="model-installer", daemon=True,
            )
            try:
                self._thread.start()
            except BaseException:
                lock.release()
                raise
            return deepcopy(self._task)

    def cancel(self) -> dict:
        with self._mutex:
            if self._task["status"] == "running":
                self._cancel.set()
                self._task["status"] = "cancelling"
            return deepcopy(self._task)

    def _update(self, **fields) -> None:
        with self._mutex:
            self._task.update(fields)

    def _check_cancelled(self) -> None:
        if self._cancel.is_set():
            raise InstallCancelled()

    def _run_task(self, run: Callable[[Path], None], root: Path, lock: DataRootLock) -> None:
        try:
            run(root)
            with self._mutex:
                if self._task["status"] == "running":
                    self._update(status="completed")
                elif self._task["status"] == "cancelling":
                    self._update(status="cancelled")
        except InstallCancelled:
            self._update(status="cancelled")
        except Exception as error:
            LOGGER.warning("Model setup failed: %s", error)
            self._update(status="failed", error=str(error))
        finally:
            lock.release()

    def _run(self, model_ids: list[str], root: Path, source: str, verify: bool) -> None:
        completed = 0
        failed = False
        for model_id in model_ids:
            self._check_cancelled()
            model = MODELS_BY_ID[model_id]
            self._update(currentModelId=model_id)
            try:
                self._install_model(model, root, source, completed, verify)
                completed += model.archive.size
                self._update(downloadedBytes=completed)
                result = {"status": "completed", "error": None}
            except InstallCancelled:
                raise
            except Exception as error:
                failed = True
                LOGGER.warning("Model %s installation failed: %s", model_id, error)
                result = {"status": "failed", "error": str(error)}
            with self._mutex:
                self._task["results"][model_id] = result
        self._update(currentModelId=None, currentFile=None)
        if failed:
            self._update(status="failed")

    def _install_model(self, model: LocalModel, root: Path, source: str, completed: int, verify: bool) -> None:
        source_root = installed_root(model.model_id)
        for file in model.files:
            self._check_cancelled()
            self._update(currentFile=file.path, downloadedBytes=completed, phase="verifying")
            if not file_matches(source_root / file.path, file):
                if verify:
                    raise ValueError(f"文件缺失或校验不匹配：{file.path}，请点击下载/修复")
                archive = self._download_archive(model, root, source, completed)
                self._extract(model, archive, root)
                archive.unlink()
                return

    def _download_archive(self, model: LocalModel, root: Path, source: str, completed: int) -> Path:
        if not model.github_parts or source == "modelscope":
            return self._download(model.archive, root, download_urls(model.archive, source),
                                  completed, extract_bytes=model.size)
        if source == "auto" and "modelscope" in download_sources():
            try:
                return self._download(model.archive, root, download_urls(model.archive, "modelscope"),
                                      completed, extract_bytes=model.size)
            except RuntimeError as error:
                LOGGER.info("ModelScope download failed for %s, trying GitHub: %s", model.model_id, error)

        # The release assets are byte ranges of the same pinned ZIP. Resume in
        # that ZIP directly, including when the prefix came from ModelScope.
        file = model.archive
        partial = _partial_path(root, file)
        partial.parent.mkdir(parents=True, exist_ok=True)
        if partial.exists() and partial.stat().st_size > file.size:
            partial.unlink()
        downloaded = partial.stat().st_size if partial.exists() else 0
        if downloaded < file.size:
            self._require_download_space(partial.parent, file.size - downloaded + model.size)
        start = 0
        for part in model.github_parts:
            self._check_cancelled()
            if downloaded < start + part.size:
                url, = download_urls(part, "github")
                self._update(currentFile=part.asset_name, phase="downloading", downloadHost=urlsplit(url).hostname)
                self._stream(url, partial, part, max(0, downloaded - start), completed + start, start=start)
            start += part.size
        return self._verify_download(partial, file)

    @staticmethod
    def _require_download_space(staging: Path, required: int) -> None:
        if shutil.disk_usage(staging).free < required + 64 * 1024 * 1024:
            raise OSError(f"磁盘空间不足，安装当前文件还需约 {required / 1024 ** 2:.0f} MiB")

    def _verify_download(self, partial: Path, file: ModelFile) -> Path:
        self._check_cancelled()
        self._update(phase="verifying")
        if not file_matches(partial, file):
            partial.unlink(missing_ok=True)
            raise ValueError(f"文件校验失败：{file.path}，请重试或切换下载源")
        return partial

    def _download(self, file: ModelFile, root: Path, urls: tuple[str, ...],
                  completed: int, *, extract_bytes: int = 0) -> Path:
        staging = root / ".downloads"
        staging.mkdir(parents=True, exist_ok=True)
        partial = _partial_path(root, file)
        self._update(currentFile=file.asset_name)
        last_error: Exception | None = None
        for url in urls:
            self._check_cancelled()
            try:
                if partial.exists() and partial.stat().st_size > file.size:
                    partial.unlink()
                offset = partial.stat().st_size if partial.exists() else 0
                if offset < file.size:
                    self._require_download_space(staging, file.size - offset + extract_bytes)
                    self._update(phase="downloading", downloadHost=urlsplit(url).hostname)
                    self._stream(url, partial, file, offset, completed)
                return self._verify_download(partial, file)
            except InstallCancelled:
                raise
            except (requests.RequestException, OSError, ValueError) as error:
                last_error = error
        raise RuntimeError(str(last_error)) from last_error

    def _extract(self, model: LocalModel, archive: Path, root: Path) -> None:
        self._update(phase="extracting")
        if shutil.disk_usage(archive.parent).free < model.size + 64 * 1024 * 1024:
            raise OSError("磁盘空间不足，无法解压模型；已下载的 ZIP 会保留供重试")
        # Validate every member before replacing any existing model files. Only
        # catalogued files are accepted; ZIP paths never choose install targets.
        with zipfile.ZipFile(archive) as package, TemporaryDirectory(prefix="extract-", dir=archive.parent) as temporary:
            members = package.infolist()
            expected = {file.path: file for file in (*model.files, *model.archive_docs)}
            if len(members) != len(expected) or {member.filename for member in members} != expected.keys():
                raise ValueError("模型 ZIP 文件列表不匹配")
            for member in members:
                self._check_cancelled()
                file = expected[member.filename]
                if member.file_size != file.size or member.is_dir() or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"模型 ZIP 成员无效：{file.path}")
                target = Path(temporary) / file.path
                if not target.resolve().is_relative_to(Path(temporary).resolve()):
                    raise ValueError("模型文件路径无效")
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                self._update(currentFile=file.path)
                with package.open(member) as source, target.open("wb") as output:
                    while chunk := source.read(256 * 1024):
                        self._check_cancelled()
                        output.write(chunk)
                        digest.update(chunk)
                if digest.hexdigest() != file.sha256:
                    raise ValueError(f"文件校验失败：{file.path}")
            self._check_cancelled()
            for file in expected.values():
                destination = root / file.path
                destination.parent.mkdir(parents=True, exist_ok=True)
                (Path(temporary) / file.path).replace(destination)

    def _stream(self, url: str, partial: Path, file: ModelFile, offset: int, completed: int, *, start: int = 0) -> None:
        headers = {"Accept-Encoding": "identity", "User-Agent": "Saber-Translator-model-installer"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        # The manifests contain only public files. Disable netrc credential lookup.
        with requests.Session() as session:
            session.auth = lambda request: request
            with session.get(url, headers=headers, stream=True, timeout=(10, 10)) as response:
                if response.status_code == 404:
                    raise ValueError("此版本的模型文件尚未发布到该下载源，请联系维护者或切换下载源")
                response.raise_for_status()
                # ModelScope's Git-backed files can return a valid byte range
                # with HTTP 200; use the same strict range validation for both.
                if response.status_code == 206 or (response.status_code == 200 and response.headers.get("Content-Range")):
                    expected = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not expected or tuple(map(int, expected.groups())) != (offset, file.size - 1, file.size):
                        raise ValueError("下载源返回了无效的续传范围，请切换下载源")
                elif response.status_code == 200:
                    offset = 0
                else:
                    raise ValueError(f"下载源返回了异常状态 {response.status_code}")
                self._update(downloadedBytes=completed + offset)
                with partial.open("r+b" if partial.exists() else "wb") as handle:
                    handle.seek(start + offset)
                    handle.truncate()
                    received = offset
                    for chunk in response.iter_content(chunk_size=256 * 1024):
                        self._check_cancelled()
                        if received + len(chunk) > file.size:
                            raise ValueError("下载文件大小超过清单，请切换下载源")
                        handle.write(chunk)
                        received += len(chunk)
                        self._update(downloadedBytes=completed + received)
                    if received != file.size:
                        raise ValueError("下载未完成，请重试以继续下载")
