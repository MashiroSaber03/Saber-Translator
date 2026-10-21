"""Source deployment uses matched prebuilt files, including interrupted updates."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
from urllib.request import urlopen
from zipfile import ZipFile

import pytest

from scripts.build_editor import package_runtime, source_fingerprint
from src.backend_v2 import editor_runtime as runtime


BUILD = "a" * 24
NEXT_BUILD = "b" * 24


def write_runtime(root, build):
    for relative in runtime.runtime_files():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{build}:{relative}".encode())
    (root / runtime.STAMP).write_text(json.dumps(runtime.runtime_stamp(root, build)), encoding="utf-8")


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "source checkout 中文"
    root.mkdir()
    (root / "editor-runtime.json").write_text(json.dumps({"build": BUILD}), encoding="utf-8")
    return root


@contextmanager
def serve_bundle(tmp_path, monkeypatch, build=BUILD, *, transform=None, bad_checksum=False):
    built = tmp_path / "built"
    write_runtime(built, build)
    archive = package_runtime(built, tmp_path / "assets", build)
    if transform:
        transform(archive)
        archive.with_suffix(".zip.sha256").write_text(runtime.checksum(archive), encoding="ascii")
    if bad_checksum:
        archive.with_suffix(".zip.sha256").write_text("0" * 64, encoding="ascii")
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            path = archive.parent / self.path.rsplit("/", 1)[-1]
            if not path.is_file():
                self.send_error(404)
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def local_download(url, **kwargs):
        return urlopen(f"http://127.0.0.1:{server.server_port}/{url.rsplit('/', 1)[-1]}", **kwargs)
    monkeypatch.setattr(runtime, "urlopen", local_download)
    try:
        yield requests, archive
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_first_start_downloads_one_pair_and_repeated_start_works_offline(checkout, tmp_path, monkeypatch):
    progress = []
    with serve_bundle(tmp_path, monkeypatch) as (requests, _):
        runtime.ensure_editor_runtime(progress.append, root=checkout)
        assert len(requests) == 2
    monkeypatch.setattr(runtime, "urlopen", lambda *_args, **_kwargs: pytest.fail("matching runtime must work offline"))
    runtime.ensure_editor_runtime(progress.append, root=checkout)
    assert runtime.runtime_ready(checkout, BUILD)
    assert progress[-1] == "编辑内核已就绪"
    assert not list(checkout.glob(".tmp-editor-*"))


def test_update_replaces_both_halves_once(checkout, tmp_path, monkeypatch):
    write_runtime(checkout, BUILD)
    (checkout / "editor-runtime.json").write_text(json.dumps({"build": NEXT_BUILD}), encoding="utf-8")
    with serve_bundle(tmp_path, monkeypatch, NEXT_BUILD) as (requests, _):
        runtime.ensure_editor_runtime(lambda *_: None, root=checkout)
        runtime.ensure_editor_runtime(lambda *_: None, root=checkout)
    assert len(requests) == 2
    assert runtime.runtime_ready(checkout, NEXT_BUILD)


@pytest.mark.parametrize("damage", ["checksum", "zip", "version", "extra", "missing", "content", "truncated"])
def test_failed_download_or_validation_preserves_previous_pair(checkout, tmp_path, monkeypatch, damage):
    write_runtime(checkout, BUILD)
    before = {path: (checkout / path).read_bytes() for path in (*runtime.runtime_files(), runtime.STAMP)}
    (checkout / "editor-runtime.json").write_text(json.dumps({"build": NEXT_BUILD}), encoding="utf-8")

    def transform(archive):
        if damage in {"zip", "truncated"}:
            archive.write_bytes(b"not a zip" if damage == "zip" else archive.read_bytes()[:10])
            return
        with ZipFile(archive) as source:
            entries = {name: source.read(name) for name in source.namelist()}
        if damage == "extra":
            entries["../outside.txt"] = b"must never escape"
        elif damage == "missing":
            entries.pop(runtime.runtime_files()[0].as_posix())
        elif damage == "content":
            entries[runtime.runtime_files()[0].as_posix()] = b"corrupt executable"
        elif damage == "version":
            entries[runtime.STAMP.as_posix()] = json.dumps(runtime.runtime_stamp(checkout, BUILD)).encode()
        with ZipFile(archive, "w") as output:
            for name, body in entries.items():
                output.writestr(name, body)

    with serve_bundle(tmp_path, monkeypatch, NEXT_BUILD, transform=transform, bad_checksum=damage == "checksum"):
        with pytest.raises(RuntimeError, match="编辑内核准备失败") as caught:
            runtime.ensure_editor_runtime(lambda *_: None, root=checkout)
    assert "https://github.com/" in str(caught.value)
    assert {path: (checkout / path).read_bytes() for path in before} == before
    assert not list(checkout.glob(".tmp-editor-*"))
    assert not (tmp_path / "outside.txt").exists()


def test_replace_failure_rolls_back_all_updated_files(checkout, tmp_path, monkeypatch):
    write_runtime(checkout, BUILD)
    (checkout / "editor-runtime.json").write_text(json.dumps({"build": NEXT_BUILD}), encoding="utf-8")
    replace = Path.replace
    def denied(path, target):
        if path.name == "photocraft.js" and "unpacked" in path.parts:
            raise PermissionError("test: browser files locked")
        return replace(path, target)
    monkeypatch.setattr(Path, "replace", denied)
    with serve_bundle(tmp_path, monkeypatch, NEXT_BUILD):
        with pytest.raises(RuntimeError, match="browser files locked"):
            runtime.ensure_editor_runtime(lambda *_: None, root=checkout)
    assert runtime.runtime_ready(checkout, BUILD)


def test_missing_file_is_repaired_without_silently_accepting_stamp(checkout, tmp_path, monkeypatch):
    write_runtime(checkout, BUILD)
    (checkout / runtime.WEB_ROOT / "photocraft_bg.wasm").unlink()
    with serve_bundle(tmp_path, monkeypatch):
        runtime.ensure_editor_runtime(lambda *_: None, root=checkout)
    assert runtime.runtime_ready(checkout, BUILD)


@pytest.mark.skipif(os.name != "nt", reason="Windows directory sharing semantics")
def test_update_succeeds_with_an_open_directory_handle(checkout, tmp_path, monkeypatch):
    import ctypes
    from ctypes import wintypes

    write_runtime(checkout, BUILD)
    (checkout / "editor-runtime.json").write_text(json.dumps({"build": NEXT_BUILD}), encoding="utf-8")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel32.CreateFileW
    create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create.restype = wintypes.HANDLE
    close = kernel32.CloseHandle
    close.argtypes = (wintypes.HANDLE,)
    # Allow reading/writing children, but deny renaming the directory itself.
    handle = create(str(checkout / "native"), 0x80, 3, None, 3, 0x02000000, None)
    assert handle not in (None, ctypes.c_void_p(-1).value)
    try:
        with serve_bundle(tmp_path, monkeypatch, NEXT_BUILD):
            runtime.ensure_editor_runtime(lambda *_: None, root=checkout)
        assert runtime.runtime_ready(checkout, NEXT_BUILD)
    finally:
        close(handle)


def test_manual_zip_install_works_offline(checkout, tmp_path, monkeypatch):
    write_runtime(tmp_path / "built", BUILD)
    archive = package_runtime(tmp_path / "built", tmp_path / "assets", BUILD)
    with ZipFile(archive) as package:
        package.extractall(checkout)
    monkeypatch.setattr(runtime, "urlopen", lambda *_args, **_kwargs: pytest.fail("manual install is complete"))
    runtime.ensure_editor_runtime(root=checkout)


def test_unpublished_version_reports_url_and_preserves_previous_runtime(checkout, tmp_path, monkeypatch):
    write_runtime(checkout, BUILD)
    (checkout / "editor-runtime.json").write_text(json.dumps({"build": NEXT_BUILD}), encoding="utf-8")
    with serve_bundle(tmp_path, monkeypatch):
        with pytest.raises(RuntimeError, match="内核包尚未发布") as caught:
            runtime.ensure_editor_runtime(lambda *_: None, root=checkout)
    assert f"editor-{NEXT_BUILD}" in str(caught.value)
    assert runtime.runtime_ready(checkout, BUILD)


def test_other_preparation_cannot_install_at_the_same_time(checkout, tmp_path, monkeypatch):
    from src.storage_migrator.control import DataRootLock

    with DataRootLock(checkout / "native"), serve_bundle(tmp_path, monkeypatch) as (requests, _):
        with pytest.raises(RuntimeError, match="另一个实例正在准备"):
            runtime.ensure_editor_runtime(lambda *_: None, root=checkout)
        assert requests == []


def test_launcher_aborts_before_children_or_database_when_runtime_preparation_fails(tmp_path, monkeypatch):
    from src.backend_v2.launcher import entrypoint

    def fail(progress):
        progress("test download failed")
        raise RuntimeError("test unavailable runtime")
    monkeypatch.setattr(runtime, "ensure_editor_runtime", fail)
    monkeypatch.setattr(entrypoint, "_spawn", lambda *_args, **_kwargs: pytest.fail("must not start children"))
    states = []
    root = tmp_path / "data"
    supervisor = entrypoint.LauncherSupervisor(
        entrypoint.LauncherConfig(data_root=root, host="127.0.0.1", port=5000), status_callback=states.append,
    )
    with pytest.raises(RuntimeError, match="test unavailable"):
        supervisor.run()
    assert states[-1].state is entrypoint.LauncherState.DEGRADED
    assert not (root / "saber.sqlite3").exists()


def test_frozen_packages_do_not_read_manifest_or_contact_network(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime.sys, "frozen", True, raising=False)
    runtime.ensure_editor_runtime(root=tmp_path / "does-not-exist")


def test_build_fingerprint_matches_lf_and_crlf_checkout(tmp_path):
    root = Path(__file__).resolve().parents[2]
    # Exercise the actual input enumerator on a small source checkout.
    for relative in ("third_party/photocraft/Cargo.toml", "third_party/photocraft/Cargo.lock",
                     "third_party/photocraft/crates/text/src/layout.rs", "scripts/build_editor.py",
                     "scripts/editor/host.js", "src/backend_v2/editor_runtime.py", ".github/workflows/editor-runtime.yml",
                     *(f"third_party/photocraft/{name}" for name in runtime.NOTICE_FILES)):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / relative).read_bytes().replace(b"\r\n", b"\n"))
    before = source_fingerprint(tmp_path)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert source_fingerprint(tmp_path) == before
    (tmp_path / "scripts/editor/host.js").write_text("changed", encoding="utf-8")
    assert source_fingerprint(tmp_path) != before
