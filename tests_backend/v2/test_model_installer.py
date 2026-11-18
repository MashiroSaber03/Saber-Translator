from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
import zipfile

from flask import Flask
import pytest

from src.backend_v2 import local_models, model_installer
from src.backend_v2.api.model_routes import create_model_blueprint
from src.backend_v2.model_installer import ModelInstallBusy, ModelInstaller
from src.backend_v2.runtime_profile import resolve_runtime_profile
from src.shared import model_catalog
from src.shared.model_catalog import LocalModel, ModelFile, MissingModelsError


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    contents = {"one/weight.bin": b"model-one" * 131072, "one/config.json": b'{"model":"one"}',
                "two/weight.bin": b"model-two" * 131072}
    def file(name):
        data = contents[name]
        return ModelFile(name, len(data), hashlib.sha256(data).hexdigest())
    def package(model_id, paths):
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            for path in paths:
                archive.writestr(path, contents[path])
        name = f"models/{model_id}/{model_id}.zip"
        contents[name] = output.getvalue()
        return file(name)
    models = (
        LocalModel("detector_yolo", "检测模型", "测试检测模型", "one", (file("one/weight.bin"), file("one/config.json")), package("detector_yolo", ("one/weight.bin", "one/config.json"))),
        LocalModel("manga_ocr", "识别模型", "测试识别模型", "two", (file("two/weight.bin"),), package("manga_ocr", ("two/weight.bin",))),
    )
    for module in (model_catalog, model_installer):
        monkeypatch.setattr(module, "LOCAL_MODELS", models)
        monkeypatch.setattr(module, "MODELS_BY_ID", {model.model_id: model for model in models})
    monkeypatch.setenv("SABER_MODELS_DIR", str(tmp_path / "models"))
    return models, contents


@contextmanager
def server(catalog, monkeypatch, *, ignore_range=False, range_status=206, corrupt=False, truncate_once=False, not_found=False, slow=False):
    models, contents = catalog
    by_name = {model.archive.asset_name: contents[model.archive.path] for model in models}
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            data = by_name[self.path.rsplit("/", 1)[-1]]
            requested = self.headers.get("Range")
            requests.append(requested)
            if not_found:
                self.send_error(404)
                return
            offset = int(requested.split("=")[1].split("-")[0]) if requested and not ignore_range else 0
            self.send_response(range_status if offset else 200)
            self.send_header("Content-Length", str(len(data) - offset))
            if offset:
                self.send_header("Content-Range", f"bytes {offset}-{len(data)-1}/{len(data)}")
            self.end_headers()
            payload = data[offset:]
            if corrupt:
                payload = bytes([payload[0] ^ 1]) + payload[1:]
            if truncate_once and len(requests) == 1:
                payload = payload[:len(payload)//2]
            try:
                for index in range(0, len(payload), 16384):
                    self.wfile.write(payload[index:index+16384])
                    if slow:
                        time.sleep(.01)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{http.server_port}"
    monkeypatch.setattr(model_installer, "download_urls", lambda file, source: (base + "/" + file.asset_name,))
    try:
        yield requests
    finally:
        http.shutdown()
        http.server_close()
        thread.join(5)


def wait(installer):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        snapshot = installer.snapshot()
        if snapshot["task"]["status"] not in {"running", "cancelling"}:
            if installer._thread:
                installer._thread.join(1)
            return snapshot
        time.sleep(.01)
    raise AssertionError(installer.snapshot())


def test_catalog_is_single_authority_and_contains_only_safe_unique_paths():
    ids = set()
    paths = set()
    for model in model_catalog.LOCAL_MODELS:
        assert model.files
        assert model.model_id not in ids
        ids.add(model.model_id)
        for file in model.files:
            assert file.size > 0 and len(file.sha256) == 64
            assert not Path(file.path).is_absolute() and ".." not in Path(file.path).parts
            assert file.path.startswith(model.directory + "/")
            assert file.path not in paths
            paths.add(file.path)
    assert ids == set(local_models.LOCAL_MODEL_IDS)


def test_empty_startup_does_not_download_or_load_resident_models(catalog, monkeypatch):
    monkeypatch.setattr(local_models, "_load_local_model", lambda _: pytest.fail("loaded missing weights"))
    assert local_models.preload_local_models(["detector_yolo", "manga_ocr"]) == ()
    installer = ModelInstaller()
    installer.start()
    assert installer.snapshot()["task"]["status"] == "idle"
    assert not model_catalog.models_root().exists()
    with pytest.raises(MissingModelsError, match="模型管理") as error:
        model_catalog.require_local_models(["manga_ocr", "detector_yolo", "manga_ocr"])
    assert error.value.model_ids == ("manga_ocr", "detector_yolo")
    installer.close()


def test_invalid_resident_weights_do_not_prevent_opening_model_management(monkeypatch):
    monkeypatch.setattr(local_models, "model_available", lambda _: True)
    released = []
    def load(key):
        if key == "detector_yolo":
            raise ValueError("corrupt checkpoint")
    monkeypatch.setattr(local_models, "_load_local_model", load)
    monkeypatch.setattr(local_models, "release_loaded_local_model", released.append)
    assert local_models.preload_local_models(["detector_yolo", "manga_ocr"]) == ("manga_ocr",)
    assert released == ["detector_yolo"]


def test_explicit_model_directory_never_merges_partial_groups(catalog, monkeypatch, tmp_path):
    models, contents = catalog
    legacy = tmp_path / "legacy"
    for name, data in contents.items():
        path = legacy / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    monkeypatch.setattr(model_catalog, "resource_path", lambda _: str(legacy))
    # An explicit empty installation root isolates even a complete old bundle.
    assert not model_catalog.model_available("detector_yolo")
    assert model_catalog.installed_root("detector_yolo") == model_catalog.models_root()
    monkeypatch.delenv("SABER_MODELS_DIR")
    monkeypatch.setattr(model_catalog, "models_root", lambda: tmp_path / "app/models")
    assert model_catalog.model_available("detector_yolo")
    assert model_catalog.installed_root("detector_yolo") == legacy
    # A new incomplete directory must not shadow that complete legacy model.
    target = model_catalog.models_root() / models[0].files[0].path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(contents[models[0].files[0].path])
    assert model_catalog.installed_root("detector_yolo") == legacy


def test_packaged_models_live_beside_executable(monkeypatch, tmp_path):
    monkeypatch.delenv("SABER_MODELS_DIR", raising=False)
    monkeypatch.setattr(model_catalog.sys, "frozen", True, raising=False)
    monkeypatch.setattr(model_catalog.sys, "executable", str(tmp_path / "app/Saber-Translator.exe"))
    assert model_catalog.models_root() == tmp_path / "app/models"


def test_modelscope_can_be_configured_once_in_manifest_for_all_deployments(monkeypatch):
    monkeypatch.delenv("SABER_MODELS_MODELSCOPE_REPO", raising=False)
    monkeypatch.delenv("SABER_MODELS_MODELSCOPE_REVISION", raising=False)
    monkeypatch.setattr(model_catalog, "_manifest", lambda: {"modelscope": {"repository": "owner/models", "revision": "fixed-revision"}})
    assert "modelscope" in model_catalog.download_sources()
    file = model_catalog.LOCAL_MODELS[0].archive
    urls = model_catalog.download_urls(file, "auto")
    assert len(urls) == 2 and urls[0].startswith("https://modelscope.cn/") and urls[1].startswith("https://github.com/")
    assert "Revision=fixed-revision" in urls[0]
    assert "FilePath=detector_default%2Fdetector_default.zip" in urls[0]
    assert urls[1].endswith("/detector_default.zip")


def test_individual_then_install_all_reuses_verified_files_and_hot_availability(catalog, monkeypatch):
    with server(catalog, monkeypatch) as requests:
        installer = ModelInstaller()
        try:
            installer.install(["detector_yolo"])
            first = wait(installer)
            assert first["task"]["status"] == "completed"
            assert [m["installed"] for m in first["models"]] == [True, False]
            assert len(requests) == 1
            installer.install(["detector_yolo", "manga_ocr", "detector_yolo"])
            assert wait(installer)["task"]["status"] == "completed"
            assert len(requests) == 2
            model_catalog.require_local_models(["detector_yolo", "manga_ocr"])
            installer.install(["detector_yolo", "manga_ocr"], verify=True)
            assert wait(installer)["task"]["status"] == "completed"
            assert len(requests) == 2
        finally:
            installer.close()


@pytest.mark.parametrize("ignore_range,range_status", [(False, 206), (True, 206), (False, 200)])
def test_resume_after_process_restart_handles_range_or_full_response(catalog, monkeypatch, ignore_range, range_status):
    model, _ = catalog[0]
    file = model.archive
    partial = model_installer._partial_path(model_catalog.models_root(), file)
    partial.parent.mkdir(parents=True)
    partial.write_bytes(catalog[1][file.path][:100000])
    with server(catalog, monkeypatch, ignore_range=ignore_range, range_status=range_status) as requests:
        installer = ModelInstaller()
        try:
            installer.install([model.model_id])
            assert wait(installer)["task"]["status"] == "completed"
            assert requests[0] == "bytes=100000-"
            assert all(model_catalog.file_matches(model_catalog.models_root() / member.path, member) for member in model.files)
            assert not partial.exists()
        finally:
            installer.close()


def test_interrupted_download_preserves_partial_and_retry_continues(catalog, monkeypatch):
    with server(catalog, monkeypatch, truncate_once=True) as requests:
        installer = ModelInstaller()
        try:
            installer.install(["detector_yolo"])
            failed = wait(installer)
            assert failed["task"]["status"] == "failed"
            assert failed["models"][0]["partialBytes"] > 0
            assert not failed["models"][0]["installed"]
            installer.install(["detector_yolo"])
            assert wait(installer)["task"]["status"] == "completed"
            assert requests[1].startswith("bytes=")
        finally:
            installer.close()


def test_cancel_keeps_existing_model_and_releases_cross_process_lock(catalog, monkeypatch):
    with server(catalog, monkeypatch, slow=True):
        installer, other = ModelInstaller(), ModelInstaller()
        try:
            installer.install(["detector_yolo"])
            with pytest.raises(ModelInstallBusy):
                other.install(["manga_ocr"])
            deadline = time.monotonic() + 3
            while installer.snapshot()["task"]["downloadedBytes"] == 0 and time.monotonic() < deadline:
                time.sleep(.01)
            installer.cancel()
            state = wait(installer)
            assert state["task"]["status"] == "cancelled"
            assert state["models"][0]["partialBytes"] > 0
            other.install(["detector_yolo"])
            assert wait(other)["task"]["status"] == "completed"
        finally:
            installer.close()
            other.close()


def test_checksum_failure_never_publishes_broken_weights(catalog, monkeypatch):
    with server(catalog, monkeypatch, corrupt=True):
        installer = ModelInstaller()
        try:
            installer.install(["detector_yolo", "manga_ocr"])
            state = wait(installer)
            assert state["task"]["status"] == "failed"
            assert all(not model["installed"] for model in state["models"])
            assert all("校验失败" in result["error"] for result in state["task"]["results"].values())
            assert not list(model_catalog.models_root().rglob("*.part"))
        finally:
            installer.close()


def test_disk_full_fails_before_downloading_and_keeps_models_uninstalled(catalog, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(model_installer.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with server(catalog, monkeypatch) as requests:
        installer = ModelInstaller()
        try:
            installer.install(["detector_yolo"])
            state = wait(installer)
            assert state["task"]["status"] == "failed"
            assert "磁盘空间不足" in state["task"]["results"]["detector_yolo"]["error"]
            assert not state["models"][0]["installed"] and requests == []
        finally:
            installer.close()


def test_auto_source_falls_back_without_a_second_installation_path(catalog, monkeypatch):
    with server(catalog, monkeypatch) as requests:
        local_urls = model_installer.download_urls
        monkeypatch.setattr(model_installer, "download_urls", lambda file, source: ("http://127.0.0.1:1/unavailable", *local_urls(file, source)))
        installer = ModelInstaller()
        try:
            installer.install(["detector_yolo"])
            assert wait(installer)["task"]["status"] == "completed"
            assert len(requests) == 1
        finally:
            installer.close()


def test_unpublished_source_is_clear_and_unconfigured_modelscope_is_rejected(catalog, monkeypatch):
    monkeypatch.delenv("SABER_MODELS_MODELSCOPE_REPO", raising=False)
    monkeypatch.delenv("SABER_MODELS_MODELSCOPE_REVISION", raising=False)
    monkeypatch.setattr(model_catalog, "_manifest", lambda: {"modelscope": None})
    with server(catalog, monkeypatch, not_found=True):
        installer = ModelInstaller()
        try:
            with pytest.raises(ValueError, match="尚未配置"):
                installer.install(["detector_yolo"], source="modelscope")
            installer.install(["detector_yolo"])
            state = wait(installer)
            assert "尚未发布" in state["task"]["results"]["detector_yolo"]["error"]
            assert not next(source for source in state["sources"] if source["id"] == "modelscope")["configured"]
        finally:
            installer.close()


def test_complete_legacy_bundle_is_reused_and_repair_builds_complete_external_model(catalog, monkeypatch, tmp_path):
    models, contents = catalog
    legacy, external = tmp_path / "bundle/models", tmp_path / "app/models"
    monkeypatch.delenv("SABER_MODELS_DIR")
    monkeypatch.setattr(model_catalog, "models_root", lambda: external)
    monkeypatch.setattr(model_installer, "models_root", lambda: external)
    monkeypatch.setattr(model_catalog, "resource_path", lambda _: str(legacy))
    for name, data in contents.items():
        path = legacy / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    with server(catalog, monkeypatch) as requests:
        installer = ModelInstaller()
        try:
            installer.install(["detector_yolo"])
            assert wait(installer)["task"]["status"] == "completed"
            assert requests == []
            assert model_catalog.installed_root("detector_yolo") == legacy
            (legacy / models[0].files[0].path).write_bytes(b"old weights")
            installer.install(["detector_yolo"])
            assert wait(installer)["task"]["status"] == "completed"
            assert len(requests) == 1
            assert model_catalog.installed_root("detector_yolo") == external
            assert all(model_catalog.file_matches(external / file.path, file) for file in models[0].files)
        finally:
            installer.close()


def test_public_users_cannot_read_paths_download_or_cancel(catalog, monkeypatch):
    app = Flask(__name__)
    installer = ModelInstaller()
    app.register_blueprint(create_model_blueprint(installer=installer, profile=resolve_runtime_profile("public")))
    monkeypatch.setattr("src.backend_v2.api.model_routes.require_admin", lambda: (_ for _ in ()).throw(PermissionError()))
    app.register_error_handler(PermissionError, lambda _: ({"error": "forbidden"}, 403))
    with app.test_client() as client:
        assert client.get("/api/v2/system/models").status_code == 403
        assert client.post("/api/v2/system/models/install", json={"modelIds": ["detector_yolo"]}).status_code == 403
        assert client.post("/api/v2/system/models/cancel").status_code == 403
    installer.close()


def test_runtime_and_extension_inventory_follow_install_and_file_changes(catalog, monkeypatch, tmp_path):
    from sqlalchemy import create_engine
    from src.backend_v2.api.app import ApiSettings, _create_v2_blueprint
    from src.backend_v2.browser_extension.auth import BrowserExtensionAccess
    from src.backend_v2.browser_extension.routes import create_browser_extension_blueprint
    from src.backend_v2.runtime_identity import RuntimeIdentity

    models, contents = catalog
    app = Flask(__name__)
    engine = create_engine("sqlite://")
    app.register_blueprint(_create_v2_blueprint(ApiSettings(
        data_root=tmp_path, engine=engine,
        identity=RuntimeIdentity(epoch_id="inventory", epoch_token="test", test_mode=True),
    )))
    token = "inventory-test-token-12345678901234567890"
    app.register_blueprint(create_browser_extension_blueprint(
        service=object(), access=BrowserExtensionAccess(enabled=True, token=token),
    ))
    client = app.test_client()
    extension_path = "/api/v2/browser-extension/manage/system/models/installed"
    assert client.get(extension_path).status_code == 401

    def inventory():
        response = client.get("/api/v2/system/capabilities")
        assert response.status_code == 200
        result = response.get_json()["installedModels"]
        extension = client.get(extension_path, headers={"Authorization": f"Bearer {token}"})
        assert extension.status_code == 200
        assert extension.get_json() == {"installedModels": result}
        return result

    installer = ModelInstaller()
    try:
        assert inventory() == []
        # A partial model folder is not an installation, nor is a cached ZIP.
        weight = model_catalog.models_root() / models[0].files[0].path
        weight.parent.mkdir(parents=True)
        weight.write_bytes(contents[models[0].files[0].path])
        assert inventory() == []
        with server(catalog, monkeypatch):
            installer.install([model.model_id for model in models])
            assert wait(installer)["task"]["status"] == "completed"
        assert inventory() == ["detector_yolo", "manga_ocr"]
        config = model_catalog.models_root() / models[0].files[1].path
        config.unlink()
        assert inventory() == ["manga_ocr"]
        config.write_bytes(contents[models[0].files[1].path])
        assert inventory() == ["detector_yolo", "manga_ocr"]
        weight.write_bytes(b"")
        assert inventory() == ["manga_ocr"]
    finally:
        installer.close()
        engine.dispose()


@pytest.mark.parametrize("problem", ["missing", "unexpected", "duplicate", "traversal", "symlink", "wrong-content"])
def test_invalid_zip_never_replaces_existing_model_files(catalog, monkeypatch, problem):
    models, contents = catalog
    model = models[0]
    root = model_catalog.models_root()
    for file in model.files:
        path = root / file.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"original installed data")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for file in model.files:
            if problem == "missing" and file == model.files[-1]:
                continue
            info = zipfile.ZipInfo(file.path)
            if problem == "symlink" and file == model.files[0]:
                info.create_system = 3
                info.external_attr = 0o120777 << 16
            data = contents[file.path]
            if problem == "wrong-content" and file == model.files[-1]:
                data = b"x" * len(data)
            archive.writestr(info, data)
        if problem in {"unexpected", "duplicate", "traversal"}:
            name = {"unexpected": "one/extra.py", "duplicate": model.files[0].path, "traversal": "../outside.bin"}[problem]
            if problem == "duplicate":
                with pytest.warns(UserWarning, match="Duplicate name"):
                    archive.writestr(name, b"invalid member")
            else:
                archive.writestr(name, b"invalid member")
    package = output.getvalue()
    contents[model.archive.path] = package
    changed = replace(model, archive=replace(model.archive, size=len(package), sha256=hashlib.sha256(package).hexdigest()))
    for module in (model_catalog, model_installer):
        monkeypatch.setattr(module, "LOCAL_MODELS", (changed, models[1]))
        monkeypatch.setattr(module, "MODELS_BY_ID", {changed.model_id: changed, models[1].model_id: models[1]})
    with server(((changed, models[1]), contents), monkeypatch):
        installer = ModelInstaller()
        try:
            installer.install([changed.model_id])
            assert wait(installer)["task"]["status"] == "failed"
            assert all((root / file.path).read_bytes() == b"original installed data" for file in model.files)
            assert not (root / "outside.bin").exists()
            assert not list((root / ".downloads").glob("extract-*"))
        finally:
            installer.close()


def test_cancel_during_extraction_keeps_old_files_and_reuses_verified_zip(catalog, monkeypatch):
    model = catalog[0][0]
    root = model_catalog.models_root()
    for file in model.files:
        path = root / file.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"original installed data")
    installer = ModelInstaller()
    original_update = installer._update
    def update(**fields):
        original_update(**fields)
        if fields.get("phase") == "extracting":
            installer.cancel()
    monkeypatch.setattr(installer, "_update", update)
    with server(catalog, monkeypatch) as requests:
        try:
            installer.install([model.model_id])
            assert wait(installer)["task"]["status"] == "cancelled"
            assert all((root / file.path).read_bytes() == b"original installed data" for file in model.files)
            assert model_catalog.file_matches(model_installer._partial_path(root, model.archive), model.archive)
            monkeypatch.setattr(installer, "_update", original_update)
            installer.install([model.model_id])
            assert wait(installer)["task"]["status"] == "completed"
            assert len(requests) == 1
            assert all(model_catalog.file_matches(root / file.path, file) for file in model.files)
        finally:
            installer.close()
