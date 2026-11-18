import hashlib
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import requests

from src.shared.model_catalog import LocalModel, ModelFile

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("publish_model_files", ROOT / ".github/scripts/publish_models.py")
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


@pytest.fixture
def model(tmp_path, monkeypatch):
    data = b"real stream test data"
    file = ModelFile("test/weight.bin", len(data), hashlib.sha256(data).hexdigest())
    path = tmp_path / file.path
    path.parent.mkdir()
    path.write_bytes(data)
    package = LocalModel("test", "Test", "Test", "test", (file,), ModelFile("test/test.zip", 0, ""))
    zip_path = tmp_path / package.archive.path
    publisher.write_archive(tmp_path, [asdict(file)], zip_path)
    archive_data = zip_path.read_bytes()
    archive = ModelFile(package.archive.path, len(archive_data), hashlib.sha256(archive_data).hexdigest())
    monkeypatch.setattr(publisher, "LOCAL_MODELS", (LocalModel("test", "Test", "Test", "test", (file,), archive),))
    return tmp_path, archive, archive_data


@pytest.mark.parametrize("state", ["missing", "uploaded", "starter", "mismatch", "untagged-draft"])
def test_model_publication_streams_and_never_replaces_uploaded_assets(model, monkeypatch, state):
    root, file, data = model
    calls = []
    assets = [] if state in {"missing", "untagged-draft"} else [{"id": 7, "name": file.asset_name, "size": 0 if state == "starter" else file.size,
        "state": "starter" if state == "starter" else "uploaded", "digest": "sha256:" + ("0" * 64 if state == "mismatch" else file.sha256)}]
    release = {"id": 1, "tag_name": publisher.MODEL_RELEASE_TAG, "draft": True, "assets": assets, "upload_url": "https://uploads.github.com/repos/owner/repo/releases/1/assets{?name,label}"}

    class Response:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return release

    class Session:
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def get(self, url, **_kwargs):
            response = Response()
            if state == "untagged-draft":
                if '/tags/' in url:
                    response.status_code = 404
                else:
                    response.json = lambda: [release]
            return response
        def delete(self, url, **_kwargs):
            calls.append(("delete", url))
            return Response()
        def patch(self, url, **kwargs):
            assert kwargs["json"] == {"draft": False, "make_latest": "true"}
            calls.append(("publish", url))
            return Response()
        def post(self, url, **kwargs):
            handle = kwargs["data"]
            assert not isinstance(handle, (bytes, str)) and handle.read() == data
            assert kwargs["headers"]["Content-Length"] == str(len(data))
            assert kwargs["params"] == {"name": file.asset_name}
            calls.append(("upload", url))
            response = Response()
            response.json = lambda: {"name": file.asset_name, "size": len(data), "digest": "sha256:" + file.sha256}
            return response

    monkeypatch.setenv("GH_TOKEN", "test-only-token")
    monkeypatch.setattr(requests, "Session", Session)
    if state == "mismatch":
        with pytest.raises(ValueError, match="Refusing to replace"):
            publisher.publish(root, "owner/repo", "dev", dry_run=False)
        assert calls == []
    else:
        publisher.publish(root, "owner/repo", "dev", dry_run=False)
        assert [call[0] for call in calls] == {"missing": ["upload", "publish"], "uploaded": ["publish"], "starter": ["delete", "upload", "publish"], "untagged-draft": ["upload", "publish"]}[state]


def test_offline_publication_plan_checks_hashes_without_credentials(model, monkeypatch, capsys):
    root, file, _ = model
    monkeypatch.delenv("GH_TOKEN", raising=False)
    publisher.publish(root, "owner/repo", "dev", dry_run=True)
    assert file.asset_name in capsys.readouterr().out
    (root / "test/weight.bin").write_bytes(b"bad weights")
    with pytest.raises(ValueError, match="mismatched"):
        publisher.publish(root, "owner/repo", "dev", dry_run=True)


@pytest.mark.parametrize("ref", [None, "main"])
def test_new_model_release_uses_model_repository_default_branch(model, monkeypatch, ref):
    root, file, _ = model
    release = {"id": 1, "draft": True, "upload_url": "https://uploads.github.com/assets{?name}",
        "assets": [{"name": file.asset_name, "size": file.size, "digest": "sha256:" + file.sha256}]}

    class Response:
        def __init__(self, status, payload):
            self.status_code, self.payload = status, payload
        def raise_for_status(self):
            assert self.status_code == 200
        def json(self):
            return self.payload

    class Session:
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def get(self, url, **_kwargs):
            return Response(404, {}) if '/tags/' in url else Response(200, [])
        def post(self, _url, *, json, **_kwargs):
            assert json["tag_name"] == publisher.MODEL_RELEASE_TAG
            if ref is None:
                assert "target_commitish" not in json
            else:
                assert json["target_commitish"] == ref
            return Response(200, release)
        def patch(self, *_args, **_kwargs): return Response(200, release)

    monkeypatch.setenv("GH_TOKEN", "test-only-token")
    monkeypatch.setattr(requests, "Session", Session)
    publisher.publish(root, "owner/model-files", ref, dry_run=False)


def test_preparing_archives_is_reproducible_and_preserves_the_selected_release(model, monkeypatch):
    root, file, data = model
    manifest_path = root / "src/shared/model_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"releaseTag": "model-files-v9", "models": [{"id": "test", "files": [asdict(publisher.LOCAL_MODELS[0].files[0])]}]}), encoding="utf-8")
    monkeypatch.setattr(publisher, "__file__", str(root / ".github/scripts/publish_models.py"))
    publisher.prepare_archives(root, root / "first")
    first = json.loads(manifest_path.read_text(encoding="utf-8"))
    publisher.prepare_archives(root, root / "second")
    second = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert first == second and first["releaseTag"] == "model-files-v9"
    assert (root / "first" / file.path).read_bytes() == data == (root / "second" / file.path).read_bytes()


def test_prepare_cli_bootstraps_a_new_model_without_archive_metadata(tmp_path):
    script = tmp_path / ".github/scripts/publish_models.py"
    script.parent.mkdir(parents=True)
    script.write_text(Path(publisher.__file__).read_text(encoding="utf-8"), encoding="utf-8")
    weight = tmp_path / "models/test/weight.bin"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"new model weights")
    file = {"path": "test/weight.bin", "size": weight.stat().st_size, "sha256": hashlib.sha256(weight.read_bytes()).hexdigest()}
    manifest = tmp_path / "src/shared/model_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"releaseTag": "model-files-v9", "models": [{"id": "test", "files": [file]}]}), encoding="utf-8")
    result = subprocess.run([sys.executable, str(script), "--prepare-dir", str(tmp_path / "packages")], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    prepared = json.loads(manifest.read_text(encoding="utf-8"))
    assert prepared["releaseTag"] == "model-files-v9"
    archive = prepared["models"][0]["archive"]
    assert (tmp_path / "packages" / archive["path"]).stat().st_size == archive["size"]
