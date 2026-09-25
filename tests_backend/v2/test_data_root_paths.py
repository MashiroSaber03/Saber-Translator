import sys

import pytest

from src.backend_v2.paths import project_root, resolve_data_root
from src.backend_v2.storage.lifecycle import initialize_database
from src.storage_migrator.control import control_root
from src.storage_migrator.contracts import read_identity
from src.storage_migrator.runner import StorageManager
from src.version import STORAGE_VERSION


@pytest.mark.parametrize("has_appdata", [True, False])
def test_packaged_default_stays_beside_exe_despite_cwd_and_legacy_data(tmp_path, monkeypatch, has_appdata):
    bundle = tmp_path / "portable app"
    bundle.mkdir()
    elsewhere = tmp_path / "working-directory"
    elsewhere.mkdir()
    legacy = tmp_path / "AppData" / "SaberTranslator" / "data-v2"
    legacy.mkdir(parents=True)
    old_db = legacy / "saber.sqlite3"
    old_db.write_bytes(b"legacy database must not be touched")
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("SABER_V2_DATA_ROOT", str(legacy))
    if has_appdata:
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData"))
    else:
        monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(bundle / "Saber-Translator.exe"))
    root = resolve_data_root()
    assert root == bundle / "data-v2"
    assert control_root(root) == bundle / ".data-v2.saber-storage"
    with StorageManager(root, "local") as manager, manager.startup_guard():
        manager.prepare(initialize_database)
    assert read_identity(root) == (STORAGE_VERSION, "local")
    assert old_db.read_bytes() == b"legacy database must not be touched"
    assert list(elsewhere.iterdir()) == []


@pytest.mark.parametrize("frozen", [True, False])
def test_only_explicit_data_root_overrides_default(tmp_path, monkeypatch, frozen):
    monkeypatch.setattr(sys, "frozen", frozen, raising=False)
    monkeypatch.setenv("SABER_V2_DATA_ROOT", str(tmp_path / "configured"))
    assert resolve_data_root() != tmp_path / "configured"
    assert resolve_data_root(tmp_path / "explicit") == tmp_path / "explicit"


def test_source_default_is_project_relative_not_cwd(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.setenv("SABER_V2_DATA_ROOT", str(tmp_path / "ignored"))
    monkeypatch.chdir(tmp_path)
    assert resolve_data_root() == project_root() / "data-v2"
