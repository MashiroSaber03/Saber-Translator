"""Application-only releases must keep admitting the same stored data."""

from pathlib import Path
import json
import runpy
import shutil
import subprocess
import sys

import pytest


def test_manifest_loads_independent_versions(tmp_path: Path) -> None:
    source = tmp_path / 'src'
    source.mkdir()
    shutil.copy2(Path(__file__).resolve().parents[2] / 'src/version.py', source / 'version.py')
    manifest = tmp_path / 'version.json'
    manifest.write_text(json.dumps({'version': '3.5.3', 'storageVersion': '3.5.2'}))
    versions = runpy.run_path(str(source / 'version.py'))
    assert versions['APP_VERSION'] == '3.5.3'
    assert versions['STORAGE_VERSION'] == '3.5.2'
    manifest.write_text(json.dumps({'version': '3.5.3', 'storageVersion': 'invalid'}))
    with pytest.raises(ValueError, match='Invalid version'):
        runpy.run_path(str(source / 'version.py'))


@pytest.mark.parametrize('app_version', ['3.5.3', '3.5.1'])
def test_app_version_does_not_control_storage(tmp_path: Path, app_version: str) -> None:
    script = '''
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import src.version as version
version.APP_VERSION = APP_VERSION
from src.backend_v2.storage.lifecycle import initialize_database
from src.backend_v2.storage.startup import current_storage_process
from src.storage_migrator.runner import StorageManager
from src.storage_migrator.contracts import read_identity
from src.backend_v2.api.app import ApiSettings, create_api_app
from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.runtime_identity import RuntimeIdentity
from src.backend_v2.worker.entrypoint import _write_ready_marker
from src.backend_v2.launcher import entrypoint as launcher

root = Path(DATA_ROOT)
assert initialize_database(root).created
assert read_identity(root)[0] == version.STORAGE_VERSION
with StorageManager(root, 'local') as migration:
    assert migration.target == version.STORAGE_VERSION
    assert migration.prepare(initialize_database)['status'] == 'current'
    assert not list(migration._operations())
assert not initialize_database(root).created
engine = create_sqlite_engine(root/'saber.sqlite3')
identity = RuntimeIdentity('version-test', 'test-token', True)
app = create_api_app(ApiSettings(root, identity, engine))
try:
    with app.test_client() as client:
        health = client.get('/api/v2/health').get_json()
    assert health['storageVersion'] == version.STORAGE_VERSION
    with patch.object(launcher, '_read_api_health', return_value=(200, health)):
        launcher._wait_for_api(0, expected_epoch_id=identity.epoch_id, expected_epoch_token=identity.epoch_token, child=SimpleNamespace(poll=lambda: None), timeout_seconds=1)
    (root/'runtime').mkdir(exist_ok=True)
    _write_ready_marker(root, identity)
    assert json.loads((root/'runtime/worker-ready.json').read_text())['storageVersion'] == version.STORAGE_VERSION
    launcher._wait_for_worker(root, expected_epoch_id=identity.epoch_id, child=SimpleNamespace(poll=lambda: None), timeout_seconds=1)
    for role in ('api', 'worker'):
        with patch.object(RuntimeIdentity, 'for_'+role, return_value=identity):
            admitted = current_storage_process(role)(lambda args: True)
            assert admitted(SimpleNamespace(test_mode=False, data_dir=root, profile='local'))
finally:
    app.extensions['saber_v2_runtime'].close()
    engine.dispose()
'''
    result = subprocess.run(
        [sys.executable, '-c', f'APP_VERSION={app_version!r}\nDATA_ROOT={str(tmp_path / "data")!r}\n' + script],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True,
        encoding='utf-8', errors='replace', timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
