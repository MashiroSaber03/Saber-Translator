from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / '.github/scripts/upload_release_assets.py'
spec = importlib.util.spec_from_file_location('release_assets', SCRIPT)
release_assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_assets)


@pytest.mark.parametrize('pattern,old,other', [
    ('Saber-Translator-9-cpu.7z*', 'Saber-Translator-9-cpu.7z.003', 'Saber-Translator-9-gpu.7z.003'),
    ('models.z*', 'models.z03', 'README.txt'),
    ('saber-translator-*.zip*', 'saber-translator-browser-extension-v0.zip', 'models.zip'),
])
def test_upload_only_cleans_its_own_stale_assets(tmp_path, monkeypatch, pattern, old, other):
    current = old.replace('003', '001').replace('z03', 'zip').replace('v0', 'v1')
    (tmp_path / current).write_bytes(b'new')
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout=json.dumps({'isDraft': True, 'assets': [
            {'name': name} for name in [current, old, other]
        ]}))
    monkeypatch.setattr(release_assets.subprocess, 'run', run)
    release_assets.upload_assets('v9', tmp_path, pattern)
    assert commands[1][:3] == ['gh', 'release', 'upload']
    assert commands[2] == ['gh', 'release', 'delete-asset', 'v9', old, '--yes']
    assert len(commands) == 3


def test_upload_failure_does_not_delete_stale_assets(tmp_path, monkeypatch):
    (tmp_path / 'models.zip').write_bytes(b'new')
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        if command[2] == 'upload':
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(stdout=json.dumps({'isDraft': True, 'assets': [{'name': 'models.z02'}]}))
    monkeypatch.setattr(release_assets.subprocess, 'run', run)
    with pytest.raises(subprocess.CalledProcessError):
        release_assets.upload_assets('v9', tmp_path, 'models.z*')
    assert len(commands) == 2


def test_published_release_is_not_modified(tmp_path, monkeypatch):
    (tmp_path / 'models.zip').write_bytes(b'new')
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout='{"isDraft": false, "assets": []}')
    monkeypatch.setattr(release_assets.subprocess, 'run', run)
    with pytest.raises(ValueError, match='published'):
        release_assets.upload_assets('v9', tmp_path, 'models.z*')
    assert len(commands) == 1


def test_missing_local_assets_fail_before_contacting_release(tmp_path, monkeypatch):
    def run(*args, **kwargs):
        pytest.fail('No remote operation is allowed without local assets')
    monkeypatch.setattr(release_assets.subprocess, 'run', run)
    with pytest.raises(ValueError, match='No release assets'):
        release_assets.upload_assets('v9', tmp_path, 'models.z*')
