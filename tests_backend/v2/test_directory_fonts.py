from contextlib import closing
from pathlib import Path
import hashlib
import shutil
import sqlite3
import uuid

from flask import Flask
from PIL import ImageFont
import pytest
from sqlalchemy import insert, select, update

from src.backend_v2.auth.constants import LOCAL_USER_ID
from src.backend_v2.auth.ownership import owner_scope
from src.backend_v2.rendering.fonts import resolve_font_path
from src.backend_v2.storage import font_files
from src.backend_v2.storage.assets import AssetQuotaExceeded, AssetStorageService
from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.defaults import DEFAULT_FONT_ID
from src.backend_v2.storage.lifecycle import initialize_database
from src.backend_v2.storage.platform_repositories import FontRepository, RevisionConflict
from src.backend_v2.storage.schema import assets, fonts, pages, platform_config
from src.backend_v2.storage.seeding import QUICK_WORKSPACE_CHAPTER_ID
from src.storage_migrator.contracts import contract_sql, read_identity
from src.storage_migrator.runner import StorageManager
from src.version import STORAGE_VERSION


@pytest.fixture
def directory_fonts(tmp_path, monkeypatch):
    source = next(f.path for f in font_files.bundled_font_files() if f.path.name == 'ALGER.TTF')
    bundled = tmp_path / 'bundle'
    bundled.mkdir()
    shutil.copyfile(source, bundled / 'Alpha.ttf')
    monkeypatch.setattr(font_files, 'resource_path', lambda _: str(bundled))
    root = tmp_path / 'data'
    initialize_database(root)
    engine = create_sqlite_engine(root / 'saber.sqlite3')
    try:
        yield root, engine, FontRepository(engine), AssetStorageService(root, engine), source.read_bytes()
    finally:
        engine.dispose()


def test_manual_files_refresh_without_restart_and_missing_file_is_explicit(directory_fonts):
    root, engine, catalog, storage, payload = directory_fonts
    assert catalog.list()[0]['isDefault']
    added = root / 'fonts/shared/手动字体.ttf'
    added.write_bytes(payload)
    item = next(f for f in catalog.list() if f['displayName'] == '手动字体')
    with engine.connect() as connection:
        path = resolve_font_path(connection, storage, item['id'])
        assert ImageFont.truetype(path, 16).getbbox('A') is not None
    added.unlink()
    assert item['id'] not in {f['id'] for f in catalog.list()}
    with engine.connect() as connection:
        with pytest.raises(LookupError, match='移除'):
            resolve_font_path(connection, storage, item['id'])
        assert connection.execute(select(fonts.c.id).where(fonts.c.id == item['id'])).scalar_one()
    added.write_bytes(payload)
    assert item in catalog.list()


def test_first_use_installation_can_race_without_overwriting_files(directory_fonts, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    root, _, _, _, payload = directory_fonts
    barrier = Barrier(4)
    copy = shutil.copyfile

    def copy_together(source, destination):
        result = copy(source, destination)
        barrier.wait(timeout=10)
        return result

    monkeypatch.setattr(font_files.shutil, 'copyfile', copy_together)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(font_files.prepare_font_directory, [root] * 4))
    assert (root / 'fonts/shared/Alpha.ttf').read_bytes() == payload
    assert not list((root / 'fonts').glob('.install-*'))


def test_upload_and_manual_private_files_are_isolated_and_not_assets(directory_fonts):
    root, engine, catalog, storage, payload = directory_fonts
    with owner_scope('alice'):
        alice, replay = catalog.upload(filename='same.ttf', payload=payload, display_name='Alice', idempotency_key='upload')
        assert not replay and alice['scope'] == 'private'
        assert catalog.upload(filename='same.ttf', payload=payload, display_name='Alice', idempotency_key='upload') == (alice, True)
        (root / 'fonts/users/alice/manual.ttf').write_bytes(payload)
        assert any(f['displayName'] == 'manual' for f in catalog.list())
    with owner_scope('bob'):
        bob, _ = catalog.upload(filename='same.ttf', payload=payload, display_name='Bob', idempotency_key='upload')
        assert alice['id'] != bob['id']
        assert alice['id'] not in {f['id'] for f in catalog.list()}
        with engine.connect() as connection:
            with pytest.raises(LookupError):
                resolve_font_path(connection, storage, alice['id'])
            assert Path(resolve_font_path(connection, storage, bob['id'])).read_bytes() == payload
        with pytest.raises(LookupError):
            catalog.delete(font_id=alice['id'], idempotency_key='foreign-delete')
    with engine.connect() as connection:
        assert not connection.execute(select(assets.c.id)).all()


@pytest.mark.parametrize('filename', ['../escape.ttf', '..\\escape.ttf', '/escape.ttf', 'C:\\escape.ttf', 'a.ttf:stream', 'nested/a.ttf'])
def test_upload_rejects_paths(directory_fonts, filename):
    _, _, catalog, _, payload = directory_fonts
    with pytest.raises(ValueError):
        catalog.upload(filename=filename, payload=payload, display_name='unsafe', idempotency_key='unsafe')


def test_same_name_never_overwrites_and_referenced_font_cannot_be_deleted(directory_fonts):
    root, engine, catalog, _, payload = directory_fonts
    item, _ = catalog.upload(filename='saved.ttf', payload=payload, display_name='Saved', idempotency_key='one')
    path = root / f'fonts/users/{LOCAL_USER_ID}/saved.ttf'
    with pytest.raises(RevisionConflict, match='同名'):
        catalog.upload(filename='saved.ttf', payload=b'other', display_name='Other', idempotency_key='two')
    assert path.read_bytes() == payload
    with engine.begin() as connection:
        connection.execute(insert(pages).values(id='font-page', chapter_id=QUICK_WORKSPACE_CHAPTER_ID, ordinal=1, logical_source_path='page.png', default_font_id=item['id']))
    with pytest.raises(RevisionConflict, match='referenced'):
        catalog.delete(font_id=item['id'], idempotency_key='delete')
    assert path.is_file()


def test_private_font_usage_remains_quota_limited(directory_fonts):
    _, engine, catalog, storage, payload = directory_fonts
    with engine.begin() as connection:
        connection.execute(update(platform_config).values(asset_quota_bytes=len(payload) + 1))
    app = Flask('font-quota')
    app.config['SABER_V2_PROFILE'] = 'public'
    with app.test_request_context('/'):
        catalog.upload(filename='one.ttf', payload=payload, display_name='One', idempotency_key='one')
        with pytest.raises(AssetQuotaExceeded):
            catalog.upload(filename='two.ttf', payload=payload, display_name='Two', idempotency_key='two')
        with pytest.raises(AssetQuotaExceeded):
            storage.publish_bytes(b'xx', extension='bin', mime_type='application/octet-stream')


@pytest.mark.parametrize('display_name', ['用户字体', '长文件名字体验收' * 10])
def test_350_upgrade_preserves_font_ids_page_references_and_bytes(tmp_path, display_name):
    root = tmp_path / 'old'
    root.mkdir()
    payload = next(f.path for f in font_files.bundled_font_files() if f.path.name == 'ALGER.TTF').read_bytes()
    relative = 'objects/old/custom.ttf'
    (root / relative).parent.mkdir(parents=True)
    (root / relative).write_bytes(payload)
    uploaded = str(uuid.uuid4())
    with closing(sqlite3.connect(root / 'saber.sqlite3')) as db, db:
        db.executescript(contract_sql('3.5.0'))
        db.execute("INSERT INTO schema_metadata(singleton_id,runtime_profile,storage_version) VALUES (1,'local','3.5.0')")
        db.execute("INSERT INTO fonts(id,kind,display_name,builtin_key) VALUES (?,'builtin','思源黑体','default')", (DEFAULT_FONT_ID,))
        db.execute("INSERT INTO assets(id,relative_path,mime_type,checksum,byte_size) VALUES ('old-font',?,'font/ttf',?,?)", (relative, hashlib.sha256(payload).hexdigest(), len(payload)))
        db.execute("INSERT INTO fonts(id,kind,display_name,asset_id) VALUES (?,'uploaded',?,'old-font')", (uploaded, display_name))
        db.execute("INSERT INTO books(id,title) VALUES ('book','book')")
        db.execute("INSERT INTO chapters(id,book_id,title,ordinal) VALUES ('chapter','book','chapter',1)")
        db.execute("INSERT INTO pages(id,chapter_id,ordinal,logical_source_path,default_font_id) VALUES ('page','chapter',1,'page.png',?)", (uploaded,))
    with StorageManager(root, 'local') as manager:
        assert manager.prepare(initialize_database)['status'] == 'awaiting_health'
        assert read_identity(root)[0] == STORAGE_VERSION
        with closing(sqlite3.connect(root / 'saber.sqlite3')) as db:
            path, owner = db.execute('SELECT relative_path,owner_user_id FROM fonts WHERE id=?', (uploaded,)).fetchone()
            resolved = font_files.font_path(root, path, owner)
            assert owner == LOCAL_USER_ID and resolved.read_bytes() == payload
            assert ImageFont.truetype(str(resolved), 20).getbbox('test') is not None
            assert (path, owner) in font_files.scan_font_files(root, owner)
            assert db.execute("SELECT default_font_id FROM pages WHERE id='page'").fetchone()[0] == uploaded
            assert not db.execute('SELECT id FROM assets').fetchall()
        assert not (root / relative).exists()
        manager.before_start()
        assert manager.confirm_ready() == []
