from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

import pytest

from src.version import STORAGE_VERSION, parse_version
from src.backend_v2.storage.lifecycle import initialize_database, schema_smoke_test
from src.storage_migrator.contracts import StorageError, contract_sql, read_identity
from src.storage_migrator.control import DataRootLock, DataRootAlreadyLocked, business_ready, atomic_json, control_root, registered_process, wait_for_children
from src.storage_migrator.registry import Migration, migration_chain
from src.storage_migrator.runner import StorageManager, OWNER_FILE
from src.storage_migrator.paths import filesystem_path


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    root = tmp_path_factory.mktemp("version-baseline") / "data"
    initialize_database(root)
    # The migration protocol fixtures need existing files, not real font payloads.
    with closing(sqlite3.connect(root / "saber.sqlite3")) as db, db:
        # Freeze protocol fixtures at 3.5.1; later releases use the same schema.
        db.execute("UPDATE schema_metadata SET storage_version='3.5.1'")
        for (relative,) in db.execute("SELECT relative_path FROM fonts"):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"font fixture")
    return root


@pytest.fixture
def root(tmp_path, baseline):
    result = tmp_path / "data"
    shutil.copytree(baseline, result)
    (result / "objects").mkdir(exist_ok=True)
    (result / "objects" / "saved.txt").write_text("用户已保存的成果", encoding="utf-8")
    return result


def convert(root):
    with closing(sqlite3.connect(root / "saber.sqlite3")) as db, db:
        db.execute("ALTER TABLE app_settings ADD COLUMN migration_test TEXT")
    (root / "objects" / "saved.txt").write_text("已转换的成果", encoding="utf-8")


def sql_for(version):
    sql = contract_sql("3.5.1")
    if version != "3.5.1":
        sql += "\nALTER TABLE app_settings ADD COLUMN migration_test TEXT;"
    return sql


def manager(root, **kwargs):
    return StorageManager(root, "local", target="3.6.0", steps=(Migration("3.5.1", "3.6.0", convert),), sql_for=sql_for, **kwargs)


def test_version_numeric_and_registry():
    assert STORAGE_VERSION == "3.5.2"
    assert parse_version("3.5.10") > parse_version("3.5.9")
    for value in ("3.5", "3.05.0", "3.6.0-rc1", "v3.5.1"):
        with pytest.raises(ValueError):
            parse_version(value)
    with pytest.raises(StorageError, match="降级"):
        migration_chain("3.6.0", "3.5.1")
    with pytest.raises(StorageError, match="缺少"):
        migration_chain("3.5.1", "3.6.1", [Migration("3.5.1", "3.6.0")])


def test_current_check_and_readonly_cli(root):
    with closing(sqlite3.connect(root / "saber.sqlite3")) as db, db:
        db.execute("UPDATE schema_metadata SET storage_version=?", (STORAGE_VERSION,))
    assert StorageManager(root, "local").check()["status"] == "current"
    result = subprocess.run([sys.executable, "saber_v2.py", "--role", "storage-migrator", "--action", "check", "--data-dir", str(root)], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "current"
    assert not control_root(root).exists()


def test_empty_check_does_not_create_directories(tmp_path):
    root = tmp_path / "new"
    assert StorageManager(root, "local").check()["status"] == "new"
    assert not root.exists() and not control_root(root).exists()
    root.mkdir()
    (root / "launcher-settings.json").write_text("{}")
    with pytest.raises(StorageError, match="旧数据"):
        StorageManager(root, "local").check()
    assert not (root / "saber.sqlite3").exists()


def test_unversioned_database_is_never_stamped(root):
    with closing(sqlite3.connect(root / "saber.sqlite3")) as db, db:
        db.execute("ALTER TABLE schema_metadata DROP COLUMN storage_version")
    before = (root / "saber.sqlite3").read_bytes()
    with StorageManager(root, "local") as migration:
        with pytest.raises(StorageError, match="无版本"):
            migration.prepare(initialize_database)
    assert (root / "saber.sqlite3").read_bytes() == before


def test_profile_schema_and_json_rejected(root):
    with pytest.raises(StorageError, match="模式"):
        StorageManager(root, "public").check()
    with closing(sqlite3.connect(root / "saber.sqlite3")) as db, db:
        db.execute("UPDATE app_settings SET payload_json='broken'")
    with pytest.raises(StorageError, match="JSON"):
        StorageManager(root, "local").check()
    with closing(sqlite3.connect(root / "saber.sqlite3")) as db, db:
        db.execute("DROP INDEX ix_process_epochs_role_status")
    with pytest.raises(StorageError, match="结构不匹配"):
        schema_smoke_test(root / "saber.sqlite3")


def test_351_upgrade_to_current_preserves_data_without_backup(root):
    assert contract_sql("3.5.1") == contract_sql("3.5.2")
    with closing(sqlite3.connect(root / "saber.sqlite3")) as db:
        before = list(db.iterdump())
    saved = (root / "objects" / "saved.txt").read_bytes()
    with StorageManager(root, "local") as migration:
        assert migration.check()["status"] == "upgrade_required"
        assert migration.prepare(initialize_database)["status"] == "updated"
        assert read_identity(root)[0] == STORAGE_VERSION
        assert not list(migration._operations())
        assert business_ready(root)
        assert migration.check()["status"] == "current"
    with closing(sqlite3.connect(root / "saber.sqlite3")) as db:
        after = list(db.iterdump())
    assert [line for line in before if not line.startswith('INSERT INTO "schema_metadata"')] == [
        line for line in after if not line.startswith('INSERT INTO "schema_metadata"')
    ]
    assert (root / "objects" / "saved.txt").read_bytes() == saved


def test_noop_chain_preserves_files_and_does_not_backup(root):
    before = (root / "objects" / "saved.txt").read_bytes()
    steps = (Migration("3.5.1", "3.6.0"), Migration("3.6.0", "3.6.1"))
    with StorageManager(root, "local", target="3.6.1", steps=steps, sql_for=lambda _: contract_sql("3.5.1")) as migration:
        assert migration.prepare(initialize_database)["status"] == "updated"
        assert read_identity(root)[0] == "3.6.1"
        assert not list(migration._operations())
        assert business_ready(root)
    assert (root / "objects" / "saved.txt").read_bytes() == before


def test_noop_failure_keeps_version(root):
    def fail(_):
        raise StorageError("bad target")
    with StorageManager(root, "local", target="3.6.0", steps=(Migration("3.5.1", "3.6.0", validate=fail),), sql_for=lambda _: contract_sql("3.5.1")) as migration:
        with pytest.raises(StorageError, match="bad target"):
            migration.prepare(initialize_database)
    assert read_identity(root)[0] == "3.5.1"


def test_real_upgrade_commit_cleanup_and_user_backup(root):
    user_backup = root.parent / "my-backup"
    user_backup.mkdir()
    (user_backup / "keep").write_text("user backup")
    with manager(root) as migration:
        assert migration.prepare(initialize_database)["status"] == "awaiting_health"
        op, state = list(migration._operations())[0]
        assert read_identity(op / "backup")[0] == "3.5.1"
        assert read_identity(root)[0] == "3.6.0"
        assert not business_ready(root)
        with pytest.raises(StorageError, match="尚未开始"):
            migration.confirm_ready()
        assert (op / "backup").exists() and not business_ready(root)
        migration.before_start()
        assert not business_ready(root)
        assert migration.confirm_ready() == []
        assert business_ready(root)
        assert not (op / "backup").exists()
        assert not (root / OWNER_FILE).exists()
        assert (root / "objects" / "saved.txt").read_text(encoding="utf-8") == "已转换的成果"
        assert list(migration._operations())[0][1]["stage"] == "cleaned"
    assert (user_backup / "keep").exists()


class PowerLoss(BaseException):
    pass


@pytest.mark.parametrize("stage", ["copying", "converting", "validated", "old_moved", "installed", "starting", "committed"])
def test_recovery_at_persisted_boundaries(root, stage):
    def crash(current):
        if current == stage:
            raise PowerLoss()
    with pytest.raises(PowerLoss), manager(root, failpoint=crash) as migration:
        migration.prepare(initialize_database)
        migration.before_start()
        migration.confirm_ready()
    with manager(root) as recovered:
        if stage in {"installed", "committed"}:
            recovered.prepare(initialize_database)
            recovered.before_start()
            recovered.confirm_ready()
            assert read_identity(root)[0] == "3.6.0"
        else:
            with pytest.raises(StorageError, match="已恢复旧数据"):
                recovered.prepare(initialize_database)
            assert read_identity(root)[0] == "3.5.1"
            assert (root / "objects" / "saved.txt").read_text(encoding="utf-8") == "用户已保存的成果"


def test_startup_failure_rolls_back(root):
    with manager(root) as migration:
        migration.prepare(initialize_database)
        migration.before_start()
        migration.startup_failed()
        assert migration.rolled_back
        assert read_identity(root)[0] == "3.5.1"
        assert business_ready(root)


def test_cleanup_failure_persists_and_next_start_retries(root, monkeypatch):
    with manager(root) as migration:
        migration.prepare(initialize_database)
        migration.before_start()
        with monkeypatch.context() as patch:
            patch.setattr(migration, "_remove", lambda *_: (_ for _ in ()).throw(PermissionError("busy")))
            assert len(migration.confirm_ready()) == 1
        op, state = list(migration._operations())[0]
        assert state["stage"] == "committed" and state["cleanup_error"] == "busy"
        assert business_ready(root)
    with manager(root) as migration:
        migration.prepare(initialize_database)
        migration.before_start()
        assert migration.confirm_ready() == []
        assert not (op / "backup").exists()
        assert read_identity(root)[0] == "3.6.0"


def test_lock_outside_root_and_live_process_protection(root):
    with DataRootLock(root):
        with pytest.raises(DataRootAlreadyLocked):
            with DataRootLock(root):
                pass
    with registered_process(root, "worker"):
        with pytest.raises(StorageError, match="尚未退出"):
            wait_for_children(root, timeout=0)


def test_api_gate_allows_only_health(root):
    with closing(sqlite3.connect(root / "saber.sqlite3")) as db, db:
        db.execute("UPDATE schema_metadata SET storage_version=?", (STORAGE_VERSION,))
    from src.backend_v2.api.app import ApiSettings, create_api_app
    from src.backend_v2.runtime_identity import RuntimeIdentity
    from src.backend_v2.storage.database import create_sqlite_engine
    engine = create_sqlite_engine(root / "saber.sqlite3")
    app = create_api_app(ApiSettings(root, RuntimeIdentity("test", "token", True), engine))
    try:
        control_root(root).mkdir()
        atomic_json(control_root(root) / "upgrade-pending.json", {"id": "test"})
        with app.test_client() as client:
            health = client.get("/api/v2/health")
            assert health.status_code == 200 and health.json["storageVersion"] == STORAGE_VERSION
            assert client.get("/api/v2/jobs").status_code == 503
    finally:
        app.extensions["saber_v2_runtime"].close()
        engine.dispose()


def test_real_conversion_ends_unfinished_graph_preserves_completed_items(root):
    progress = {"jobStatus": "queued", "executionMode": "sequential", "pools": [],
                "totalItems": 2, "completedItems": 1, "failedItems": 0, "skippedItems": 0, "cancelledItems": 0}
    with closing(sqlite3.connect(root / "saber.sqlite3")) as db, db:
        db.execute("INSERT INTO jobs(id,kind,status,config_json,latest_progress_json) VALUES ('job','translation','queued','{}',?)", (json.dumps(progress),))
        db.execute("INSERT INTO job_items(id,job_id,ordinal,status,result_json) VALUES ('saved','job',1,'completed','{\"text\":\"saved translation\"}')")
        db.execute("INSERT INTO job_items(id,job_id,ordinal,status) VALUES ('pending','job',2,'pending')")
        db.execute("INSERT INTO job_steps(id,job_item_id,ordinal,kind,status,checkpoint_json) VALUES ('step','pending',1,'ocr','pending','{}')")
    with manager(root) as migration:
        migration.prepare(initialize_database)
        with closing(sqlite3.connect(root / "saber.sqlite3")) as db:
            status, encoded = db.execute("SELECT status,latest_progress_json FROM jobs WHERE id='job'").fetchone()
            assert status == "failed" and json.loads(encoded)["failedItems"] == 1
            assert db.execute("SELECT result_json FROM job_items WHERE id='saved'").fetchone()[0] == '{"text":"saved translation"}'
            assert db.execute("SELECT status,checkpoint_json FROM job_steps").fetchone() == ("failed", None)
            assert "STORAGE_UPGRADE_INTERRUPTED" in db.execute("SELECT error_json FROM job_steps").fetchone()[0]


@pytest.mark.parametrize("after_install", [False, True])
def test_crash_between_directory_rename_and_journal_write(root, monkeypatch, after_install):
    import src.storage_migrator.runner as runner
    original = runner.os.replace
    def rename(source, target):
        original(source, target)
        if (Path(source).name == "work" if after_install else Path(source) == root):
            raise PowerLoss()
    with pytest.raises(PowerLoss), manager(root) as migration, monkeypatch.context() as patch:
        patch.setattr(runner.os, "replace", rename)
        migration.prepare(initialize_database)
    with manager(root) as migration:
        with pytest.raises(StorageError, match="已恢复旧数据"):
            migration.prepare(initialize_database)
    assert read_identity(root)[0] == "3.5.1"


def test_insufficient_space_and_failed_target_leave_original(root, monkeypatch):
    import src.storage_migrator.runner as runner
    from types import SimpleNamespace
    with manager(root) as migration, monkeypatch.context() as patch:
        patch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
        with pytest.raises(StorageError, match="空间不足"):
            migration.prepare(initialize_database)
    assert read_identity(root)[0] == "3.5.1"
    with manager(root) as migration:
        migration.sql_for = lambda _: contract_sql("3.5.1")
        with pytest.raises(StorageError, match="结构不匹配"):
            migration.prepare(initialize_database)
    assert read_identity(root)[0] == "3.5.1"


def test_unknown_backup_marker_never_deleted(root):
    with manager(root) as migration:
        migration.prepare(initialize_database)
        op, _ = list(migration._operations())[0]
        (op / "backup" / OWNER_FILE).write_text('{}')
        migration.before_start()
        assert migration.confirm_ready()
        assert (op / "backup" / "saber.sqlite3").exists()
        assert read_identity(root)[0] == "3.6.0"


def test_business_process_imports_no_historical_steps():
    result = subprocess.run([sys.executable, "-c", "import src.backend_v2.storage.startup; import sys; assert 'src.storage_migrator.registry' not in sys.modules; assert 'src.storage_migrator.v3_5_0' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_launcher_setup_failure_restores_candidate(root, monkeypatch):
    from src.backend_v2.launcher import entrypoint
    with manager(root) as migration:
        migration.prepare(initialize_database)
        monkeypatch.setattr(entrypoint, "create_sqlite_engine", lambda _: (_ for _ in ()).throw(OSError("open failed")))
        supervisor = entrypoint.LauncherSupervisor(entrypoint.LauncherConfig(root, host="127.0.0.1", port=5000), storage_manager=migration)
        with pytest.raises(OSError, match="open failed"):
            supervisor.run()
        assert migration.rolled_back and read_identity(root)[0] == "3.5.1"


def test_real_launcher_confirms_candidate_and_cleans_backup(root):
    import socket
    import threading
    from src.backend_v2.launcher.entrypoint import LauncherConfig, LauncherState, LauncherSupervisor
    from src.storage_migrator.v3_5_0 import stop_unfinished
    with closing(sqlite3.connect(root / "saber.sqlite3")) as db, db:
        db.execute("UPDATE schema_metadata SET storage_version='3.4.9'")
        db.execute("ALTER TABLE app_settings ADD COLUMN old_fixture_column TEXT")
    def fixture_sql(version):
        return contract_sql(STORAGE_VERSION) + ("\nALTER TABLE app_settings ADD COLUMN old_fixture_column TEXT;" if version == "3.4.9" else "")
    def fixture_convert(work):
        with closing(sqlite3.connect(work / "saber.sqlite3")) as db, db:
            db.execute("ALTER TABLE app_settings DROP COLUMN old_fixture_column")
    ready = threading.Event()
    errors = []
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with StorageManager(root, "local", steps=(Migration("3.4.9", STORAGE_VERSION, fixture_convert),), sql_for=fixture_sql, preparers={"3.4.9": stop_unfinished}) as migration:
        supervisor = LauncherSupervisor(LauncherConfig(root, host="127.0.0.1", port=port, open_browser=False), storage_manager=migration,
            status_callback=lambda status: ready.set() if status.state == LauncherState.RUNNING else None)
        def run():
            try:
                supervisor.run()
            except BaseException as error:
                errors.append(error)
                ready.set()
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            assert ready.wait(40), "candidate did not become healthy"
            assert not errors, errors
            assert business_ready(root)
            op, state = list(migration._operations())[0]
            assert state["stage"] == "cleaned" and not (op / "backup").exists()
            assert read_identity(root)[0] == STORAGE_VERSION
        finally:
            supervisor.request_stop()
            thread.join(40)
            assert not thread.is_alive()


def test_installed_candidate_corruption_rolls_back_on_next_prepare(root):
    with manager(root) as migration:
        migration.prepare(initialize_database)
    with closing(sqlite3.connect(root / "saber.sqlite3")) as db, db:
        db.execute("DROP INDEX ix_process_epochs_role_status")
    with manager(root) as migration:
        with pytest.raises(StorageError, match="结构不匹配"):
            migration.prepare(initialize_database)
        assert migration.rolled_back
        assert read_identity(root)[0] == "3.5.1"
        schema_smoke_test(root / "saber.sqlite3")


def test_desktop_setup_error_before_backend_restores_installed_data(root):
    with manager(root) as migration:
        with pytest.raises(ValueError, match="desktop"), migration.startup_guard():
            migration.prepare(initialize_database)
            raise ValueError("desktop settings construction failed")
        assert read_identity(root)[0] == "3.5.1"


@pytest.mark.parametrize("filename", [OWNER_FILE, "upgrade-pending.json"])
def test_post_commit_marker_cleanup_denied_does_not_block_business(root, monkeypatch, filename):
    original = Path.unlink
    def denied(path, *args, **kwargs):
        if path.name == filename:
            raise PermissionError("Windows handle held")
        return original(path, *args, **kwargs)
    with manager(root) as migration:
        migration.prepare(initialize_database)
        migration.before_start()
        with monkeypatch.context() as patch:
            patch.setattr(Path, "unlink", denied)
            assert migration.confirm_ready()
            assert business_ready(root)
            migration.startup_failed()
            assert read_identity(root)[0] == "3.6.0"
        assert migration.confirm_ready() == []
        assert list(migration._operations())[0][1]["stage"] == "cleaned"


def test_lost_operation_journal_never_initializes_empty_root(root):
    def crash(stage):
        if stage == "old_moved":
            raise PowerLoss()
    with pytest.raises(PowerLoss), manager(root, failpoint=crash) as migration:
        migration.prepare(initialize_database)
    op = next((control_root(root) / "operations").iterdir())
    (op / "state.json").unlink()
    with manager(root) as migration:
        with pytest.raises(StorageError, match="记录缺失"):
            migration.prepare(initialize_database)
    assert not root.exists()
    assert read_identity(op / "backup")[0] == "3.5.1"


def test_orphan_pending_gate_never_initializes_empty_root(tmp_path):
    root = tmp_path / "absent"
    with StorageManager(root, "local") as migration:
        atomic_json(migration.pending, {"id": "unregistered"})
        with pytest.raises(StorageError, match="缺少对应恢复记录"):
            migration.prepare(initialize_database)
    assert not root.exists()


def test_pending_candidate_profile_mismatch_is_readonly(root):
    with manager(root) as migration:
        migration.prepare(initialize_database)
    with StorageManager(root, "public") as migration:
        with pytest.raises(StorageError, match="模式"):
            migration.prepare(initialize_database)
    assert read_identity(root) == ("3.6.0", "local")


def test_release_registers_complete_fixed_contracts():
    from src.storage_migrator.registry import MIGRATIONS, SOURCE_PREPARERS, VERSION_VALIDATORS
    versions = {"3.5.0", STORAGE_VERSION} | {step.source for step in MIGRATIONS} | {step.target for step in MIGRATIONS}
    for version in versions:
        assert parse_version(version) <= parse_version(STORAGE_VERSION)
        migration_chain(version, STORAGE_VERSION, MIGRATIONS)
        assert contract_sql(version)
        assert version in SOURCE_PREPARERS
        assert version in VERSION_VALIDATORS


def test_current_desktop_json_is_validated_before_loading_ui(root):
    path = root / "launcher-settings.json"
    path.write_text('{"server": {}}')
    with pytest.raises(StorageError, match="桌面设置"):
        StorageManager(root, "local").check()
    assert path.read_text() == '{"server": {}}'


def test_missing_ready_vector_collection_is_rejected(root):
    from src.storage_migrator.v3_5_0 import validate
    with closing(sqlite3.connect(root / "saber.sqlite3")) as db, db:
        book = db.execute("SELECT id FROM books LIMIT 1").fetchone()[0]
        db.execute("INSERT INTO vector_generations(id,book_id,generation,status,dependency_fingerprint) VALUES ('vector',?,1,'ready',?)", (book, '0'*64))
        with pytest.raises(StorageError, match="Chroma"):
            validate(db, root=root, files=True)


def test_real_upgrade_converges_rendering_page_projection(root):
    from src.storage_migrator.v3_5_0 import stop_unfinished
    with closing(sqlite3.connect(root / "saber.sqlite3")) as db, db:
        chapter = db.execute("SELECT id FROM chapters LIMIT 1").fetchone()[0]
        db.execute("INSERT INTO pages(id,chapter_id,ordinal,logical_source_path,render_status) VALUES ('rendering-page',?,1,'render-test.png','rendering')", (chapter,))
    stop_unfinished(root)
    with closing(sqlite3.connect(root / "saber.sqlite3")) as db:
        assert db.execute("SELECT render_status FROM pages WHERE id='rendering-page'").fetchone()[0] == 'render_failed'


@pytest.mark.skipif(sys.platform != 'win32', reason='Actual Windows file-sharing behavior')
def test_real_windows_open_backup_file_retries_cleanup(root):
    with manager(root) as migration:
        migration.prepare(initialize_database)
        migration.before_start()
        op, _ = list(migration._operations())[0]
        with (op / 'backup' / 'objects' / 'saved.txt').open('rb'):
            assert migration.confirm_ready()
            assert business_ready(root)
            assert list(migration._operations())[0][1]['stage'] == 'committed'
        assert migration.confirm_ready() == []
        assert not (op / 'backup').exists()


def test_mixed_chain_uses_one_copy_and_orders_version_markers(root):
    def second_step(work):
        assert read_identity(work)[0] == '3.6.0'
        convert(work)
    steps = (Migration('3.5.1', '3.6.0'), Migration('3.6.0', '3.6.1', second_step))
    with StorageManager(root, 'local', target='3.6.1', steps=steps,
                        sql_for=lambda v: sql_for('3.5.1' if v == '3.6.0' else v)) as migration:
        migration.prepare(initialize_database)
        operations = list(migration._operations())
        assert len(operations) == 1
        assert read_identity(operations[0][0] / 'backup')[0] == '3.5.1'
        assert read_identity(root)[0] == '3.6.1'


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows extended filesystem paths')
@pytest.mark.parametrize('rollback', [False, True])
def test_long_work_and_backup_paths_preserve_identity_and_can_be_removed(root, rollback):
    # The source is below MAX_PATH, but the migration's extra directories exceed it.
    name = '字' * (235 - len(str(root / 'objects'))) + '.bin'
    saved = root / 'objects' / name
    saved.write_bytes(b'old long-path content')
    with manager(root) as migration:
        migration.prepare(initialize_database)
        op, state = next(migration._operations())
        assert state['root'] == str(root.resolve())
        assert len(str(op / 'backup' / 'objects' / name)) > 260
        assert (op / 'backup' / 'objects' / name).read_bytes() == saved.read_bytes()
        migration.before_start()
        if rollback:
            migration.startup_failed()
            assert read_identity(root)[0] == '3.5.1'
            assert saved.read_bytes() == b'old long-path content'
            assert not (op / 'discarded').exists()
        else:
            assert migration.confirm_ready() == []
            assert read_identity(root)[0] == '3.6.0'
        assert not (op / 'backup').exists()
        assert not (op / 'work').exists()


@pytest.mark.parametrize('deep', [False, pytest.param(True, marks=pytest.mark.skipif(sys.platform != 'win32', reason='Windows long SQLite path'))])
def test_readonly_database_uri_handles_literal_uri_characters(tmp_path, deep):
    from src.storage_migrator.contracts import read_database
    path = tmp_path / '数据库 #100%.sqlite3'
    if deep:
        path = tmp_path / ('nested-' * 18) / ('nested-' * 18) / path.name
    filesystem_path(path.parent).mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(filesystem_path(path))) as db:
        db.execute('CREATE TABLE saved(value TEXT)')
    with closing(read_database(path)) as db:
        assert db.execute('SELECT count(*) FROM saved').fetchone() == (0,)
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            db.execute("INSERT INTO saved VALUES ('must not write')")


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows path spelling')
def test_extended_path_preserves_unc_and_is_idempotent():
    path = Path(r'\\server\share\folder')
    extended = filesystem_path(path)
    assert str(extended) == r'\\?\UNC\server\share\folder'
    assert filesystem_path(extended) == extended


@pytest.mark.parametrize('valid_identity', [True, False])
def test_recovery_of_marker_interrupted_before_atomic_replace(root, valid_identity):
    def crash(stage):
        if stage == 'copying':
            raise PowerLoss()
    with pytest.raises(PowerLoss), manager(root, failpoint=crash) as migration:
        migration.prepare(initialize_database)
    op, state = next(migration._operations())
    marker = op / 'work' / OWNER_FILE
    temporary = marker.with_suffix(marker.suffix + '.tmp')
    marker.replace(temporary)
    if not valid_identity:
        temporary.write_text(json.dumps({'id': state['id'], 'root': 'another-directory'}))
    with manager(root) as recovered:
        with pytest.raises(StorageError, match='已恢复旧数据'):
            recovered.prepare(initialize_database)
        assert read_identity(root)[0] == '3.5.1'
        assert (root / 'objects' / 'saved.txt').read_text(encoding='utf-8') == '用户已保存的成果'
        assert (op / 'work').exists() is not valid_identity
        if not valid_identity:
            assert temporary.exists()
