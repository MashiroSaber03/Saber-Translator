import os
from pathlib import Path
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QProgressDialog

from src.backend_v2.desktop.entrypoint import DesktopStoragePreparation
from src.backend_v2.desktop.settings import DesktopSettings, DesktopSettingsStore
from src.backend_v2.desktop.window import DesktopWindow, SettingsPage
from src.storage_migrator.control import DataRootAlreadyLocked, DataRootLock
from src.storage_migrator.runner import StorageManager


ASSETS = Path(__file__).resolve().parents[2] / "src/backend_v2/desktop/assets"


def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module", autouse=True)
def keep_application_alive():
    application = app()
    yield application


def wait_until(predicate):
    deadline = time.monotonic() + 15
    while not predicate():
        app().processEvents()
        assert time.monotonic() < deadline, "desktop preparation timed out"
        time.sleep(0.01)


def test_preparation_uses_main_window_without_loading_settings_or_hiding_on_close(tmp_path):
    application = app()
    root = tmp_path / "not-created"
    window = DesktopWindow(None, native_icon_path=ASSETS / "app-icon.ico",
                           brand_logo_path=ASSETS / "app-icon.png", data_root=root)
    window.show()
    application.processEvents()
    try:
        assert not root.exists()
        assert not window.findChildren(SettingsPage)
        assert not any(isinstance(w, QProgressDialog) for w in application.topLevelWidgets())
        assert not window.sidebar.isEnabled()
        window.close()
        assert window.isVisible()
        window.show_storage_preparation("正在复制数据")
        assert window.preparation_message.text() == "正在复制数据"
        identity = window.winId()
        window.initialize_pages(DesktopSettings(port=60359), root)
        assert window.winId() == identity
        assert window.sidebar.isEnabled()
        assert window.stack.currentWidget() is window.overview
        assert len(window.findChildren(SettingsPage)) == 1
    finally:
        window.allow_close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_failed_preparation_stays_in_main_window_and_allows_exit(tmp_path):
    application = app()
    window = DesktopWindow(None, native_icon_path=ASSETS / "app-icon.ico",
                           brand_logo_path=ASSETS / "app-icon.png", data_root=tmp_path)
    exits = []
    window.quit_requested.connect(lambda: exits.append(True))
    window.show()
    application.processEvents()
    try:
        window.show_storage_preparation("旧库没有版本号\n诊断目录：test", failed=True)
        assert window.preparation_progress.isHidden()
        assert not window.sidebar.isEnabled()
        assert not window.findChildren(SettingsPage)
        window.request_close_to_tray()
        assert exits == [True]
    finally:
        window.allow_close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_worker_does_not_block_gui_and_keeps_lock_until_released(tmp_path, monkeypatch):
    application = app()
    manager = StorageManager(tmp_path / "data", "local")
    started, proceed = threading.Event(), threading.Event()
    thread_ids = []
    original_prepare = manager.prepare

    def slow_prepare(initialize):
        thread_ids.append(threading.get_ident())
        started.set()
        assert proceed.wait(10)
        return original_prepare(initialize)

    monkeypatch.setattr(manager, "prepare", slow_prepare)
    worker = DesktopStoragePreparation(manager, DesktopSettings(port=60359))
    prepared = []
    worker.prepared.connect(lambda settings, result: prepared.append((settings, result)))
    worker.thread.start()
    try:
        assert started.wait(5)
        application.processEvents()
        assert thread_ids != [threading.get_ident()]
        assert not prepared
        with pytest.raises(DataRootAlreadyLocked):
            DataRootLock(manager.root).acquire()
        proceed.set()
        wait_until(lambda: bool(prepared))
        assert prepared[0][0].port == 60359
        assert prepared[0][1]["status"] == "created"
        with pytest.raises(DataRootAlreadyLocked):
            DataRootLock(manager.root).acquire()
    finally:
        proceed.set()
        worker.release()
        worker.thread.join(15)
    assert not worker.thread.is_alive()
    assert worker.error is None
    with DataRootLock(manager.root):
        pass


def test_failed_settings_load_unwinds_storage_guard_before_reporting(tmp_path, monkeypatch):
    app()
    manager = StorageManager(tmp_path / "data", "local")
    failures = []
    calls = []
    original_failure = manager.startup_failed

    def failed(*, include_installed=False):
        calls.append(include_installed)
        return original_failure(include_installed=include_installed)

    def bad_settings(*args):
        raise ValueError("invalid desktop settings")

    monkeypatch.setattr(manager, "startup_failed", failed)
    monkeypatch.setattr(DesktopSettingsStore, "load", bad_settings)
    worker = DesktopStoragePreparation(manager, DesktopSettings())
    worker.failed.connect(failures.append)
    worker.thread.start()
    try:
        wait_until(lambda: bool(failures))
    finally:
        worker.release()
        worker.thread.join(15)
    assert True in calls
    assert "invalid desktop settings" in failures[0]
    with DataRootLock(manager.root):
        pass


def test_gui_initialization_failure_is_delivered_to_storage_guard(tmp_path):
    app()
    manager = StorageManager(tmp_path / "data", "local")
    worker = DesktopStoragePreparation(manager, DesktopSettings())
    prepared, failures = [], []
    worker.prepared.connect(lambda *_: prepared.append(True))
    worker.failed.connect(failures.append)
    worker.thread.start()
    try:
        wait_until(lambda: bool(prepared))
        worker.release(ValueError("GUI initialization failed"))
        worker.release()  # Shutdown must not discard the pending startup error.
        wait_until(lambda: bool(failures))
    finally:
        worker.release()
        worker.thread.join(15)
    assert "GUI initialization failed" in failures[0]
    with DataRootLock(manager.root):
        pass
