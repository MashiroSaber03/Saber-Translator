from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from src.backend_v2.desktop.entrypoint import DesktopController
from src.backend_v2.desktop.settings import DesktopSettings, DesktopSettingsStore
from src.backend_v2.desktop.task_client import TaskApiClient
from src.backend_v2.desktop.window import SettingsPage, TaskCenterPage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ICON = PROJECT_ROOT / "src" / "backend_v2" / "desktop" / "assets" / "app-icon.png"


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_task_command_fails_locally_when_backend_is_disconnected() -> None:
    _app()
    client = TaskApiClient()
    errors: list[str] = []
    results: list[tuple[str, str, bool]] = []
    client.error.connect(errors.append)
    client.command_finished.connect(
        lambda job_id, action, success: results.append((job_id, action, success))
    )

    client.command("job-1", "pause")

    assert errors == ["任务操作失败：后端尚未连接"]
    assert results == [("job-1", "pause", False)]


def test_settings_page_accepts_pet_menu_updates(tmp_path) -> None:
    _app()
    page = SettingsPage(DesktopSettings(), tmp_path)
    changes: list[DesktopSettings] = []
    page.settings_changed.connect(changes.append)
    updated = DesktopSettings(
        pet_enabled=False,
        pet_always_on_top=False,
        pet_scale_percent=150,
    )

    page.apply_pet_settings(updated)

    assert not page.pet_enabled.isChecked()
    assert not page.pet_top.isChecked()
    assert page.pet_scale.currentText() == "150%"
    assert changes == []


def test_settings_emit_immediately_without_a_save_button(tmp_path) -> None:
    _app()
    page = SettingsPage(DesktopSettings(), tmp_path)
    changes: list[DesktopSettings] = []
    page.settings_changed.connect(changes.append)

    page.pet_enabled.click()

    assert changes[-1].pet_enabled is False
    assert "保存设置" not in [button.text() for button in page.findChildren(QPushButton)]


def test_backend_startup_controls_lock_while_service_is_running(tmp_path) -> None:
    _app()
    page = SettingsPage(DesktopSettings(), tmp_path)

    page.set_backend_running(True)

    assert not page.port.isEnabled()
    assert not page.allow_lan.isEnabled()
    assert not page.log_level.isEnabled()
    assert page.open_browser.isEnabled()
    assert page.pet_enabled.isEnabled()


def test_controller_persists_and_applies_settings_as_the_control_changes(tmp_path) -> None:
    app = _app()
    store = DesktopSettingsStore(tmp_path)
    controller = DesktopController(
        app,
        data_root=tmp_path,
        settings_store=store,
        settings=DesktopSettings(),
        app_icon_path=APP_ICON,
    )

    controller.window.settings.port.setValue(5111)
    controller.window.settings.pet_scale.setCurrentText("125%")

    assert controller.settings.port == 5111
    assert controller.pet.scale_percent == 125
    assert store.load().port == 5111
    assert store.load().pet_scale_percent == 125
    assert controller.window.settings.auto_save_status.text() == "已自动保存"

    controller._pet_timer.stop()
    controller.tray.hide()
    controller.pet.close()
    controller.window.allow_close()
    controller.deleteLater()
    app.processEvents()


def test_controller_uses_one_rounded_logo_for_window_tray_and_sidebar(tmp_path) -> None:
    app = _app()
    controller = DesktopController(
        app,
        data_root=tmp_path,
        settings_store=DesktopSettingsStore(tmp_path),
        settings=DesktopSettings(),
        app_icon_path=APP_ICON,
    )

    expected_icon = QIcon(str(APP_ICON)).pixmap(64, 64).toImage()
    assert controller.window.windowIcon().pixmap(64, 64).toImage() == expected_icon
    assert controller.tray.icon().pixmap(64, 64).toImage() == expected_icon

    logo = controller.window.sidebar.findChild(QLabel, "brandLogo")
    assert logo is not None
    expected_logo = QPixmap(str(APP_ICON)).scaled(
        38,
        38,
        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
        Qt.TransformationMode.SmoothTransformation,
    )
    assert logo.pixmap().toImage() == expected_logo.toImage()

    controller._pet_timer.stop()
    controller.tray.hide()
    controller.pet.close()
    controller.window.allow_close()
    controller.deleteLater()
    app.processEvents()


def test_interrupted_task_offers_continue_and_cancel() -> None:
    _app()
    page = TaskCenterPage()

    actions = page._actions({"jobId": "job-1", "status": "interrupted"})

    assert [button.text() for button in actions.findChildren(QPushButton)] == ["继续", "取消"]
