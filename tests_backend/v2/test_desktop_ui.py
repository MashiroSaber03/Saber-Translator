from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtNetwork import QNetworkReply
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from src.backend_v2.desktop.entrypoint import DesktopController
from src.backend_v2.launcher.entrypoint import LauncherState, LauncherStatus
from src.backend_v2.desktop.settings import DesktopSettings, DesktopSettingsStore
from src.backend_v2.desktop.task_client import TaskApiClient
from src.backend_v2.desktop.window import SettingsPage, TaskCenterPage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT_ROOT / "src" / "backend_v2" / "desktop" / "assets"
NATIVE_ICON = ASSET_ROOT / ("app-icon.ico" if os.name == "nt" else "app-icon.png")
BRAND_LOGO = ASSET_ROOT / "app-icon.png"


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


def test_backend_startup_controls_remain_editable_while_service_is_running(
    tmp_path,
) -> None:
    _app()
    page = SettingsPage(DesktopSettings(), tmp_path)

    page.set_backend_running(True)

    assert page.port.isEnabled()
    assert page.allow_lan.isEnabled()
    assert page.log_level.isEnabled()
    assert page.open_browser.isEnabled()
    assert page.pet_enabled.isEnabled()


def test_running_backend_restarts_after_startup_setting_changes(
    tmp_path,
    monkeypatch,
) -> None:
    app = _app()
    controller = DesktopController(
        app,
        data_root=tmp_path,
        settings_store=DesktopSettingsStore(tmp_path),
        settings=DesktopSettings(),
        native_icon_path=NATIVE_ICON,
        brand_logo_path=BRAND_LOGO,
    )
    restarts: list[bool] = []
    monkeypatch.setattr(controller, "restart_backend", lambda: restarts.append(True))
    controller._status = LauncherStatus(LauncherState.RUNNING, "运行中")

    controller.apply_settings(controller.settings.updated(port=5112))

    assert controller.settings.port == 5112
    assert restarts == [True]
    assert controller.window.settings.auto_save_status.text() == (
        "已自动保存 · 正在重启后端"
    )
    controller._pet_timer.stop()
    controller.tray.hide()
    controller.pet.close()
    controller.window.allow_close()
    controller.deleteLater()
    app.processEvents()


def test_task_list_failure_releases_refresh_gate() -> None:
    _app()
    client = TaskApiClient()
    client._running = True
    client._base_url = "http://127.0.0.1:5000"
    client._refresh_inflight = True

    class FailedReply:
        def error(self):
            return QNetworkReply.NetworkError.ConnectionRefusedError

        def errorString(self) -> str:
            return "connection refused"

        def deleteLater(self) -> None:
            pass

    client._finish_list("queue", FailedReply(), client._generation)
    client._finish_list("history", FailedReply(), client._generation)

    assert client._refresh_inflight is False
    client.stop()


def test_controller_persists_and_applies_settings_as_the_control_changes(tmp_path) -> None:
    app = _app()
    store = DesktopSettingsStore(tmp_path)
    controller = DesktopController(
        app,
        data_root=tmp_path,
        settings_store=store,
        settings=DesktopSettings(),
        native_icon_path=NATIVE_ICON,
        brand_logo_path=BRAND_LOGO,
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
        native_icon_path=NATIVE_ICON,
        brand_logo_path=BRAND_LOGO,
    )

    expected_icon = QIcon(str(NATIVE_ICON)).pixmap(64, 64).toImage()
    assert controller.window.windowIcon().pixmap(64, 64).toImage() == expected_icon
    assert controller.tray.icon().pixmap(64, 64).toImage() == expected_icon

    logo = controller.window.sidebar.findChild(QLabel, "brandLogo")
    assert logo is not None
    expected_logo = QPixmap(str(BRAND_LOGO)).scaled(
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
