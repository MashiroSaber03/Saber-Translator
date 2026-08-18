from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtNetwork import QNetworkReply
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from src.backend_v2.desktop.entrypoint import DesktopController
from src.backend_v2.launcher.entrypoint import LauncherState, LauncherStatus
from src.backend_v2.desktop.settings import DesktopSettings, DesktopSettingsStore
from src.backend_v2.desktop.task_client import TaskApiClient
from src.backend_v2.desktop.theme import WINDOW_STYLESHEET
from src.backend_v2.desktop.window import (
    DesktopWindow,
    LogPage,
    OverviewPage,
    SettingsPage,
    TaskCenterPage,
    _progress_summary,
)


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


def test_task_command_rejects_an_empty_job_id() -> None:
    _app()
    client = TaskApiClient()

    with pytest.raises(ValueError, match="job id"):
        client.command("", "pause")


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
    assert [page.log_level.itemText(index) for index in range(page.log_level.count())] == [
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
    ]


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


def test_setting_change_during_explicit_stop_does_not_restart_backend(
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
    controller._status = LauncherStatus(LauncherState.STOPPING, "停止中")

    controller.apply_settings(controller.settings.updated(port=5113))

    assert controller.settings.port == 5113
    assert restarts == []
    assert controller.window.settings.auto_save_status.text() == "已自动保存"
    controller._pet_timer.stop()
    controller.tray.hide()
    controller.pet.close()
    controller.window.allow_close()
    controller.deleteLater()
    app.processEvents()


def test_explicit_stop_clears_a_pending_restart(tmp_path) -> None:
    app = _app()
    controller = DesktopController(
        app,
        data_root=tmp_path,
        settings_store=DesktopSettingsStore(tmp_path),
        settings=DesktopSettings(),
        native_icon_path=NATIVE_ICON,
        brand_logo_path=BRAND_LOGO,
    )
    controller._restart_pending = True

    controller.stop_backend()

    assert controller._restart_pending is False
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


def test_task_refresh_never_limits_the_live_queue() -> None:
    _app()
    client = TaskApiClient()
    requested_urls: list[str] = []

    class FinishedSignal:
        def connect(self, _callback) -> None:
            pass

    class PendingReply:
        finished = FinishedSignal()

    class RecordingManager:
        def get(self, request):
            requested_urls.append(request.url().toString())
            return PendingReply()

    client._manager = RecordingManager()
    client._base_url = "http://127.0.0.1:5000"
    client._running = True

    client.refresh()

    assert requested_urls == [
        "http://127.0.0.1:5000/api/v2/jobs?scope=queue",
        "http://127.0.0.1:5000/api/v2/jobs?scope=history&limit=200",
    ]
    client.stop()


def test_task_stop_tolerates_a_synchronous_sse_finished_signal() -> None:
    _app()
    client = TaskApiClient()

    class Reply:
        aborted = False
        deleted = 0

        def abort(self) -> None:
            self.aborted = True
            client._finish_sse(self)

        def deleteLater(self) -> None:
            self.deleted += 1

    reply = Reply()
    client._running = True
    client._sse_reply = reply

    client.stop()

    assert reply.aborted is True
    assert reply.deleted == 1
    assert client._sse_reply is None
    assert client._running is False


def test_task_json_requests_have_a_finite_timeout() -> None:
    _app()
    client = TaskApiClient()
    client._base_url = "http://127.0.0.1:5000"

    request = client._json_request("/api/v2/jobs?scope=queue")

    assert request.transferTimeout() == 15_000


def test_stale_task_command_reply_is_ignored_after_backend_restart() -> None:
    _app()
    client = TaskApiClient()
    results: list[tuple[str, str, bool]] = []
    client.command_finished.connect(
        lambda job_id, action, success: results.append((job_id, action, success))
    )

    class StaleReply:
        deleted = False

        def deleteLater(self) -> None:
            self.deleted = True

    reply = StaleReply()
    client._generation = 2

    client._finish_command(reply, "job-1", "pause", 1)

    assert reply.deleted is True
    assert results == []


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
    assert controller._tray_menu.parent() is controller.window

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


def test_overview_keeps_paused_task_visible() -> None:
    _app()
    page = OverviewPage()

    page.update_jobs(
        [
            {
                "jobId": "job-1",
                "kind": "translation",
                "status": "paused",
                "progress": {"totalItems": 10, "completedItems": 4},
            }
        ]
    )

    assert page.task_value.text() == "漫画翻译"
    assert page.current_title.text() == "漫画翻译"
    assert page.current_detail.text().startswith("已暂停 ·")
    assert page.current_progress.value() == 40


def test_degraded_service_offers_restart_instead_of_an_inert_start_button() -> None:
    _app()
    page = OverviewPage()

    page.update_status(LauncherStatus(LauncherState.DEGRADED, "服务异常"))

    assert not page.start_button.isEnabled()
    assert page.restart_button.isEnabled()


def test_progress_summary_treats_invalid_counts_as_empty() -> None:
    assert _progress_summary(
        {
            "totalItems": "10",
            "completedItems": True,
            "currentStep": {},
        }
    ) == (0, "等待调度")


def test_log_view_and_backing_buffer_share_the_same_bound() -> None:
    _app()
    page = LogPage()

    for index in range(5001):
        page.add_line("API", f"[INFO] line {index}")

    assert len(page._lines) == 5000
    assert page.output.document().blockCount() == 5000


def test_log_view_inherits_the_bundled_application_font() -> None:
    log_style = WINDOW_STYLESHEET.split("QPlainTextEdit {", 1)[1].split("}", 1)[0]

    assert "font-family" not in log_style
    assert "Fixedsys" not in WINDOW_STYLESHEET


def test_window_close_requests_quit_when_system_tray_is_unavailable(tmp_path: Path) -> None:
    app = _app()
    window = DesktopWindow(
        DesktopSettings(),
        native_icon_path=NATIVE_ICON,
        brand_logo_path=BRAND_LOGO,
        data_root=tmp_path,
    )
    requests: list[bool] = []
    window.quit_requested.connect(lambda: requests.append(True))
    window.set_close_to_tray_enabled(False)

    window.request_close_to_tray()

    assert requests == [True]
    window.allow_close()
    window.deleteLater()
    app.processEvents()
