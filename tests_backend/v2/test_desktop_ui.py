from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtNetwork import QNetworkReply
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
)

from src.backend_v2.desktop.entrypoint import (
    DesktopController,
    DesktopLogBridge,
    DesktopLogHandler,
)
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
from src.shared.user_logging import (
    CATEGORY_FIELD,
    STREAM_FRAME_PREFIX,
    USER_LOG_MARKER,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT_ROOT / "src" / "backend_v2" / "desktop" / "assets"
NATIVE_ICON = ASSET_ROOT / ("app-icon.ico" if os.name == "nt" else "app-icon.png")
BRAND_LOGO = ASSET_ROOT / "app-icon.png"


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_desktop_log_bridge_respects_the_configured_level() -> None:
    bridge = DesktopLogBridge()
    handler = DesktopLogHandler(bridge, level="INFO")

    assert handler.level == logging.INFO

    handler.set_log_level("DEBUG")
    assert handler.level == logging.DEBUG

    handler.set_log_level("invalid")
    assert handler.level == logging.INFO


def test_desktop_log_handler_uses_the_desktop_source_and_role() -> None:
    bridge = DesktopLogBridge()
    lines: list[tuple[str, str]] = []
    bridge.line.connect(lambda source, line: lines.append((source, line)))
    handler = DesktopLogHandler(bridge, level="INFO")
    record = logging.LogRecord(
        "saber.user",
        logging.INFO,
        __file__,
        1,
        "桌面设置已应用",
        (),
        None,
    )
    setattr(record, USER_LOG_MARKER, True)
    setattr(record, CATEGORY_FIELD, "system")

    handler.handle(record)

    assert lines[0][0] == "DESKTOP"
    assert "[桌面窗口] [系统]" in lines[0][1]


def test_task_command_fails_locally_when_backend_is_disconnected() -> None:
    _app()
    client = TaskApiClient()
    errors: list[str] = []
    client.error.connect(errors.append)

    client.command("job-1", "pause")

    assert errors == ["任务操作失败：后端尚未连接"]


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


def test_settings_page_selects_the_default_75_percent_pet_scale(tmp_path) -> None:
    _app()
    page = SettingsPage(DesktopSettings(), tmp_path)

    assert page.pet_scale.currentText() == "75%"


def test_settings_emit_immediately_without_a_save_button(tmp_path) -> None:
    _app()
    page = SettingsPage(DesktopSettings(), tmp_path)
    changes: list[DesktopSettings] = []
    page.settings_changed.connect(changes.append)

    page.pet_enabled.click()

    assert changes[-1].pet_enabled is False
    assert "保存设置" not in [button.text() for button in page.findChildren(QPushButton)]


def test_browser_extension_controls_emit_and_regenerate_a_masked_token(tmp_path) -> None:
    app = _app()
    original_token = "browser-extension-token-0123456789abcdef"
    page = SettingsPage(
        DesktopSettings(browser_extension_token=original_token),
        tmp_path,
    )
    changes: list[DesktopSettings] = []
    page.settings_changed.connect(changes.append)

    assert page.browser_extension_token.isReadOnly()
    assert page.browser_extension_token.echoMode() == QLineEdit.EchoMode.Password
    page.browser_extension_enabled.click()
    assert changes[-1].browser_extension_enabled is True

    copy_button = next(
        button
        for button in page.findChildren(QPushButton)
        if button.text() == "复制令牌"
    )
    copy_button.click()
    assert app.clipboard().text() == original_token

    page._regenerate_browser_extension_token()
    regenerated = page.browser_extension_token.text()
    assert regenerated != original_token
    assert len(regenerated) >= 32
    assert changes[-1].browser_extension_token == regenerated


def test_settings_page_emits_resident_models_in_catalog_order(tmp_path) -> None:
    _app()
    page = SettingsPage(
        DesktopSettings(resident_models=("paddle_ocr",)),
        tmp_path,
    )
    changes: list[DesktopSettings] = []
    page.settings_changed.connect(changes.append)

    assert page.resident_model_switches["paddle_ocr"].isChecked()
    page.resident_model_switches["detector_yolo"].click()

    assert changes[-1].resident_models == ("detector_yolo", "paddle_ocr")


def test_port_field_keeps_native_stepper_buttons(tmp_path) -> None:
    _app()
    page = SettingsPage(DesktopSettings(), tmp_path)

    assert page.port.buttonSymbols() == QAbstractSpinBox.ButtonSymbols.UpDownArrows


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
    assert page.browser_extension_enabled.isEnabled()
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


def test_running_backend_saves_resident_models_for_the_next_restart(
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

    controller.apply_settings(
        controller.settings.updated(resident_models=("paddle_ocr",))
    )

    assert controller.settings.resident_models == ("paddle_ocr",)
    assert restarts == []
    assert controller.window.settings.auto_save_status.text() == (
        "已自动保存 · 重启后端后生效"
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


@pytest.mark.parametrize("auto_open", [True, False])
def test_desktop_auto_opens_browser_only_once_but_keeps_manual_open(
    tmp_path,
    monkeypatch,
    auto_open,
) -> None:
    app = _app()
    controller = DesktopController(
        app,
        data_root=tmp_path,
        settings_store=DesktopSettingsStore(tmp_path),
        settings=DesktopSettings(open_browser_on_start=auto_open),
        native_icon_path=NATIVE_ICON,
        brand_logo_path=BRAND_LOGO,
    )
    opened: list[str] = []
    task_starts: list[str] = []
    monkeypatch.setattr("src.backend_v2.desktop.entrypoint.webbrowser.open_new", opened.append)
    monkeypatch.setattr(controller.tasks, "start", task_starts.append)
    monkeypatch.setattr(controller.tasks, "stop", lambda: None)

    controller._on_launcher_status(
        LauncherStatus(LauncherState.RUNNING, "首次启动完成")
    )
    controller._on_launcher_status(
        LauncherStatus(LauncherState.DEGRADED, "正在恢复")
    )
    controller._on_launcher_status(
        LauncherStatus(LauncherState.RUNNING, "恢复完成")
    )
    controller._on_launcher_status(
        LauncherStatus(LauncherState.STOPPED, "已停止")
    )
    controller._on_launcher_status(
        LauncherStatus(LauncherState.RUNNING, "再次启动完成")
    )

    expected_automatic = ["http://127.0.0.1:5000/"] if auto_open else []
    assert opened == expected_automatic
    assert task_starts == ["http://127.0.0.1:5000"] * 3

    controller.open_web()

    assert opened == [*expected_automatic, "http://127.0.0.1:5000/"]
    controller._pet_timer.stop()
    controller.tray.hide()
    controller.pet.close()
    controller.window.allow_close()
    controller.deleteLater()
    app.processEvents()


def test_desktop_backend_supervisor_never_owns_browser_activation(
    tmp_path,
    monkeypatch,
) -> None:
    app = _app()
    controller = DesktopController(
        app,
        data_root=tmp_path,
        settings_store=DesktopSettingsStore(tmp_path),
        settings=DesktopSettings(
            open_browser_on_start=True,
            resident_models=("paddle_ocr",),
        ),
        native_icon_path=NATIVE_ICON,
        brand_logo_path=BRAND_LOGO,
    )
    captured_configs = []

    class FakeSupervisor:
        def __init__(self, config, **_kwargs) -> None:
            captured_configs.append(config)

        def run(self) -> None:
            return None

        def request_stop(self) -> None:
            return None

    monkeypatch.setattr(
        "src.backend_v2.desktop.entrypoint.LauncherSupervisor",
        FakeSupervisor,
    )

    controller.start_backend()
    assert controller._supervisor_thread is not None
    controller._supervisor_thread.join(timeout=2)

    assert not controller._supervisor_thread.is_alive()
    assert len(captured_configs) == 1
    assert captured_configs[0].open_browser is False
    assert captured_configs[0].resident_models == ("paddle_ocr",)
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

    client._finish_list(FailedReply(), client._generation)

    assert client._refresh_inflight is False
    client.stop()


def test_task_list_requires_and_emits_current_scheduler_contract() -> None:
    _app()
    client = TaskApiClient()
    updates: list[tuple[object, ...]] = []
    errors: list[str] = []
    client.jobs_updated.connect(lambda *values: updates.append(values))
    client.error.connect(errors.append)
    client._running = True
    client._refresh_inflight = True
    client._sse_reply = object()

    class Reply:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def error(self):
            return QNetworkReply.NetworkError.NoError

        def readAll(self) -> bytes:
            return json.dumps(self.payload).encode("utf-8")

        def deleteLater(self) -> None:
            pass

    payload = {
        "items": [],
        "queuePaused": False,
        "eventCursor": 7,
        "workerOnline": True,
        "executorBusy": True,
        "waitingReason": "executor_busy",
    }
    client._finish_list(Reply(payload), client._generation)
    assert updates == [([], [], True, False, True, "executor_busy")]
    assert errors == []

    client._refresh_inflight = True
    client._finish_list(
        Reply({key: value for key, value in payload.items() if key != "executorBusy"}),
        client._generation,
    )
    assert errors[-1] == "任务列表响应格式无效"
    client._sse_reply = None
    client.stop()


def test_task_refresh_reads_queue_and_history_in_one_snapshot() -> None:
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
        "http://127.0.0.1:5000/api/v2/jobs?scope=all&limit=200",
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

    class StaleReply:
        deleted = False

        def deleteLater(self) -> None:
            self.deleted = True

    reply = StaleReply()
    client._generation = 2

    client._finish_command(reply, 1)

    assert reply.deleted is True


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

    assert app.styleSheet() == WINDOW_STYLESHEET
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


def test_task_action_buttons_fit_their_dedicated_column() -> None:
    app = _app()
    page = TaskCenterPage()
    page.setStyleSheet(WINDOW_STYLESHEET)
    page.resize(920, 640)
    page.set_jobs(
        [
            {
                "jobId": "job-1",
                "kind": "translation",
                "status": "running",
                "createdAt": "2026-08-23T07:05:02",
                "progress": {"totalItems": 10, "completedItems": 2},
            }
        ],
        [],
        False,
        True,
        True,
        "executor_busy",
    )
    assert page.scheduler_state.text() == "执行器正忙"
    page.show()
    app.processEvents()

    table = page.tables[0]
    actions = table.cellWidget(0, 5)
    assert actions is not None
    buttons = actions.findChildren(QPushButton)
    assert [button.text() for button in buttons] == ["暂停", "取消"]
    assert table.columnWidth(5) == TaskCenterPage.ACTION_COLUMN_WIDTH
    assert actions.layout().minimumSize().width() <= table.columnWidth(5)
    for button in buttons:
        required_text_width = button.fontMetrics().horizontalAdvance(button.text()) + 20
        assert button.width() >= required_text_width
        assert button.height() >= button.minimumSizeHint().height()

    page.close()
    page.deleteLater()
    app.processEvents()


def test_task_tabs_leave_clear_space_above_the_table() -> None:
    app = _app()
    page = TaskCenterPage()
    page.setStyleSheet(WINDOW_STYLESHEET)
    page.resize(920, 640)
    page.show()
    app.processEvents()

    tab_bar = page.tabs.tabBar()
    table = page.tables[0]
    tab_bottom = tab_bar.mapTo(page, QPoint()).y() + tab_bar.height()
    table_top = table.mapTo(page, QPoint()).y()

    assert table_top - tab_bottom >= 6

    page.close()
    page.deleteLater()
    app.processEvents()


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
    page.category_filter.setCurrentText("全部内容")

    for index in range(5001):
        page.add_line("API", f"[INFO] line {index}")

    assert len(page._lines) == 5000
    assert page.output.document().blockCount() == 5000


def test_log_view_preserves_unicode_messages() -> None:
    _app()
    page = LogPage()

    page.add_line("WORKER", "[INFO] [系统] 本地模型加载完成：漫画文字识别")

    assert page.output.toPlainText() == "[INFO] [系统] 本地模型加载完成：漫画文字识别"


def test_log_view_appends_framed_stream_chunks_to_the_same_line_immediately() -> None:
    _app()
    page = LogPage()

    def frame(action: str, chunk: str, formatted: str) -> str:
        return STREAM_FRAME_PREFIX + json.dumps(
            {
                "action": action,
                "streamId": "stream-1",
                "chunk": chunk,
                "formatted": formatted,
                "level": "INFO",
                "category": "流式",
            },
            ensure_ascii=False,
        )

    page.add_line(
        "WORKER",
        frame(
            "start",
            "",
            "2026-08-25 12:00:00 [工作进程] [流式] 高质量翻译开始流式返回：",
        ),
    )
    page.add_line(
        "WORKER",
        frame(
            "chunk",
            "你好",
            "2026-08-25 12:00:01 [工作进程] [流式] 高质量翻译流式片段｜你好",
        ),
    )
    assert page.output.toPlainText().endswith("高质量翻译开始流式返回：你好")
    page.add_line(
        "WORKER",
        frame(
            "chunk",
            "，世界",
            "2026-08-25 12:00:02 [工作进程] [流式] 高质量翻译流式片段｜，世界",
        ),
    )
    assert page.output.toPlainText().endswith("高质量翻译开始流式返回：你好，世界")
    page.add_line(
        "WORKER",
        frame(
            "end",
            "",
            "2026-08-25 12:00:03 [工作进程] [流式] 高质量翻译流式接收完成",
        ),
    )

    lines = page.output.toPlainText().splitlines()
    assert len(lines) == 2
    assert lines[0].endswith("高质量翻译开始流式返回：你好，世界")
    assert lines[1].endswith("高质量翻译流式接收完成")


def test_stream_line_appears_when_a_later_chunk_matches_search() -> None:
    _app()
    page = LogPage()
    page.search.setText("世界")

    def frame(action: str, chunk: str, formatted: str) -> str:
        return STREAM_FRAME_PREFIX + json.dumps(
            {
                "action": action,
                "streamId": "stream-search",
                "chunk": chunk,
                "formatted": formatted,
                "level": "INFO",
                "category": "流式",
            },
            ensure_ascii=False,
        )

    page.add_line("WORKER", frame("start", "", "高质量翻译开始流式返回："))
    page.add_line("WORKER", frame("chunk", "你好，", "高质量翻译流式片段｜你好，"))
    assert page.output.toPlainText() == ""
    page.add_line("WORKER", frame("chunk", "世界", "高质量翻译流式片段｜世界"))

    assert page.output.toPlainText() == "高质量翻译开始流式返回：你好，世界"


def test_log_view_hides_unstructured_diagnostics_by_default() -> None:
    _app()
    page = LogPage()

    page.add_line("WORKER", "native runtime debug output")
    page.add_line("WORKER", "[INFO] [结果] OCR 识别完成")

    assert page.output.toPlainText() == "[INFO] [结果] OCR 识别完成"
    page.category_filter.setCurrentText("全部内容")
    assert "native runtime debug output" in page.output.toPlainText()


def test_log_view_filters_new_chinese_categories_and_levels() -> None:
    _app()
    page = LogPage()
    page.add_line(
        "WORKER",
        "2026-08-25 12:00:00 [工作进程 42] [结果] 第 1 页 OCR 结果",
    )
    page.add_line(
        "WORKER",
        "2026-08-25 12:00:01 [工作进程 42] [警告] 正在重试",
    )

    page.level_filter.setCurrentIndex(page.level_filter.findData("WARNING"))

    assert page.output.toPlainText().endswith("[警告] 正在重试")
    assert "OCR 结果" not in page.output.toPlainText()


def test_log_view_recognizes_detailed_product_logs_as_debug() -> None:
    _app()
    page = LogPage()
    page.add_line(
        "WORKER",
        "2026-08-25 12:00:00 [工作进程] [调试] [模型] 完整提示词",
    )

    page.level_filter.setCurrentIndex(page.level_filter.findData("DEBUG"))

    assert "完整提示词" in page.output.toPlainText()


def test_log_view_groups_launcher_with_desktop_and_treats_critical_as_error() -> None:
    _app()
    page = LogPage()
    page.add_line("LAUNCHER", "[CRITICAL] 子进程无法启动")

    page.source_filter.setCurrentIndex(page.source_filter.findData("DESKTOP"))
    page.level_filter.setCurrentIndex(page.level_filter.findData("ERROR"))
    page.category_filter.setCurrentText("全部内容")

    assert page.output.toPlainText() == "[CRITICAL] 子进程无法启动"


def test_log_auto_scroll_keeps_the_start_of_new_lines_visible() -> None:
    app = _app()
    page = LogPage()
    page.resize(400, 300)
    for index in range(40):
        page.add_line("WORKER", f"[INFO] {index} " + "很长的日志内容" * 100)
    page.show()
    app.processEvents()

    horizontal = page.output.horizontalScrollBar()
    vertical = page.output.verticalScrollBar()
    assert horizontal.value() == horizontal.minimum()
    assert vertical.value() == vertical.maximum()

    page.close()


def test_log_view_inherits_the_bundled_application_font() -> None:
    log_style = WINDOW_STYLESHEET.split("QPlainTextEdit {", 1)[1].split("}", 1)[0]

    assert "font-family" not in log_style
    assert "Fixedsys" not in WINDOW_STYLESHEET


def test_message_box_width_applies_to_text_without_stretching_the_icon() -> None:
    app = _app()
    box = QMessageBox(
        QMessageBox.Icon.Warning,
        "Saber-Translator",
        "设置自动保存失败：测试消息",
    )
    box.setStyleSheet(WINDOW_STYLESHEET)
    box.show()
    app.processEvents()

    labels = {label.objectName(): label for label in box.findChildren(QLabel)}
    assert labels["qt_msgbox_label"].minimumWidth() == 280
    assert labels["qt_msgboxex_icon_label"].minimumWidth() < 280

    box.close()
    box.deleteLater()
    app.processEvents()


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
