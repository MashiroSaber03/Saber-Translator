"""PySide6 desktop entrypoint and runtime orchestration."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import threading
import webbrowser

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QAction, QFont, QFontDatabase, QIcon
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon

from src.backend_v2.desktop.pet import PetWindow
from src.backend_v2.desktop.pet_state import PetStateMachine
from src.backend_v2.desktop.settings import DesktopSettings, DesktopSettingsStore
from src.backend_v2.desktop.task_client import TaskApiClient
from src.backend_v2.desktop.theme import WINDOW_STYLESHEET
from src.backend_v2.desktop.window import DesktopWindow
from src.backend_v2.launcher.entrypoint import (
    LauncherConfig,
    LauncherState,
    LauncherStatus,
    LauncherSupervisor,
)
from src.backend_v2.logging_config import (
    configure_product_handler,
    configure_backend_logging,
    product_stream_frame,
    set_backend_console_level,
)
from src.backend_v2.paths import (
    data_root_fingerprint,
    ensure_data_root,
    project_root,
    resolve_data_root,
)
from src.shared.user_logging import user_log


LOGGER = logging.getLogger("saber.desktop")


class DesktopLogBridge(QObject):
    line = Signal(str, str)


class DesktopLogHandler(logging.Handler):
    def __init__(self, bridge: DesktopLogBridge, *, level: str = "INFO") -> None:
        super().__init__()
        self._bridge = bridge
        self.set_log_level(level)

    def set_log_level(self, level: str) -> None:
        configure_product_handler(
            self,
            role="desktop",
            level=level,
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            formatted = self.format(record)
            self._bridge.line.emit(
                "DESKTOP",
                product_stream_frame(record, formatted) or formatted,
            )
        except Exception:
            self.handleError(record)


class DesktopController(QObject):
    launcher_status = Signal(object)
    launcher_output = Signal(str, str)
    launcher_finished = Signal(object)
    log_level_changed = Signal(str)

    def __init__(
        self,
        app: QApplication,
        *,
        data_root: Path,
        settings_store: DesktopSettingsStore,
        settings: DesktopSettings,
        native_icon_path: Path,
        brand_logo_path: Path,
    ) -> None:
        super().__init__()
        self.app = app
        self.app.setStyleSheet(WINDOW_STYLESHEET)
        self.data_root = data_root
        self.settings_store = settings_store
        self.settings = settings
        self._status = LauncherStatus(LauncherState.STOPPED, "后端未启动")
        self._supervisor: LauncherSupervisor | None = None
        self._supervisor_thread: threading.Thread | None = None
        self._restart_pending = False
        self._browser_auto_opened = False
        self._quitting = False
        self._queue_jobs: list[dict[str, object]] = []
        self._history_jobs: list[dict[str, object]] = []
        self._queue_paused = False
        self._worker_online = False
        self._executor_busy = False
        self._waiting_reason: str | None = None
        self._pet_state = PetStateMachine()

        self.window = DesktopWindow(
            settings,
            native_icon_path=native_icon_path,
            brand_logo_path=brand_logo_path,
            data_root=data_root,
        )
        pet_root = Path(__file__).resolve().parent / "assets" / "pet" / "saber_chan"
        self.pet = PetWindow(
            pet_root / "pet.json",
            fallback_logo=brand_logo_path,
            scale_percent=settings.pet_scale_percent,
            always_on_top=settings.pet_always_on_top,
        )
        self.tasks = TaskApiClient(self)
        self.tray = QSystemTrayIcon(QIcon(str(native_icon_path)), self)
        self._tray_available = QSystemTrayIcon.isSystemTrayAvailable()
        self.window.set_close_to_tray_enabled(self._tray_available)
        self._configure_tray()
        self._connect_signals()
        self.app.aboutToQuit.connect(self._prepare_app_exit)
        self._pet_timer = QTimer(self)
        self._pet_timer.setInterval(250)
        self._pet_timer.timeout.connect(self._update_pet_state)
        self._pet_timer.start()

    def show(self) -> None:
        self.window.show()
        if self._tray_available:
            self.tray.show()
        if self.settings.pet_enabled:
            self._set_pet_visible(True)
            QTimer.singleShot(250, self.pet.greet)

    def start_backend(self) -> None:
        if self._supervisor_thread is not None and self._supervisor_thread.is_alive():
            return
        host = "0.0.0.0" if self.settings.allow_lan else "127.0.0.1"
        supervisor = LauncherSupervisor(
            LauncherConfig(
                data_root=self.data_root,
                host=host,
                port=self.settings.port,
                log_level=self.settings.log_level,
                open_browser=False,
                resident_models=self.settings.resident_models,
            ),
            status_callback=self.launcher_status.emit,
            output_callback=self.launcher_output.emit,
        )
        self._supervisor = supervisor

        def run() -> None:
            error: BaseException | None = None
            try:
                supervisor.run()
            except BaseException as caught:
                error = caught
            finally:
                self.launcher_finished.emit(error)

        thread = threading.Thread(target=run, name="saber-launcher-supervisor", daemon=True)
        self._supervisor_thread = thread
        thread.start()

    def stop_backend(self) -> None:
        self._restart_pending = False
        if self._supervisor is not None:
            self._supervisor.request_stop()

    def request_stop_backend(self) -> None:
        active_count = sum(
            1
            for job in self._queue_jobs
            if job.get("status") == "running"
        )
        if active_count:
            answer = QMessageBox.question(
                self.window,
                "停止后端",
                f"停止后端会将 {active_count} 个正在执行的任务标记为已中断，"
                "稍后可从检查点继续。确定停止吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.stop_backend()

    def restart_backend(self) -> None:
        if self._supervisor_thread is None or not self._supervisor_thread.is_alive():
            self.start_backend()
            return
        self._restart_pending = True
        if self._supervisor is not None:
            self._supervisor.request_stop()

    def open_web(self) -> None:
        if self._status.state != LauncherState.RUNNING:
            self.window.show_notice("请先启动后端。")
            return
        self._open_web_url()

    def _open_web_url(self) -> None:
        url = f"http://127.0.0.1:{self.settings.port}/"
        webbrowser.open_new(url)
        LOGGER.debug("已请求打开浏览器：%s", url)

    def apply_settings(self, submitted: DesktopSettings) -> None:
        previous_log_level = self.settings.log_level
        resident_models_changed = (
            submitted.resident_models != self.settings.resident_models
        )
        restart_required = (
            submitted.port != self.settings.port
            or submitted.allow_lan != self.settings.allow_lan
            or submitted.log_level != self.settings.log_level
        )
        updated = submitted.updated(
            pet_screen_name=self.settings.pet_screen_name,
            pet_position_x=self.settings.pet_position_x,
            pet_position_y=self.settings.pet_position_y,
            window_width=self.window.width(),
            window_height=self.window.height(),
        )
        if not self._store_settings(updated):
            return
        if updated.log_level != previous_log_level:
            set_backend_console_level(updated.log_level)
            self.log_level_changed.emit(updated.log_level)
        self.pet.set_scale_percent(self.settings.pet_scale_percent)
        self.pet.set_always_on_top(self.settings.pet_always_on_top)
        self.window.settings.apply_pet_settings(self.settings)
        self._set_pet_visible(self.settings.pet_enabled)
        if restart_required and self._status.state in {
            LauncherState.STARTING,
            LauncherState.RUNNING,
            LauncherState.DEGRADED,
        }:
            self.window.settings.show_auto_save_status("已自动保存 · 正在重启后端")
            self.restart_backend()
        elif resident_models_changed and self._status.state in {
            LauncherState.STARTING,
            LauncherState.RUNNING,
            LauncherState.DEGRADED,
        }:
            self.window.settings.show_auto_save_status(
                "已自动保存 · 重启后端后生效"
            )
        else:
            self.window.settings.show_auto_save_status("已自动保存")

    def toggle_pet(self) -> None:
        enabled = not self.pet.isVisible()
        self.apply_settings(self.settings.updated(pet_enabled=enabled))

    def request_quit(self) -> None:
        if self._quitting:
            return
        if self._status.state in {
            LauncherState.STARTING,
            LauncherState.RUNNING,
            LauncherState.DEGRADED,
            LauncherState.STOPPING,
        }:
            answer = QMessageBox.question(
                self.window,
                "退出 Saber-Translator",
                "退出会停止 API、Worker 和正在运行的任务，确定继续吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._quitting = True
        self.stop_backend()
        if self._supervisor_thread is None or not self._supervisor_thread.is_alive():
            self._finish_quit()

    def _configure_tray(self) -> None:
        menu = QMenu(self.window)
        self._tray_menu = menu
        show_action = menu.addAction("显示控制中心")
        service_action = menu.addAction("启动 / 停止后端")
        open_action = menu.addAction("打开网页")
        pet_action = menu.addAction("显示 / 隐藏桌宠")
        menu.addSeparator()
        quit_action = QAction("退出", menu)
        menu.addAction(quit_action)
        show_action.triggered.connect(self._show_main)
        service_action.triggered.connect(self._toggle_service)
        open_action.triggered.connect(self.open_web)
        pet_action.triggered.connect(self.toggle_pet)
        quit_action.triggered.connect(self.request_quit)
        self.tray.setContextMenu(menu)
        self.tray.setToolTip("Saber-Translator")
        self.tray.activated.connect(
            lambda reason: self._show_main()
            if reason == QSystemTrayIcon.ActivationReason.DoubleClick
            else None
        )

    def _connect_signals(self) -> None:
        self.window.start_requested.connect(self.start_backend)
        self.window.stop_requested.connect(self.request_stop_backend)
        self.window.restart_requested.connect(self.restart_backend)
        self.window.open_web_requested.connect(self.open_web)
        self.window.settings_changed.connect(self.apply_settings)
        self.window.job_command_requested.connect(self.tasks.command)
        self.window.queue_pause_requested.connect(self.tasks.set_queue_paused)
        self.window.quit_requested.connect(self.request_quit)
        self.launcher_status.connect(self._on_launcher_status)
        self.launcher_output.connect(self.window.add_log)
        self.launcher_finished.connect(self._on_launcher_finished)
        self.tasks.jobs_updated.connect(self._on_jobs_updated)
        self.tasks.connected_changed.connect(self.window.set_task_connected)
        self.tasks.error.connect(self._on_task_error)
        self.pet.show_main_requested.connect(self._show_main)
        self.pet.start_stop_requested.connect(self._toggle_service)
        self.pet.open_web_requested.connect(self.open_web)
        self.pet.quit_requested.connect(self.request_quit)
        self.pet.hidden_requested.connect(self.toggle_pet)
        self.pet.scale_requested.connect(self._set_pet_scale)
        self.pet.always_on_top_requested.connect(self._set_pet_top)
        self.pet.position_changed.connect(self._save_pet_position)

    def _on_launcher_status(self, status: LauncherStatus) -> None:
        previous = self._status.state
        self._status = status
        self.window.set_launcher_status(status)
        self.window.settings.set_backend_running(status.state != LauncherState.STOPPED)
        if status.state == LauncherState.RUNNING and previous != LauncherState.RUNNING:
            self.tasks.start(f"http://127.0.0.1:{self.settings.port}")
            if self.settings.open_browser_on_start and not self._browser_auto_opened:
                self._browser_auto_opened = True
                self._open_web_url()
        elif status.state == LauncherState.STOPPED:
            self.tasks.stop()
            self._queue_jobs = []
            self._history_jobs = []
            self._worker_online = False
            self._executor_busy = False
            self._waiting_reason = None
            self.window.set_jobs(
                [],
                [],
                self._queue_paused,
                self._worker_online,
                self._executor_busy,
                self._waiting_reason,
            )
        self._update_pet_state()

    def _on_launcher_finished(self, error: object) -> None:
        self._supervisor_thread = None
        self._supervisor = None
        self.window.settings.set_backend_running(False)
        if error is not None and not self._quitting:
            self.window.show_error(f"后端启动或运行失败：{error}")
        if self._quitting:
            self._finish_quit()
            return
        if self._restart_pending:
            self._restart_pending = False
            QTimer.singleShot(0, self.start_backend)

    def _on_jobs_updated(
        self,
        queue: object,
        history: object,
        worker_online: bool,
        queue_paused: bool,
        executor_busy: bool,
        waiting_reason: object,
    ) -> None:
        self._queue_jobs = [dict(item) for item in queue if isinstance(item, dict)]
        self._history_jobs = [dict(item) for item in history if isinstance(item, dict)]
        self._queue_paused = queue_paused
        self._worker_online = worker_online
        self._executor_busy = executor_busy
        self._waiting_reason = (
            str(waiting_reason) if isinstance(waiting_reason, str) else None
        )
        self.window.set_jobs(
            self._queue_jobs,
            self._history_jobs,
            self._queue_paused,
            self._worker_online,
            self._executor_busy,
            self._waiting_reason,
        )
        self._update_pet_state()

    def _on_task_error(self, message: str) -> None:
        if message.startswith("任务操作失败"):
            LOGGER.warning("桌面任务中心：%s", message)
            self.window.show_error(message)
        else:
            LOGGER.debug("桌面任务中心：%s", message)

    def _update_pet_state(self) -> None:
        state = self._pet_state.update(
            self._status.state.value,
            self._queue_jobs,
            self._history_jobs,
        )
        self.pet.set_base_state(state)

    def _toggle_service(self) -> None:
        if self._supervisor_thread is not None and self._supervisor_thread.is_alive():
            self.request_stop_backend()
        else:
            self.start_backend()

    def _show_main(self) -> None:
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()

    def _set_pet_visible(self, enabled: bool) -> None:
        if not enabled:
            self.pet.hide()
            return
        if self.pet.isVisible():
            return
        self.pet.restore_position(
            self.settings.pet_screen_name,
            self.settings.pet_position_x,
            self.settings.pet_position_y,
        )
        self.pet.show()

    def _set_pet_scale(self, percent: int) -> None:
        self.apply_settings(self.settings.updated(pet_scale_percent=percent))

    def _set_pet_top(self, enabled: bool) -> None:
        self.apply_settings(self.settings.updated(pet_always_on_top=enabled))

    def _save_pet_position(self, screen_name: str, x: float, y: float) -> None:
        updated = self.settings.updated(
            pet_screen_name=screen_name,
            pet_position_x=x,
            pet_position_y=y,
        )
        self._store_settings(updated)

    def _store_settings(
        self,
        updated: DesktopSettings,
        *,
        show_error: bool = True,
    ) -> bool:
        try:
            self.settings_store.save(updated)
        except OSError as error:
            if show_error:
                self.window.settings.show_auto_save_status("自动保存失败")
                self.window.show_error(f"设置自动保存失败：{error}")
            else:
                LOGGER.error("退出时保存桌面设置失败：%s", error)
            return False
        self.settings = updated
        return True

    def _finish_quit(self) -> None:
        updated = self.settings.updated(
            window_width=max(920, self.window.width()),
            window_height=max(640, self.window.height()),
        )
        self._store_settings(updated, show_error=False)
        self.tasks.stop()
        self.tray.hide()
        self.pet.close()
        self.window.allow_close()
        self.app.quit()

    def _prepare_app_exit(self) -> None:
        """Best-effort cleanup for OS/session initiated application exits."""

        self.tasks.stop()
        self.stop_backend()


def _desktop_probe(data_root: Path, settings: DesktopSettings) -> dict[str, object]:
    return {
        "role": "desktop",
        "status": "ready",
        "dataRoot": str(data_root),
        "dataRootFingerprint": data_root_fingerprint(data_root),
        "port": settings.port,
        "petEnabled": settings.pet_enabled,
        "residentModels": list(settings.resident_models),
    }


def _configure_windows_app_identity() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("SaberTranslator.Desktop")
    except (AttributeError, OSError):
        LOGGER.debug("无法设置 Windows AppUserModelID", exc_info=True)


def run_desktop(args: object) -> int:
    data_root = ensure_data_root(resolve_data_root(args.data_dir))
    store = DesktopSettingsStore(data_root)
    defaults = DesktopSettings(
        port=args.port,
        open_browser_on_start=not args.no_browser,
    )
    settings = store.load(defaults)
    if args.probe:
        print(json.dumps(_desktop_probe(data_root, settings), sort_keys=True))
        return 0

    log_path = configure_backend_logging(
        role="launcher",
        data_root=data_root,
        console_level=settings.log_level,
    )
    _configure_windows_app_identity()
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Saber-Translator")
    app.setOrganizationName("SaberTranslator")
    app.setQuitOnLastWindowClosed(False)
    bundled_font = project_root() / "src" / "backend_v2" / "resources" / "fonts" / "msyh.ttc"
    font_id = QFontDatabase.addApplicationFont(str(bundled_font))
    if font_id >= 0:
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            app.setFont(QFont(families[0], 10))
    asset_root = Path(__file__).resolve().parent / "assets"
    native_icon_path = asset_root / ("app-icon.ico" if os.name == "nt" else "app-icon.png")
    brand_logo_path = asset_root / "app-icon.png"
    native_icon = QIcon(str(native_icon_path))
    if native_icon.isNull():
        raise RuntimeError(f"桌面应用图标不可用：{native_icon_path}")
    app.setWindowIcon(native_icon)
    controller = DesktopController(
        app,
        data_root=data_root,
        settings_store=store,
        settings=settings,
        native_icon_path=native_icon_path,
        brand_logo_path=brand_logo_path,
    )
    bridge = DesktopLogBridge()
    bridge.line.connect(controller.window.add_log)
    handler = DesktopLogHandler(bridge, level=settings.log_level)
    controller.log_level_changed.connect(handler.set_log_level)
    logging.getLogger().addHandler(handler)
    LOGGER.debug(
        "桌面控制中心运行参数：pid=%s data_root=%s log=%s",
        os.getpid(), data_root, log_path,
    )
    user_log("system", "桌面控制中心已启动")
    controller.show()
    try:
        return app.exec()
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
