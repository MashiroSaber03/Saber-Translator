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
from src.backend_v2.desktop.window import DesktopWindow
from src.backend_v2.launcher.entrypoint import (
    LauncherConfig,
    LauncherState,
    LauncherStatus,
    LauncherSupervisor,
)
from src.backend_v2.logging_config import SecretSafeFormatter, configure_backend_logging
from src.backend_v2.paths import (
    data_root_fingerprint,
    ensure_data_root,
    project_root,
    resolve_data_root,
)


LOGGER = logging.getLogger("saber.desktop")


class DesktopLogBridge(QObject):
    line = Signal(str, str)


class DesktopLogHandler(logging.Handler):
    def __init__(self, bridge: DesktopLogBridge) -> None:
        super().__init__(logging.DEBUG)
        self._bridge = bridge
        self.setFormatter(
            SecretSafeFormatter(
                "%(asctime)s [%(levelname)s] [LAUNCHER:%(process)d] %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._bridge.line.emit("LAUNCHER", self.format(record))
        except Exception:
            self.handleError(record)


class DesktopController(QObject):
    launcher_status = Signal(object)
    launcher_output = Signal(str, str)
    launcher_finished = Signal(object)

    def __init__(
        self,
        app: QApplication,
        *,
        data_root: Path,
        settings_store: DesktopSettingsStore,
        settings: DesktopSettings,
        app_icon_path: Path,
    ) -> None:
        super().__init__()
        self.app = app
        self.data_root = data_root
        self.settings_store = settings_store
        self.settings = settings
        self._status = LauncherStatus(LauncherState.STOPPED, "后端未启动")
        self._supervisor: LauncherSupervisor | None = None
        self._supervisor_thread: threading.Thread | None = None
        self._restart_pending = False
        self._quitting = False
        self._queue_jobs: list[dict[str, object]] = []
        self._history_jobs: list[dict[str, object]] = []
        self._pet_state = PetStateMachine()

        self.window = DesktopWindow(settings, app_icon_path=app_icon_path, data_root=data_root)
        pet_root = Path(__file__).resolve().parent / "assets" / "pet" / "saber_chan"
        self.pet = PetWindow(
            pet_root / "pet.json",
            fallback_logo=app_icon_path,
            scale_percent=settings.pet_scale_percent,
            always_on_top=settings.pet_always_on_top,
        )
        self.tasks = TaskApiClient(self)
        self.tray = QSystemTrayIcon(QIcon(str(app_icon_path)), self)
        self._configure_tray()
        self._connect_signals()
        self.app.aboutToQuit.connect(self._prepare_app_exit)
        self._pet_timer = QTimer(self)
        self._pet_timer.setInterval(250)
        self._pet_timer.timeout.connect(self._update_pet_state)
        self._pet_timer.start()

    def show(self) -> None:
        self.window.show()
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
                open_browser=self.settings.open_browser_on_start,
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
        if self._supervisor is not None:
            self._supervisor.request_stop()

    def restart_backend(self) -> None:
        if self._supervisor_thread is None or not self._supervisor_thread.is_alive():
            self.start_backend()
            return
        self._restart_pending = True
        self.stop_backend()

    def open_web(self) -> None:
        if self._status.state != LauncherState.RUNNING:
            self.window.show_notice("请先启动后端。")
            return
        webbrowser.open_new(f"http://127.0.0.1:{self.settings.port}/")

    def apply_settings(self, submitted: DesktopSettings) -> None:
        startup_changed = (
            submitted.port != self.settings.port
            or submitted.allow_lan != self.settings.allow_lan
            or submitted.log_level != self.settings.log_level
            or submitted.open_browser_on_start != self.settings.open_browser_on_start
        )
        updated = submitted.updated(
            pet_screen_name=self.settings.pet_screen_name,
            pet_position_x=self.settings.pet_position_x,
            pet_position_y=self.settings.pet_position_y,
            window_width=self.window.width(),
            window_height=self.window.height(),
        )
        try:
            self.settings_store.save(updated)
        except OSError as error:
            self.window.settings.show_auto_save_status("自动保存失败")
            self.window.show_error(f"设置自动保存失败：{error}")
            return
        self.settings = updated
        self.pet.set_scale_percent(self.settings.pet_scale_percent)
        self.pet.set_always_on_top(self.settings.pet_always_on_top)
        self._set_pet_visible(self.settings.pet_enabled)
        if startup_changed and self._status.state != LauncherState.STOPPED:
            self.window.settings.show_auto_save_status("已保存 · 下次启动生效")
        else:
            self.window.settings.show_auto_save_status("已自动保存")

    def toggle_pet(self) -> None:
        enabled = not self.pet.isVisible()
        self.settings = self.settings.updated(pet_enabled=enabled)
        self.settings_store.save(self.settings)
        self.window.settings.apply_pet_settings(self.settings)
        self._set_pet_visible(enabled)

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
        menu = QMenu()
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
        self.window.stop_requested.connect(self.stop_backend)
        self.window.restart_requested.connect(self.restart_backend)
        self.window.open_web_requested.connect(self.open_web)
        self.window.settings_changed.connect(self.apply_settings)
        self.window.job_command_requested.connect(self.tasks.command)
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
        elif status.state == LauncherState.STOPPED:
            self.tasks.stop()
            self._queue_jobs = []
            self._history_jobs = []
            self.window.set_jobs([], [])
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

    def _on_jobs_updated(self, queue: object, history: object, _worker_online: bool) -> None:
        self._queue_jobs = [dict(item) for item in queue if isinstance(item, dict)]
        self._history_jobs = [dict(item) for item in history if isinstance(item, dict)]
        self.window.set_jobs(self._queue_jobs, self._history_jobs)
        self._update_pet_state()

    def _on_task_error(self, message: str) -> None:
        LOGGER.warning("桌面任务中心：%s", message)
        if message.startswith("任务操作失败"):
            self.window.show_error(message)

    def _update_pet_state(self) -> None:
        state = self._pet_state.update(
            self._status.state.value,
            self._queue_jobs,
            self._history_jobs,
        )
        self.pet.set_base_state(state)

    def _toggle_service(self) -> None:
        if self._supervisor_thread is not None and self._supervisor_thread.is_alive():
            self.stop_backend()
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
        self.pet.show()
        self.pet.restore_position(
            self.settings.pet_screen_name,
            self.settings.pet_position_x,
            self.settings.pet_position_y,
        )

    def _set_pet_scale(self, percent: int) -> None:
        self.pet.set_scale_percent(percent)
        self.settings = self.settings.updated(pet_scale_percent=percent)
        self.settings_store.save(self.settings)
        self.window.settings.apply_pet_settings(self.settings)

    def _set_pet_top(self, enabled: bool) -> None:
        self.pet.set_always_on_top(enabled)
        self.settings = self.settings.updated(pet_always_on_top=enabled)
        self.settings_store.save(self.settings)
        self.window.settings.apply_pet_settings(self.settings)

    def _save_pet_position(self, screen_name: str, x: float, y: float) -> None:
        self.settings = self.settings.updated(
            pet_screen_name=screen_name,
            pet_position_x=x,
            pet_position_y=y,
        )
        self.settings_store.save(self.settings)

    def _finish_quit(self) -> None:
        self.settings = self.settings.updated(
            window_width=max(920, self.window.width()),
            window_height=max(640, self.window.height()),
        )
        self.settings_store.save(self.settings)
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
    LOGGER.info(
        "Saber-Translator 桌面控制中心启动：pid=%s，data_root=%s，日志=%s",
        os.getpid(),
        data_root,
        log_path,
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
    app_icon_path = Path(__file__).resolve().parent / "assets" / "app-icon.png"
    app.setWindowIcon(QIcon(str(app_icon_path)))
    controller = DesktopController(
        app,
        data_root=data_root,
        settings_store=store,
        settings=settings,
        app_icon_path=app_icon_path,
    )
    bridge = DesktopLogBridge()
    bridge.line.connect(controller.window.add_log)
    handler = DesktopLogHandler(bridge)
    logging.getLogger().addHandler(handler)
    controller.show()
    try:
        return app.exec()
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
