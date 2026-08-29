"""Modern pink-white desktop control center widgets."""

from __future__ import annotations

from collections import deque
import json
import re
from pathlib import Path
from typing import Iterable, Mapping

from PySide6.QtCore import QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QCloseEvent,
    QIcon,
    QMouseEvent,
    QPainter,
    QPixmap,
    QResizeEvent,
    QShowEvent,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.backend_v2.desktop.settings import DesktopSettings, LOG_LEVELS, PET_SCALES
from src.backend_v2.launcher.entrypoint import LauncherState, LauncherStatus
from src.backend_v2.local_models import LOCAL_MODEL_OPTIONS
from src.shared.user_logging import CATEGORY_LABELS, STREAM_FRAME_PREFIX, job_label


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
MAX_LOG_LINES = 5000
PRODUCT_LOG_CATEGORIES = frozenset(CATEGORY_LABELS.values())
LOG_CATEGORY_PATTERN = re.compile(
    r"\[(" + "|".join(map(re.escape, CATEGORY_LABELS.values())) + r")\]"
)
LOG_CATEGORY_FILTERS = {
    "工作日志": PRODUCT_LOG_CATEGORIES,
    "任务过程": frozenset({"任务", "步骤", "结果"}),
    "模型输出": frozenset({"模型", "流式"}),
    "警告错误": frozenset({"警告", "错误"}),
    "系统信息": frozenset({"系统"}),
    "全部内容": None,
}
STATUS_LABELS = {
    "queued": "排队中",
    "running": "运行中",
    "paused": "已暂停",
    "cancelled": "已取消",
    "completed": "已完成",
    "completed_with_errors": "部分完成",
    "failed": "失败",
    "interrupted": "已中断",
}
WAITING_REASON_LABELS = {
    "queue_paused": "队列已暂停",
    "worker_offline": "Worker 离线",
    "low_memory": "等待可用内存",
    "queue_blocked": "等待章节锁",
    "executor_busy": "执行器正忙",
}


def _label(text: str, object_name: str = "") -> QLabel:
    widget = QLabel(text)
    if object_name:
        widget.setObjectName(object_name)
    widget.setWordWrap(object_name == "muted")
    return widget


class ToggleSwitch(QAbstractButton):
    """Small, animation-free switch for immediate boolean settings."""

    def __init__(self, checked: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setFixedSize(38, 22)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self.isEnabled():
            track = QColor("#E4E1E3")
            knob = QColor("#F7F6F7")
        elif self.isChecked():
            track = QColor("#D1517F")
            knob = QColor("#FFFFFF")
        else:
            track = QColor("#CEC9CC")
            knob = QColor("#FFFFFF")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(track)
        painter.drawRoundedRect(QRectF(0, 1, 38, 20), 10, 10)
        knob_x = 20 if self.isChecked() else 2
        painter.setBrush(knob)
        painter.drawEllipse(QRectF(knob_x, 3, 16, 16))


class TitleBar(QFrame):
    def __init__(self, window: "DesktopWindow") -> None:
        super().__init__()
        self._window = window
        self.setObjectName("titleBar")
        self.setFixedHeight(58)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(26, 10, 12, 10)
        layout.setSpacing(4)
        self.title = _label("概览", "pageTitle")
        layout.addWidget(self.title)
        layout.addStretch()
        minimize = QPushButton("—")
        maximize = QPushButton("□")
        close = QPushButton("×")
        for button in (minimize, maximize, close):
            button.setObjectName("windowControl")
        close.setProperty("danger", True)
        minimize.clicked.connect(window.showMinimized)
        maximize.clicked.connect(window.toggle_maximized)
        close.clicked.connect(window.request_close_to_tray)
        layout.addWidget(minimize)
        layout.addWidget(maximize)
        layout.addWidget(close)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self._window.windowHandle()
            if handle is not None:
                handle.startSystemMove()
            event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._window.toggle_maximized()
            event.accept()


class ResizeZone(QWidget):
    def __init__(
        self,
        owner: "DesktopWindow",
        edges: Qt.Edge,
        cursor: Qt.CursorShape,
    ) -> None:
        super().__init__(owner)
        self._owner = owner
        self._edges = edges
        self.setCursor(cursor)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and not self._owner.isMaximized():
            handle = self._owner.windowHandle()
            if handle is not None:
                handle.startSystemResize(self._edges)
            event.accept()


class Sidebar(QFrame):
    page_selected = Signal(int, str)

    def __init__(self, brand_logo_path: Path) -> None:
        super().__init__()
        self.setObjectName("sidebar")
        self.setFixedWidth(184)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 20, 16, 16)
        layout.setSpacing(7)
        brand = QHBoxLayout()
        logo = QLabel()
        logo.setObjectName("brandLogo")
        pixmap = QPixmap(str(brand_logo_path))
        logo.setPixmap(
            pixmap.scaled(
                38,
                38,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        logo.setFixedSize(38, 38)
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand_text = QVBoxLayout()
        brand_text.setSpacing(0)
        brand_text.addWidget(_label("Saber", "brandTitle"))
        brand_text.addWidget(_label("TRANSLATOR", "brandSubtitle"))
        brand.addWidget(logo)
        brand.addLayout(brand_text)
        layout.addLayout(brand)
        layout.addSpacing(20)
        self.buttons: list[QPushButton] = []
        for index, title in enumerate(("概览", "任务中心", "运行日志", "设置")):
            button = QPushButton(title)
            button.setObjectName("navButton")
            button.setProperty("active", index == 0)
            button.clicked.connect(
                lambda _checked=False, index=index, title=title: self.select(index, title)
            )
            self.buttons.append(button)
            layout.addWidget(button)
        layout.addStretch()
        version = _label("Saber Desktop", "muted")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version)

    def select(self, index: int, title: str) -> None:
        for current, button in enumerate(self.buttons):
            button.setProperty("active", current == index)
            button.style().unpolish(button)
            button.style().polish(button)
        self.page_selected.emit(index, title)


class OverviewPage(QWidget):
    start_requested = Signal()
    stop_requested = Signal()
    restart_requested = Signal()
    open_web_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("page")
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 22, 28, 28)
        root.setSpacing(16)
        hero = QFrame()
        hero.setObjectName("card")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(24, 22, 24, 22)
        hero_layout.setSpacing(20)
        intro = QVBoxLayout()
        intro.setSpacing(5)
        intro.addWidget(_label("DESKTOP CONTROL CENTER", "eyebrow"))
        intro.addWidget(_label("让翻译工作安静地运行", "heroTitle"))
        self.hero_message = _label("后端尚未启动，桌宠会陪你一起等待。", "muted")
        intro.addWidget(self.hero_message)
        hero_layout.addLayout(intro, 1)
        controls = QHBoxLayout()
        controls.setSpacing(8)
        self.start_button = QPushButton("启动后端")
        self.start_button.setObjectName("primaryButton")
        self.stop_button = QPushButton("停止后端")
        self.restart_button = QPushButton("重启")
        self.open_button = QPushButton("打开网页")
        self.start_button.clicked.connect(self.start_requested)
        self.stop_button.clicked.connect(self.stop_requested)
        self.restart_button.clicked.connect(self.restart_requested)
        self.open_button.clicked.connect(self.open_web_requested)
        for button in (self.start_button, self.stop_button, self.restart_button, self.open_button):
            button.setMinimumWidth(82)
            controls.addWidget(button)
        hero_layout.addLayout(controls)
        root.addWidget(hero)

        cards = QGridLayout()
        cards.setHorizontalSpacing(14)
        cards.setVerticalSpacing(14)
        self.service_value, service_card = self._status_card("服务状态", "未启动")
        self.api_value, api_card = self._status_card("API 进程", "—")
        self.worker_value, worker_card = self._status_card("Worker 进程", "—")
        self.task_value, task_card = self._status_card("当前任务", "无")
        cards.addWidget(service_card, 0, 0)
        cards.addWidget(api_card, 0, 1)
        cards.addWidget(worker_card, 0, 2)
        cards.addWidget(task_card, 0, 3)
        root.addLayout(cards)

        current = QFrame()
        current.setObjectName("card")
        current_layout = QVBoxLayout(current)
        current_layout.setContentsMargins(22, 20, 22, 20)
        current_layout.setSpacing(11)
        header = QHBoxLayout()
        header.addWidget(_label("当前工作", "sectionTitle"))
        header.addStretch()
        self.connection_pill = _label("事件流未连接", "statusPill")
        header.addWidget(self.connection_pill)
        current_layout.addLayout(header)
        self.current_title = _label("等待任务", "statusValue")
        self.current_detail = _label("启动后端后，任务中心会在这里显示实时进度。", "muted")
        self.current_progress = QProgressBar()
        self.current_progress.setRange(0, 100)
        self.current_progress.setValue(0)
        current_layout.addWidget(self.current_title)
        current_layout.addWidget(self.current_detail)
        current_layout.addWidget(self.current_progress)
        root.addWidget(current)
        root.addStretch()
        self.update_status(LauncherStatus(LauncherState.STOPPED, "后端未启动"))

    @staticmethod
    def _status_card(title: str, value: str) -> tuple[QLabel, QFrame]:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(7)
        layout.addWidget(_label(title, "muted"))
        value_label = _label(value, "statusValue")
        layout.addWidget(value_label)
        return value_label, card

    def update_status(self, status: LauncherStatus) -> None:
        labels = {
            LauncherState.STOPPED: "未启动",
            LauncherState.STARTING: "启动中",
            LauncherState.RUNNING: "运行正常",
            LauncherState.DEGRADED: "需要注意",
            LauncherState.STOPPING: "停止中",
        }
        self.service_value.setText(labels[status.state])
        self.api_value.setText(str(status.api_pid or "—"))
        self.worker_value.setText(str(status.worker_pid or "—"))
        self.hero_message.setText(status.message)
        stopped = status.state == LauncherState.STOPPED
        running = status.state == LauncherState.RUNNING
        degraded = status.state == LauncherState.DEGRADED
        self.start_button.setEnabled(stopped)
        self.stop_button.setEnabled(
            status.state
            in {LauncherState.STARTING, LauncherState.RUNNING, LauncherState.DEGRADED}
        )
        self.restart_button.setEnabled(running or degraded)
        self.open_button.setEnabled(running)

    def update_jobs(self, queue: list[Mapping[str, object]]) -> None:
        active = next((job for job in queue if job.get("status") == "running"), None)
        if active is None:
            active = next((job for job in queue if job.get("status") != "queued"), None)
        if active is None:
            queued = sum(1 for job in queue if job.get("status") == "queued")
            self.task_value.setText("无" if queued == 0 else f"排队 {queued}")
            self.current_title.setText("等待任务" if queued == 0 else "任务正在排队")
            self.current_detail.setText(
                "当前没有正在运行的任务。"
                if queued == 0
                else f"已有 {queued} 个任务等待调度。"
            )
            self.current_progress.setValue(0)
            return
        kind = str(active.get("kind") or "")
        status = str(active.get("status") or "")
        progress = active.get("progress")
        percent, detail = _progress_summary(progress)
        self.task_value.setText(job_label(kind))
        self.current_title.setText(job_label(kind))
        self.current_detail.setText(f"{STATUS_LABELS.get(status, status)} · {detail}")
        self.current_progress.setValue(percent)

    def set_connected(self, connected: bool) -> None:
        self.connection_pill.setText("事件流已连接" if connected else "事件流未连接")


def _progress_summary(progress: object) -> tuple[int, str]:
    if not isinstance(progress, Mapping):
        return 0, "等待进度数据"
    counts = []
    for key in (
        "totalItems",
        "completedItems",
        "failedItems",
        "skippedItems",
        "cancelledItems",
    ):
        value = progress.get(key)
        counts.append(
            value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0
        )
    total, completed, failed, skipped, cancelled = counts
    done = completed + failed + skipped + cancelled
    percent = min(100, round(done * 100 / total)) if total else 0
    current = progress.get("currentStep")
    current_kind = current.get("kind") if isinstance(current, Mapping) else None
    step = current_kind if isinstance(current_kind, str) and current_kind else "等待调度"
    return percent, f"{step} · {done} / {total}" if total else step


class TaskCenterPage(QWidget):
    command_requested = Signal(str, str)
    queue_pause_requested = Signal(bool)

    ACTION_COLUMN_WIDTH = 164

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("page")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 22, 28, 28)
        layout.setSpacing(14)
        header = QHBoxLayout()
        header.addWidget(_label("集中查看并控制后台任务", "sectionTitle"))
        header.addStretch()
        self._queue_paused = False
        self.queue_button = QPushButton("暂停队列")
        self.queue_button.setObjectName("compactButton")
        self.queue_button.setEnabled(False)
        self.queue_button.clicked.connect(self._toggle_queue_pause)
        header.addWidget(self.queue_button)
        self.scheduler_state = _label("调度状态未知", "statusPill")
        header.addWidget(self.scheduler_state)
        self.connection = _label("离线", "statusPill")
        header.addWidget(self.connection)
        layout.addLayout(header)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("taskTabs")
        self.tables = [self._create_table() for _ in range(3)]
        for title, table in zip(("当前", "排队中", "最近完成"), self.tables, strict=True):
            self.tabs.addTab(table, title)
        layout.addWidget(self.tabs, 1)

    @staticmethod
    def _create_table() -> QTableWidget:
        table = QTableWidget(0, 6)
        table.setHorizontalHeaderLabels(("任务", "状态", "当前步骤", "进度", "创建时间", "操作"))
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setShowGrid(False)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        table.setWordWrap(False)
        table.setTextElideMode(Qt.TextElideMode.ElideRight)
        header = table.horizontalHeader()
        header.setMinimumSectionSize(64)
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        table.setColumnWidth(0, 108)
        table.setColumnWidth(1, 82)
        table.setColumnWidth(3, 72)
        table.setColumnWidth(4, 164)
        table.setColumnWidth(5, TaskCenterPage.ACTION_COLUMN_WIDTH)
        return table

    def set_connected(self, connected: bool) -> None:
        self.connection.setText("实时连接" if connected else "等待连接")
        self.queue_button.setEnabled(connected)

    def _toggle_queue_pause(self) -> None:
        self.queue_pause_requested.emit(not self._queue_paused)

    def set_jobs(
        self,
        queue: Iterable[Mapping[str, object]],
        history: Iterable[Mapping[str, object]],
        queue_paused: bool,
        worker_online: bool,
        executor_busy: bool,
        waiting_reason: str | None,
    ) -> None:
        self._queue_paused = queue_paused
        self.queue_button.setText("恢复队列" if queue_paused else "暂停队列")
        if waiting_reason is not None:
            scheduler_label = WAITING_REASON_LABELS[waiting_reason]
        elif not worker_online:
            scheduler_label = "Worker 离线"
        elif executor_busy:
            scheduler_label = "执行器正忙"
        else:
            scheduler_label = "调度就绪"
        self.scheduler_state.setText(scheduler_label)
        queue_list = list(queue)
        active = [job for job in queue_list if job.get("status") != "queued"]
        waiting = [job for job in queue_list if job.get("status") == "queued"]
        self._populate(self.tables[0], active)
        self._populate(self.tables[1], waiting)
        self._populate(self.tables[2], list(history))
        self.tabs.setTabText(0, f"当前 {len(active)}")
        self.tabs.setTabText(1, f"排队中 {len(waiting)}")

    def _populate(self, table: QTableWidget, jobs: list[Mapping[str, object]]) -> None:
        table.setRowCount(len(jobs))
        for row, job in enumerate(jobs):
            kind = str(job.get("kind") or "")
            status = str(job.get("status") or "")
            progress = job.get("progress")
            percent, detail = _progress_summary(progress)
            values = (
                job_label(kind),
                STATUS_LABELS.get(status, status),
                detail,
                f"{percent}%",
                str(job.get("createdAt") or "—").replace("T", " ")[:19],
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column in {1, 3}:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row, column, item)
            table.setCellWidget(row, 5, self._actions(job))
            table.setRowHeight(row, 58)

    def _actions(self, job: Mapping[str, object]) -> QWidget:
        wrapper = QWidget()
        wrapper.setObjectName("actionCell")
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(6, 5, 6, 5)
        layout.setSpacing(6)
        status = str(job.get("status") or "")
        job_id = str(job.get("jobId") or "")
        actions: list[tuple[str, str]] = []
        if status == "running":
            actions.append(("暂停", "pause"))
        elif status == "paused":
            actions.append(("恢复", "resume"))
        elif status == "interrupted":
            actions.append(("继续", "continue"))
        if status in {"queued", "running", "paused", "interrupted"}:
            actions.append(("取消", "cancel"))
        layout.addStretch()
        for label, action in actions:
            button = QPushButton(label)
            button.setObjectName("compactButton")
            button.setMinimumWidth(50)
            button.clicked.connect(
                lambda _checked=False, job_id=job_id, action=action: self.command_requested.emit(
                    job_id,
                    action,
                )
            )
            layout.addWidget(button)
        layout.addStretch()
        return wrapper


class LogPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("page")
        self._lines: deque[tuple[str, str, str, str, str | None]] = deque(
            maxlen=MAX_LOG_LINES
        )
        self._tail_stream: tuple[str, str] | None = None
        self._stream_headers: dict[tuple[str, str], str] = {}
        self._auto_scroll_timer = QTimer(self)
        self._auto_scroll_timer.setSingleShot(True)
        self._auto_scroll_timer.timeout.connect(self._scroll_to_latest)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 22, 28, 28)
        layout.setSpacing(14)
        toolbar = QFrame()
        toolbar.setObjectName("toolbarCard")
        toolbar_layout = QVBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(12, 11, 12, 11)
        toolbar_layout.setSpacing(9)
        filters = QHBoxLayout()
        filters.setSpacing(9)
        self.source_filter = QComboBox()
        self.source_filter.addItem("全部来源", "")
        self.source_filter.addItem("接口进程", "API")
        self.source_filter.addItem("工作进程", "WORKER")
        self.source_filter.addItem("桌面与启动器", "DESKTOP")
        self.source_filter.setMinimumWidth(128)
        self.level_filter = QComboBox()
        self.level_filter.addItem("全部级别", "")
        self.level_filter.addItem("调试", "DEBUG")
        self.level_filter.addItem("信息", "INFO")
        self.level_filter.addItem("警告", "WARNING")
        self.level_filter.addItem("错误", "ERROR")
        self.level_filter.setMinimumWidth(128)
        self.category_filter = QComboBox()
        self.category_filter.addItems(tuple(LOG_CATEGORY_FILTERS))
        self.category_filter.setMinimumWidth(128)
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索日志")
        self.search.setMinimumWidth(220)
        filters.addWidget(self.source_filter)
        filters.addWidget(self.level_filter)
        filters.addWidget(self.category_filter)
        filters.addWidget(self.search, 1)
        toolbar_layout.addLayout(filters)

        actions = QHBoxLayout()
        actions.setSpacing(9)
        self.auto_scroll = ToggleSwitch(True)
        self.auto_scroll.setAccessibleName("日志自动滚动")
        auto_scroll_control = QWidget()
        auto_scroll_layout = QHBoxLayout(auto_scroll_control)
        auto_scroll_layout.setContentsMargins(0, 0, 0, 0)
        auto_scroll_layout.setSpacing(7)
        auto_scroll_layout.addWidget(self.auto_scroll)
        auto_scroll_layout.addWidget(_label("自动滚动"))
        clear = QPushButton("清空视图")
        copy = QPushButton("复制当前")
        copy.setMinimumWidth(84)
        clear.setMinimumWidth(84)
        actions.addWidget(auto_scroll_control)
        actions.addStretch()
        actions.addWidget(copy)
        actions.addWidget(clear)
        toolbar_layout.addLayout(actions)
        layout.addWidget(toolbar)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.output.document().setMaximumBlockCount(MAX_LOG_LINES)
        layout.addWidget(self.output, 1)
        self.source_filter.currentTextChanged.connect(self._render)
        self.level_filter.currentTextChanged.connect(self._render)
        self.category_filter.currentTextChanged.connect(self._render)
        self.search.textChanged.connect(self._render)
        self.auto_scroll.toggled.connect(self._schedule_auto_scroll)
        clear.clicked.connect(self.clear)
        copy.clicked.connect(lambda: QApplication.clipboard().setText(self.output.toPlainText()))

    def add_line(self, source: str, line: str) -> None:
        if line.startswith(STREAM_FRAME_PREFIX) and self._add_stream_frame(source, line):
            return
        clean = ANSI_ESCAPE.sub("", line)
        detected_level = (
            "DEBUG"
            if "[调试]" in clean
            else "ERROR"
            if "[CRITICAL]" in clean
            else next(
                (
                    name
                    for name in ("ERROR", "WARNING", "INFO", "DEBUG")
                    if f"[{name}]" in clean
                ),
                None,
            )
        )
        category_match = LOG_CATEGORY_PATTERN.search(clean)
        category = (
            category_match.group(1)
            if category_match is not None
            else "错误" if detected_level == "ERROR"
            else "警告" if detected_level == "WARNING"
            else "诊断"
        )
        if detected_level is not None:
            level = detected_level
        elif category == "错误":
            level = "ERROR"
        elif category == "警告":
            level = "WARNING"
        else:
            level = "INFO"
        normalized_source = source.upper()
        if normalized_source == "LAUNCHER":
            normalized_source = "DESKTOP"
        self._tail_stream = None
        self._append_entry(normalized_source, level, category, clean)

    def _add_stream_frame(self, source: str, line: str) -> bool:
        try:
            payload = json.loads(line.removeprefix(STREAM_FRAME_PREFIX))
        except (TypeError, ValueError):
            return False
        if not isinstance(payload, dict):
            return False
        action = payload.get("action")
        stream_id = payload.get("streamId")
        formatted = payload.get("formatted")
        chunk = payload.get("chunk")
        if (
            action not in {"start", "chunk", "end"}
            or not isinstance(stream_id, str)
            or not isinstance(formatted, str)
            or not isinstance(chunk, str)
        ):
            return False
        normalized_source = source.upper()
        if normalized_source == "LAUNCHER":
            normalized_source = "DESKTOP"
        level = payload.get("level")
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            level = "INFO"
        if level == "CRITICAL":
            level = "ERROR"
        category = payload.get("category")
        if category not in PRODUCT_LOG_CATEGORIES:
            category = "流式"
        clean_formatted = ANSI_ESCAPE.sub("", formatted)
        clean_chunk = ANSI_ESCAPE.sub("", chunk)
        key = (normalized_source, stream_id)

        if action == "start":
            self._tail_stream = key
            self._stream_headers[key] = clean_formatted.replace(
                "开始流式返回：",
                "继续流式返回：",
            )
            self._append_entry(
                normalized_source,
                level,
                category,
                clean_formatted,
                stream_id=stream_id,
            )
            return True

        if action == "chunk":
            if (
                clean_chunk
                and self._tail_stream == key
                and self._lines
                and self._lines[-1][0] == normalized_source
                and self._lines[-1][4] == stream_id
            ):
                entry = self._lines[-1]
                was_visible = self._matches(*entry[:4])
                self._lines[-1] = (*entry[:3], entry[3] + clean_chunk, stream_id)
                is_visible = self._matches(*self._lines[-1][:4])
                if was_visible and is_visible:
                    cursor = self.output.textCursor()
                    cursor.movePosition(QTextCursor.MoveOperation.End)
                    cursor.insertText(clean_chunk)
                    self.output.setTextCursor(cursor)
                    self._schedule_auto_scroll()
                elif was_visible != is_visible:
                    self._render()
            else:
                self._tail_stream = key
                resumed = self._stream_headers.get(key, "") + clean_chunk
                self._append_entry(
                    normalized_source,
                    level,
                    category,
                    resumed or clean_formatted,
                    stream_id=stream_id,
                )
            return True

        self._stream_headers.pop(key, None)
        self._tail_stream = None
        self._append_entry(
            normalized_source,
            level,
            category,
            clean_formatted,
        )
        return True

    def _append_entry(
        self,
        source: str,
        level: str,
        category: str,
        text: str,
        *,
        stream_id: str | None = None,
    ) -> None:
        self._lines.append((source, level, category, text, stream_id))
        if self._matches(source, level, category, text):
            self.output.appendPlainText(text)
            self._schedule_auto_scroll()

    def clear(self) -> None:
        self._lines.clear()
        self._tail_stream = None
        self._stream_headers.clear()
        self.output.clear()

    def _matches(
        self,
        source: str,
        level: str,
        category: str,
        line: str,
    ) -> bool:
        selected_source = str(self.source_filter.currentData() or "")
        selected_level = str(self.level_filter.currentData() or "")
        selected_categories = LOG_CATEGORY_FILTERS[
            self.category_filter.currentText()
        ]
        needle = self.search.text().strip().lower()
        return (
            (not selected_source or selected_source == source)
            and (not selected_level or selected_level == level)
            and (
                selected_categories is None
                or category in selected_categories
            )
            and (not needle or needle in line.lower())
        )

    def _render(self) -> None:
        text = "\n".join(
            line
            for source, level, category, line, _stream_id in self._lines
            if self._matches(source, level, category, line)
        )
        self.output.setPlainText(text)
        self._schedule_auto_scroll()

    def _schedule_auto_scroll(self, enabled: bool | None = None) -> None:
        if enabled is False or not self.auto_scroll.isChecked():
            self._auto_scroll_timer.stop()
            return
        self._auto_scroll_timer.start(0)

    def _scroll_to_latest(self) -> None:
        if not self.auto_scroll.isChecked():
            return
        vertical = self.output.verticalScrollBar()
        horizontal = self.output.horizontalScrollBar()
        vertical.setValue(vertical.maximum())
        horizontal.setValue(horizontal.minimum())

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._schedule_auto_scroll()


def _setting_row(title: str, description: str, control: QWidget) -> QWidget:
    row = QWidget()
    row.setObjectName("settingRow")
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 12, 0, 12)
    layout.setSpacing(24)
    copy = QVBoxLayout()
    copy.setContentsMargins(0, 0, 0, 0)
    copy.setSpacing(3)
    copy.addWidget(_label(title, "settingTitle"))
    description_label = _label(description, "settingDescription")
    description_label.setWordWrap(True)
    copy.addWidget(description_label)
    layout.addLayout(copy, 1)
    layout.addWidget(control, 0, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
    return row


def _settings_card(
    title: str,
    description: str,
    rows: Iterable[QWidget],
) -> tuple[QFrame, QLabel]:
    card = QFrame()
    card.setObjectName("settingsCard")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(22, 19, 22, 8)
    layout.setSpacing(0)
    layout.addWidget(_label(title, "settingsSectionTitle"))
    description_label = _label(description, "settingsSectionDescription")
    description_label.setWordWrap(True)
    layout.addWidget(description_label)
    layout.addSpacing(7)
    for row in rows:
        divider = QFrame()
        divider.setObjectName("settingDivider")
        layout.addWidget(divider)
        layout.addWidget(row)
    return card, description_label


class SettingsPage(QWidget):
    settings_changed = Signal(object)

    def __init__(self, settings: DesktopSettings, data_root: Path) -> None:
        super().__init__()
        self._applying = True
        self.setObjectName("page")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 22, 28, 28)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        content = QWidget()
        content.setObjectName("settingsContent")
        content.setMaximumWidth(980)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 7, 0)
        layout.setSpacing(14)

        auto_save = QHBoxLayout()
        auto_save.addWidget(_label("所有修改都会自动保存，无需额外确认。", "settingsHint"))
        auto_save.addStretch()
        self.auto_save_status = _label("自动保存已开启", "autoSaveStatus")
        auto_save.addWidget(self.auto_save_status)
        layout.addLayout(auto_save)

        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setKeyboardTracking(False)
        self.port.setFixedWidth(190)
        self.port.setValue(settings.port)
        self.allow_lan = ToggleSwitch(settings.allow_lan)
        self.allow_lan.setAccessibleName("允许局域网访问")
        self.log_level = QComboBox()
        self.log_level.addItems(LOG_LEVELS)
        self.log_level.setFixedWidth(190)
        self.log_level.setCurrentText(settings.log_level)
        self.open_browser = ToggleSwitch(settings.open_browser_on_start)
        self.open_browser.setAccessibleName("自动打开网页")
        data_path = QLineEdit(str(data_root))
        data_path.setReadOnly(True)
        data_path.setMinimumWidth(300)
        data_path.setMaximumWidth(480)
        data_path.setCursorPosition(0)
        data_path.setToolTip(str(data_root))
        server_rows = (
            _setting_row("运行端口", "API 与网页前端使用的监听端口", self.port),
            _setting_row("局域网访问", "关闭时只允许本机连接", self.allow_lan),
            _setting_row("日志等级", "控制 API、Worker 与启动器的日志输出", self.log_level),
            _setting_row(
                "自动打开网页",
                "控制中心本次启动后，首次运行后端时打开默认浏览器",
                self.open_browser,
            ),
            _setting_row("数据目录", "当前数据位置，仅供查看", data_path),
        )
        server_card, self.server_description = _settings_card(
            "后端启动设置",
            "修改会自动保存；运行中调整端口、网络或日志时会自动重启后端。",
            server_rows,
        )
        layout.addWidget(server_card)

        selected_resident_models = set(settings.resident_models)
        self.resident_model_switches: dict[str, ToggleSwitch] = {}
        resident_rows: list[QWidget] = []
        for option in LOCAL_MODEL_OPTIONS:
            control = ToggleSwitch(option.model_id in selected_resident_models)
            control.setAccessibleName(f"常驻 {option.label}")
            self.resident_model_switches[option.model_id] = control
            resident_rows.append(
                _setting_row(option.label, option.description, control)
            )
        resident_card, self.resident_description = _settings_card(
            "常驻模型",
            "选中的模型会在后端启动时加载并保持驻留；会增加启动时间和内存或显存占用。",
            resident_rows,
        )
        layout.addWidget(resident_card)

        self.pet_enabled = ToggleSwitch(settings.pet_enabled)
        self.pet_enabled.setAccessibleName("显示桌宠")
        self.pet_top = ToggleSwitch(settings.pet_always_on_top)
        self.pet_top.setAccessibleName("桌宠始终置顶")
        self.pet_scale = QComboBox()
        self.pet_scale.addItems([f"{value}%" for value in sorted(PET_SCALES)])
        self.pet_scale.setFixedWidth(190)
        self.pet_scale.setCurrentText(f"{settings.pet_scale_percent}%")
        pet_rows = (
            _setting_row("显示桌宠", "随控制中心启动并记住上次位置", self.pet_enabled),
            _setting_row("始终置顶", "让桌宠保持在其他窗口上方", self.pet_top),
            _setting_row("显示大小", "调整桌宠窗口和动画尺寸", self.pet_scale),
        )
        pet_card, _pet_description = _settings_card(
            "桌宠设置",
            "桌宠相关修改会立刻应用。",
            pet_rows,
        )
        layout.addWidget(pet_card)
        layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        self._applying = False
        self.port.valueChanged.connect(self._emit_settings)
        self.allow_lan.toggled.connect(self._emit_settings)
        self.log_level.currentTextChanged.connect(self._emit_settings)
        self.open_browser.toggled.connect(self._emit_settings)
        for control in self.resident_model_switches.values():
            control.toggled.connect(self._emit_settings)
        self.pet_enabled.toggled.connect(self._emit_settings)
        self.pet_top.toggled.connect(self._emit_settings)
        self.pet_scale.currentTextChanged.connect(self._emit_settings)

    def _emit_settings(self, *_args: object) -> None:
        if self._applying:
            return
        settings = DesktopSettings(
            port=self.port.value(),
            allow_lan=self.allow_lan.isChecked(),
            log_level=self.log_level.currentText(),
            open_browser_on_start=self.open_browser.isChecked(),
            resident_models=tuple(
                option.model_id
                for option in LOCAL_MODEL_OPTIONS
                if self.resident_model_switches[option.model_id].isChecked()
            ),
            pet_enabled=self.pet_enabled.isChecked(),
            pet_always_on_top=self.pet_top.isChecked(),
            pet_scale_percent=int(self.pet_scale.currentText().rstrip("%")),
        )
        self.settings_changed.emit(settings)

    def apply_pet_settings(self, settings: DesktopSettings) -> None:
        """Keep tray and pet-menu changes reflected without emitting a duplicate write."""

        self._applying = True
        try:
            self.pet_enabled.setChecked(settings.pet_enabled)
            self.pet_top.setChecked(settings.pet_always_on_top)
            self.pet_scale.setCurrentText(f"{settings.pet_scale_percent}%")
        finally:
            self._applying = False

    def set_backend_running(self, running: bool) -> None:
        self.server_description.setText(
            "后端正在运行；调整端口、网络或日志后会自动重启并应用。"
            if running
            else "修改会自动保存；启动后端时直接使用当前设置。"
        )
        self.resident_description.setText(
            "选择会自动保存；后端正在运行时将在下次重启后生效。"
            if running
            else "选中的模型会在下次启动后端时加载并保持驻留。"
        )

    def show_auto_save_status(self, message: str) -> None:
        self.auto_save_status.setText(message)


class DesktopWindow(QMainWindow):
    start_requested = Signal()
    stop_requested = Signal()
    restart_requested = Signal()
    open_web_requested = Signal()
    quit_requested = Signal()
    settings_changed = Signal(object)
    job_command_requested = Signal(str, str)
    queue_pause_requested = Signal(bool)

    def __init__(
        self,
        settings: DesktopSettings,
        *,
        native_icon_path: Path,
        brand_logo_path: Path,
        data_root: Path,
    ) -> None:
        super().__init__()
        self._allow_close = False
        self._close_to_tray_enabled = True
        self.setWindowTitle("Saber-Translator")
        self.setWindowIcon(QIcon(str(native_icon_path)))
        self.setMinimumSize(920, 640)
        self.resize(settings.window_width, settings.window_height)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        root = QWidget()
        root.setObjectName("desktopRoot")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(5, 5, 5, 5)
        shell = QFrame()
        shell.setObjectName("windowShell")
        outer.addWidget(shell)
        shell_layout = QHBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        self.sidebar = Sidebar(brand_logo_path)
        shell_layout.addWidget(self.sidebar)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        self.title_bar = TitleBar(self)
        content_layout.addWidget(self.title_bar)
        self.stack = QStackedWidget()
        self.overview = OverviewPage()
        self.tasks = TaskCenterPage()
        self.logs = LogPage()
        self.settings = SettingsPage(settings, data_root)
        for page in (self.overview, self.tasks, self.logs, self.settings):
            self.stack.addWidget(page)
        content_layout.addWidget(self.stack, 1)
        shell_layout.addWidget(content, 1)
        self._resize_zones = self._create_resize_zones()
        self.sidebar.page_selected.connect(self._select_page)
        self.overview.start_requested.connect(self.start_requested)
        self.overview.stop_requested.connect(self.stop_requested)
        self.overview.restart_requested.connect(self.restart_requested)
        self.overview.open_web_requested.connect(self.open_web_requested)
        self.tasks.command_requested.connect(self.job_command_requested)
        self.tasks.queue_pause_requested.connect(self.queue_pause_requested)
        self.settings.settings_changed.connect(self.settings_changed)

    def _select_page(self, index: int, title: str) -> None:
        self.stack.setCurrentIndex(index)
        self.title_bar.title.setText(title)

    def toggle_maximized(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def request_close_to_tray(self) -> None:
        if self._close_to_tray_enabled:
            self.hide()
        else:
            self.quit_requested.emit()

    def set_close_to_tray_enabled(self, enabled: bool) -> None:
        self._close_to_tray_enabled = enabled

    def allow_close(self) -> None:
        self._allow_close = True
        self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._allow_close:
            event.accept()
        else:
            self.hide()
            event.ignore()

    def show_error(self, message: str) -> None:
        QMessageBox.warning(self, "Saber-Translator", message)

    def show_notice(self, message: str) -> None:
        QMessageBox.information(self, "Saber-Translator", message)

    def set_launcher_status(self, status: LauncherStatus) -> None:
        self.overview.update_status(status)

    def set_jobs(
        self,
        queue: list[Mapping[str, object]],
        history: list[Mapping[str, object]],
        queue_paused: bool,
        worker_online: bool,
        executor_busy: bool,
        waiting_reason: str | None,
    ) -> None:
        self.overview.update_jobs(queue)
        self.tasks.set_jobs(
            queue,
            history,
            queue_paused,
            worker_online,
            executor_busy,
            waiting_reason,
        )

    def set_task_connected(self, connected: bool) -> None:
        self.overview.set_connected(connected)
        self.tasks.set_connected(connected)

    def add_log(self, source: str, line: str) -> None:
        self.logs.add_line(source, line)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        thickness = 8
        corner = 16
        width = self.width()
        height = self.height()
        geometries = (
            QRect(0, 0, corner, corner),
            QRect(corner, 0, max(0, width - 2 * corner), thickness),
            QRect(width - corner, 0, corner, corner),
            QRect(0, corner, thickness, max(0, height - 2 * corner)),
            QRect(width - thickness, corner, thickness, max(0, height - 2 * corner)),
            QRect(0, height - corner, corner, corner),
            QRect(corner, height - thickness, max(0, width - 2 * corner), thickness),
            QRect(width - corner, height - corner, corner, corner),
        )
        for zone, geometry in zip(self._resize_zones, geometries, strict=True):
            zone.setGeometry(geometry)
            zone.setVisible(not self.isMaximized())
            zone.raise_()

    def _create_resize_zones(self) -> list[ResizeZone]:
        definitions = (
            (Qt.Edge.TopEdge | Qt.Edge.LeftEdge, Qt.CursorShape.SizeFDiagCursor),
            (Qt.Edge.TopEdge, Qt.CursorShape.SizeVerCursor),
            (Qt.Edge.TopEdge | Qt.Edge.RightEdge, Qt.CursorShape.SizeBDiagCursor),
            (Qt.Edge.LeftEdge, Qt.CursorShape.SizeHorCursor),
            (Qt.Edge.RightEdge, Qt.CursorShape.SizeHorCursor),
            (Qt.Edge.BottomEdge | Qt.Edge.LeftEdge, Qt.CursorShape.SizeBDiagCursor),
            (Qt.Edge.BottomEdge, Qt.CursorShape.SizeVerCursor),
            (Qt.Edge.BottomEdge | Qt.Edge.RightEdge, Qt.CursorShape.SizeFDiagCursor),
        )
        return [ResizeZone(self, edges, cursor) for edges, cursor in definitions]
