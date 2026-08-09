"""Transparent animated desktop pet window and custom atlas manifest."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import QPoint, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QBitmap, QGuiApplication, QImage, QMouseEvent, QPainter, QRegion
from PySide6.QtWidgets import QApplication, QMenu, QWidget

from src.backend_v2.desktop.pet_state import PetState
from src.backend_v2.desktop.settings import PET_SCALES


PET_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PetAnimation:
    state: PetState
    row: int
    frame_count: int
    durations_ms: tuple[int, ...]
    loop: bool


@dataclass(frozen=True, slots=True)
class PetManifest:
    spritesheet_path: Path
    cell_width: int
    cell_height: int
    columns: int
    animations: dict[PetState, PetAnimation]

    @classmethod
    def load(cls, path: Path) -> "PetManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schemaVersion") != PET_SCHEMA_VERSION:
            raise ValueError("unsupported pet manifest schema")
        cell = payload.get("cell")
        rows = payload.get("rows")
        columns = payload.get("columns")
        sheet_name = payload.get("spritesheetPath")
        if not isinstance(cell, dict) or not isinstance(rows, list):
            raise ValueError("pet manifest geometry is missing")
        width = cell.get("width")
        height = cell.get("height")
        if not isinstance(width, int) or not isinstance(height, int) or width < 1 or height < 1:
            raise ValueError("invalid pet cell geometry")
        if columns != 8 or not isinstance(sheet_name, str):
            raise ValueError("invalid pet atlas declaration")

        animations: dict[PetState, PetAnimation] = {}
        occupied_rows: set[int] = set()
        for raw in rows:
            if not isinstance(raw, dict):
                raise ValueError("invalid pet animation row")
            state = PetState(str(raw.get("state")))
            row = raw.get("row")
            frame_count = raw.get("frameCount")
            durations = raw.get("durationsMs")
            loop = raw.get("loop")
            if (
                not isinstance(row, int)
                or row < 0
                or row in occupied_rows
                or not isinstance(frame_count, int)
                or not 1 <= frame_count <= columns
                or not isinstance(durations, list)
                or len(durations) != frame_count
                or any(not isinstance(value, int) or value < 40 for value in durations)
                or not isinstance(loop, bool)
            ):
                raise ValueError(f"invalid pet animation row: {state.value}")
            occupied_rows.add(row)
            animations[state] = PetAnimation(
                state=state,
                row=row,
                frame_count=frame_count,
                durations_ms=tuple(durations),
                loop=loop,
            )
        missing = set(PetState) - set(animations)
        if missing:
            raise ValueError(f"pet manifest is missing states: {sorted(item.value for item in missing)}")
        return cls(
            spritesheet_path=path.parent / sheet_name,
            cell_width=width,
            cell_height=height,
            columns=columns,
            animations=animations,
        )

    def validate_image(self, image: QImage) -> None:
        rows = max(animation.row for animation in self.animations.values()) + 1
        expected = QSize(self.cell_width * self.columns, self.cell_height * rows)
        if image.size() != expected:
            raise ValueError(
                f"pet atlas size is {image.width()}x{image.height()}, "
                f"expected {expected.width()}x{expected.height()}"
            )


class PetWindow(QWidget):
    show_main_requested = Signal()
    start_stop_requested = Signal()
    open_web_requested = Signal()
    quit_requested = Signal()
    hidden_requested = Signal()
    scale_requested = Signal(int)
    always_on_top_requested = Signal(bool)
    position_changed = Signal(str, float, float)

    def __init__(
        self,
        manifest_path: Path,
        *,
        fallback_logo: Path,
        scale_percent: int = 100,
        always_on_top: bool = True,
    ) -> None:
        super().__init__()
        self._manifest: PetManifest | None = None
        self._atlas = QImage()
        self._fallback = QImage(str(fallback_logo))
        self._frame_cache: dict[tuple[PetState, int, int], QImage] = {}
        self._mask_cache: dict[tuple[PetState, int, int], QRegion] = {}
        self._base_state = PetState.IDLE
        self._visible_state = PetState.IDLE
        self._frame_index = 0
        self._transient = False
        self._drag_origin: QPoint | None = None
        self._window_origin: QPoint | None = None
        self._dragged = False
        self._last_drag_x = 0
        self._scale_percent = scale_percent if scale_percent in PET_SCALES else 100
        self._always_on_top = always_on_top
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._advance_frame)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle("Saber 桌宠")
        self._apply_window_flags()
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        try:
            manifest = PetManifest.load(manifest_path)
            atlas = QImage(str(manifest.spritesheet_path))
            if atlas.isNull():
                raise ValueError("pet atlas could not be loaded")
            manifest.validate_image(atlas)
            self._manifest = manifest
            self._atlas = atlas
        except (OSError, ValueError, json.JSONDecodeError):
            self._manifest = None
        self._resize_for_scale()
        self._restart_animation()

    @property
    def scale_percent(self) -> int:
        return self._scale_percent

    @property
    def always_on_top(self) -> bool:
        return self._always_on_top

    def set_base_state(self, state: PetState) -> None:
        self._base_state = state
        if not self._transient and self._drag_origin is None:
            self._play(state)

    def greet(self) -> None:
        self._transient = True
        self._play(PetState.GREETING, force=True)

    def set_scale_percent(self, percent: int) -> None:
        if percent not in PET_SCALES or percent == self._scale_percent:
            return
        center = self.geometry().center()
        self._scale_percent = percent
        self._frame_cache.clear()
        self._mask_cache.clear()
        self._resize_for_scale()
        self.move(center - QPoint(self.width() // 2, self.height() // 2))
        self.clamp_to_visible_screen()
        self._render_frame()

    def set_always_on_top(self, enabled: bool) -> None:
        if enabled == self._always_on_top:
            return
        was_visible = self.isVisible()
        self._always_on_top = enabled
        self._apply_window_flags()
        if was_visible:
            self.show()

    def restore_position(self, screen_name: str, x_ratio: float, y_ratio: float) -> None:
        screens = QGuiApplication.screens()
        screen = next((item for item in screens if item.name() == screen_name), None)
        screen = screen or QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        travel_x = max(0, area.width() - self.width())
        travel_y = max(0, area.height() - self.height())
        x = area.left() + round(travel_x * max(0.0, min(1.0, x_ratio)))
        y = area.top() + round(travel_y * max(0.0, min(1.0, y_ratio)))
        if x_ratio >= 0.99:
            x = max(area.left(), area.right() - self.width() - 23)
        if y_ratio >= 0.99:
            y = max(area.top(), area.bottom() - self.height() - 23)
        self.move(x, y)

    def clamp_to_visible_screen(self) -> None:
        center = self.frameGeometry().center()
        screen = QGuiApplication.screenAt(center) or QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        x = max(area.left(), min(self.x(), area.right() - self.width() + 1))
        y = max(area.top(), min(self.y(), area.bottom() - self.height() + 1))
        self.move(x, y)

    def paintEvent(self, _event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        frame = self._current_frame()
        painter.drawImage(self.rect(), frame)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self._show_menu(event.globalPosition().toPoint())
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = event.globalPosition().toPoint()
            self._window_origin = self.pos()
            self._last_drag_x = self._drag_origin.x()
            self._dragged = False
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_origin is None or self._window_origin is None:
            return
        current = event.globalPosition().toPoint()
        delta = current - self._drag_origin
        if delta.manhattanLength() >= QApplication.startDragDistance():
            self._dragged = True
        if self._dragged:
            self.move(self._window_origin + delta)
            direction = PetState.DRAG_RIGHT if current.x() >= self._last_drag_x else PetState.DRAG_LEFT
            if self._visible_state != direction:
                self._transient = False
                self._play(direction, force=True)
            self._last_drag_x = current.x()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._drag_origin is None:
            return
        dragged = self._dragged
        self._drag_origin = None
        self._window_origin = None
        self._dragged = False
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        if dragged:
            self.clamp_to_visible_screen()
            self._transient = False
            self._play(self._base_state, force=True)
            self._emit_position()
        else:
            self.greet()
        event.accept()

    def _apply_window_flags(self) -> None:
        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
        if self._always_on_top:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)

    def _resize_for_scale(self) -> None:
        width, height = 192, 208
        if self._manifest is not None:
            width = self._manifest.cell_width
            height = self._manifest.cell_height
        self.resize(
            round(width * self._scale_percent / 100),
            round(height * self._scale_percent / 100),
        )

    def _play(self, state: PetState, *, force: bool = False) -> None:
        if not force and state == self._visible_state:
            return
        self._visible_state = state
        self._frame_index = 0
        self._restart_animation()

    def _restart_animation(self) -> None:
        self._timer.stop()
        self._render_frame()
        animation = self._animation()
        if animation is not None and animation.frame_count > 1:
            self._timer.start(animation.durations_ms[0])

    def _advance_frame(self) -> None:
        animation = self._animation()
        if animation is None:
            return
        next_frame = self._frame_index + 1
        if next_frame >= animation.frame_count:
            if animation.loop:
                next_frame = 0
            elif self._transient:
                self._transient = False
                self._play(self._base_state, force=True)
                return
            else:
                # A non-looping task reaction (for example success) stays on
                # its final frame until the state machine advances. Replaying
                # the same row here would make the reaction pulse forever.
                self._frame_index = animation.frame_count - 1
                self._render_frame()
                return
        self._frame_index = next_frame
        self._render_frame()
        self._timer.start(animation.durations_ms[self._frame_index])

    def _animation(self) -> PetAnimation | None:
        if self._manifest is None:
            return None
        return self._manifest.animations[self._visible_state]

    def _current_frame(self) -> QImage:
        key = (self._visible_state, self._frame_index, self._scale_percent)
        cached = self._frame_cache.get(key)
        if cached is not None:
            return cached
        if self._manifest is None:
            source = self._fallback
        else:
            animation = self._manifest.animations[self._visible_state]
            source = self._atlas.copy(
                self._frame_index * self._manifest.cell_width,
                animation.row * self._manifest.cell_height,
                self._manifest.cell_width,
                self._manifest.cell_height,
            )
        rendered = source.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._frame_cache[key] = rendered
        return rendered

    def _render_frame(self) -> None:
        frame = self._current_frame()
        key = (self._visible_state, self._frame_index, self._scale_percent)
        region = self._mask_cache.get(key)
        if region is None and not frame.isNull():
            bitmap = QBitmap.fromImage(frame.createAlphaMask())
            region = QRegion(bitmap)
            self._mask_cache[key] = region
        if region is not None and not region.isEmpty():
            self.setMask(region)
        else:
            self.clearMask()
        self.update()

    def _show_menu(self, position: QPoint) -> None:
        menu = QMenu(self)
        menu.setObjectName("petMenu")
        show_action = menu.addAction("显示控制中心")
        start_action = menu.addAction("启动 / 停止后端")
        open_action = menu.addAction("打开网页")
        menu.addSeparator()
        top_action = QAction("始终置顶", menu)
        top_action.setCheckable(True)
        top_action.setChecked(self._always_on_top)
        menu.addAction(top_action)
        scale_menu = menu.addMenu("桌宠大小")
        for percent in sorted(PET_SCALES):
            action = scale_menu.addAction(f"{percent}%")
            action.setCheckable(True)
            action.setChecked(percent == self._scale_percent)
            action.triggered.connect(
                lambda _checked=False, percent=percent: self.scale_requested.emit(percent)
            )
        hide_action = menu.addAction("隐藏桌宠")
        menu.addSeparator()
        quit_action = menu.addAction("退出 Saber-Translator")
        show_action.triggered.connect(self.show_main_requested)
        start_action.triggered.connect(self.start_stop_requested)
        open_action.triggered.connect(self.open_web_requested)
        top_action.toggled.connect(self.always_on_top_requested)
        hide_action.triggered.connect(self.hidden_requested)
        quit_action.triggered.connect(self.quit_requested)
        menu.exec(position)

    def _emit_position(self) -> None:
        center = self.frameGeometry().center()
        screen = QGuiApplication.screenAt(center) or QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        travel_x = max(1, area.width() - self.width())
        travel_y = max(1, area.height() - self.height())
        x_ratio = max(0.0, min(1.0, (self.x() - area.left()) / travel_x))
        y_ratio = max(0.0, min(1.0, (self.y() - area.top()) / travel_y))
        self.position_changed.emit(screen.name(), x_ratio, y_ratio)
