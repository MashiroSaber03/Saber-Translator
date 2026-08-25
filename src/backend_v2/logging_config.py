"""One logging pipeline for readable product logs and detailed diagnostics."""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
from typing import Final

import colorama

from src.backend_v2.redaction import redact_sensitive_text
from src.shared.user_logging import (
    CATEGORY_FIELD,
    CATEGORY_LABELS,
    STREAM_ACTION_FIELD,
    STREAM_CHUNK_FIELD,
    STREAM_FRAME_PREFIX,
    STREAM_ID_FIELD,
    USER_LOG_MARKER,
)


LOG_LEVEL_ENV: Final = "SABER_V2_LOG_LEVEL"
STREAM_FRAME_ENV: Final = "SABER_V2_STREAM_FRAMES"
_HANDLER_MARKER: Final = "_saber_v2_handler"
_NOISY_LOGGERS: Final = {
    "PIL": logging.WARNING,
    "charset_normalizer": logging.WARNING,
    "filelock": logging.WARNING,
    "httpcore": logging.WARNING,
    "httpx": logging.WARNING,
    "matplotlib": logging.WARNING,
    "manga_ocr": logging.WARNING,
    "mangaocr": logging.WARNING,
    "multipart": logging.WARNING,
    "onnxruntime": logging.WARNING,
    "openai": logging.WARNING,
    "paddleocr": logging.WARNING,
    "paddlex": logging.WARNING,
    "rapidocr": logging.WARNING,
    "sqlalchemy.engine": logging.WARNING,
    "torch": logging.WARNING,
    "transformers": logging.WARNING,
    "transformers.utils": logging.ERROR,
    "ultralytics": logging.WARNING,
    "urllib3": logging.WARNING,
    "waitress": logging.WARNING,
    "werkzeug": logging.WARNING,
}
_ROLE_LABELS: Final = {
    "LAUNCHER": "启动器",
    "API": "接口进程",
    "WORKER": "工作进程",
    "DESKTOP": "桌面窗口",
}
_LEVEL_COLORS: Final = {
    logging.DEBUG: colorama.Fore.CYAN,
    logging.INFO: colorama.Fore.GREEN,
    logging.WARNING: colorama.Fore.YELLOW,
    logging.ERROR: colorama.Fore.RED,
    logging.CRITICAL: colorama.Fore.RED + colorama.Style.BRIGHT,
}


class SecretSafeFormatter(logging.Formatter):
    """Redact credentials from the final message, including tracebacks."""

    def format(self, record: logging.LogRecord) -> str:
        return redact_sensitive_text(
            super().format(record),
            redact_paths=False,
        )


class ProductLogFormatter(logging.Formatter):
    """Render only marked product records as compact, readable Chinese."""

    def __init__(
        self,
        role: str,
        *,
        colored: bool = False,
    ) -> None:
        super().__init__(datefmt="%Y-%m-%d %H:%M:%S")
        normalized_role = role.upper()
        self._role = _ROLE_LABELS.get(normalized_role, normalized_role)
        self._colored = colored

    def format(self, record: logging.LogRecord) -> str:
        category = CATEGORY_LABELS.get(
            str(getattr(record, CATEGORY_FIELD, "system")),
            "系统",
        )
        timestamp = self.formatTime(record, self.datefmt)
        level_marker = " [调试]" if record.levelno == logging.DEBUG else ""
        prefix = f"{timestamp} [{self._role}]{level_marker} [{category}]"
        message = record.getMessage().replace("\r\n", "\n").replace("\r", "\n")
        lines = message.splitlines() or [""]
        continuation_prefix = f"{timestamp} [{self._role}]{level_marker} [{category}]"
        rendered_lines = [f"{prefix} {lines[0]}"]
        rendered_lines.extend(
            f"{continuation_prefix}   {line}" for line in lines[1:]
        )
        rendered = "\n".join(rendered_lines)
        rendered = redact_sensitive_text(rendered, redact_paths=False)
        return self._color(record, rendered)

    def _color(self, record: logging.LogRecord, rendered: str) -> str:
        if not self._colored:
            return rendered
        color = _LEVEL_COLORS.get(record.levelno)
        if color is None:
            return rendered
        return f"{color}{rendered}{colorama.Style.RESET_ALL}"


class ProductLogFilter(logging.Filter):
    """Allow only records created by the new product logging vocabulary."""

    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, USER_LOG_MARKER, False))


class DiagnosticFileFilter(logging.Filter):
    """Allow only internal diagnostics in the separate diagnostic file."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not bool(getattr(record, USER_LOG_MARKER, False))


def product_stream_frame(
    record: logging.LogRecord,
    formatted: str,
) -> str | None:
    """Encode one stream event for the desktop child-process pipe."""

    action = getattr(record, STREAM_ACTION_FIELD, None)
    stream_id = getattr(record, STREAM_ID_FIELD, None)
    if action not in {"start", "chunk", "end"} or not isinstance(stream_id, str):
        return None
    chunk = getattr(record, STREAM_CHUNK_FIELD, "")
    safe_chunk = redact_sensitive_text(
        chunk if isinstance(chunk, str) else str(chunk),
        redact_paths=False,
    )
    category = CATEGORY_LABELS.get(
        str(getattr(record, CATEGORY_FIELD, "stream")),
        "流式",
    )
    payload = {
        "action": action,
        "streamId": stream_id,
        "chunk": safe_chunk,
        "formatted": formatted if action != "chunk" else "",
        "level": logging.getLevelName(record.levelno),
        "category": category,
    }
    return STREAM_FRAME_PREFIX + json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


class _ProductStreamOutput:
    """Shared direct-write behavior for console and rotating product files."""

    stream: object
    terminator: str

    def _initialize_product_stream(self) -> None:
        self._open_product_stream_id: str | None = None

    def _write_product_text(self, text: str) -> None:
        self.stream.write(text)  # type: ignore[attr-defined]
        self.flush()  # type: ignore[attr-defined]

    def _close_product_stream_line(self) -> None:
        if self._open_product_stream_id is None:
            return
        self._write_product_text(self.terminator)
        self._open_product_stream_id = None

    def _emit_direct_product_stream(self, record: logging.LogRecord) -> bool:
        action = getattr(record, STREAM_ACTION_FIELD, None)
        stream_id = getattr(record, STREAM_ID_FIELD, None)
        if action not in {"start", "chunk", "end"} or not isinstance(stream_id, str):
            return False
        chunk = getattr(record, STREAM_CHUNK_FIELD, "")
        safe_chunk = redact_sensitive_text(
            chunk if isinstance(chunk, str) else str(chunk),
            redact_paths=False,
        )
        if action == "start":
            self._close_product_stream_line()
            self._write_product_text(self.format(record))  # type: ignore[attr-defined]
            self._open_product_stream_id = stream_id
        elif action == "chunk":
            if self._open_product_stream_id != stream_id:
                self._close_product_stream_line()
                self._write_product_text(self.format(record))  # type: ignore[attr-defined]
            else:
                self._write_product_text(safe_chunk)
            self._open_product_stream_id = stream_id
        else:
            self._close_product_stream_line()
            self._write_product_text(
                self.format(record) + self.terminator  # type: ignore[attr-defined]
            )
        return True


class ProductConsoleHandler(_ProductStreamOutput, logging.StreamHandler):
    """Console handler that flushes each model chunk without adding a newline."""

    def __init__(self, stream: object, *, framed: bool) -> None:
        super().__init__(stream)
        self._initialize_product_stream()
        self._framed = framed

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if self._framed:
                frame = product_stream_frame(record, self.format(record))
                if frame is not None:
                    self._write_product_text(frame + self.terminator)
                    return
            elif self._emit_direct_product_stream(record):
                return
            self._close_product_stream_line()
            logging.StreamHandler.emit(self, record)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


class ProductRotatingFileHandler(_ProductStreamOutput, RotatingFileHandler):
    """Persist streamed responses progressively instead of only after completion."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._initialize_product_stream()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if getattr(record, STREAM_ACTION_FIELD, None) in {"start", "chunk", "end"}:
                if self.shouldRollover(record):
                    self._close_product_stream_line()
                    self.doRollover()
                if self._emit_direct_product_stream(record):
                    return
            self._close_product_stream_line()
            RotatingFileHandler.emit(self, record)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


def _console_level(explicit: str | None) -> int:
    raw = (explicit or os.environ.get(LOG_LEVEL_ENV, "INFO")).strip().upper()
    level = getattr(logging, raw, None)
    if not isinstance(level, int):
        return logging.INFO
    return level


def set_backend_console_level(console_level: str | None) -> None:
    """Update owned product-output handlers without changing log files."""

    level = _console_level(console_level)
    for handler in logging.getLogger().handlers:
        if _owned(handler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(level)


def configure_product_handler(
    handler: logging.Handler,
    *,
    role: str,
    level: str | None,
    colored: bool = False,
) -> None:
    """Apply the same product-log rules to console and desktop handlers."""

    resolved = _console_level(level)
    handler.setLevel(resolved)
    for existing in list(handler.filters):
        if isinstance(existing, ProductLogFilter):
            handler.removeFilter(existing)
    handler.addFilter(ProductLogFilter())
    handler.setFormatter(
        ProductLogFormatter(
            role,
            colored=colored,
        )
    )


def _owned(handler: logging.Handler) -> bool:
    return bool(getattr(handler, _HANDLER_MARKER, False))


def _mark(handler: logging.Handler) -> logging.Handler:
    setattr(handler, _HANDLER_MARKER, True)
    return handler


def configure_backend_logging(
    *,
    role: str,
    data_root: Path,
    console_level: str | None = None,
) -> Path:
    """Configure console INFO plus detailed rotating files for one process."""

    normalized_role = role.strip().lower()
    if normalized_role not in {"launcher", "api", "worker"}:
        raise ValueError(f"unsupported logging role: {role}")

    logs_root = data_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    log_path = logs_root / f"saber-{normalized_role}.log"
    diagnostic_path = logs_root / f"saber-{normalized_role}-diagnostic.log"
    role_label = normalized_role.upper()
    stream_frames = os.environ.get(STREAM_FRAME_ENV) == "1"
    console_formatter = ProductLogFormatter(
        role_label,
        colored=not stream_frames,
    )
    product_file_formatter = ProductLogFormatter(role_label)
    diagnostic_file_formatter = SecretSafeFormatter(
        "%(asctime)s [%(levelname)s] "
        f"[{role_label}:%(process)d] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    for handler in list(root.handlers):
        if _owned(handler):
            root.removeHandler(handler)
            handler.close()
    # DEBUG reaches the detailed file; the console handler performs its own
    # INFO/DEBUG selection so normal terminal output remains readable.
    root.setLevel(logging.DEBUG)

    # Enable ANSI colors on older Windows consoles while remaining idempotent
    # on modern terminals and non-Windows platforms.
    colorama.just_fix_windows_console()
    console = _mark(ProductConsoleHandler(sys.stdout, framed=stream_frames))
    console_level_value = _console_level(console_level)
    console.setLevel(console_level_value)
    console.addFilter(ProductLogFilter())
    console.setFormatter(console_formatter)
    root.addHandler(console)

    product_file_handler = _mark(
        ProductRotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
    )
    product_file_handler.setLevel(logging.DEBUG)
    product_file_handler.addFilter(ProductLogFilter())
    product_file_handler.setFormatter(product_file_formatter)
    root.addHandler(product_file_handler)

    diagnostic_file_handler = _mark(
        RotatingFileHandler(
            diagnostic_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
    )
    diagnostic_file_handler.setLevel(logging.DEBUG)
    diagnostic_file_handler.addFilter(DiagnosticFileFilter())
    diagnostic_file_handler.setFormatter(diagnostic_file_formatter)
    root.addHandler(diagnostic_file_handler)

    for logger_name, level in _NOISY_LOGGERS.items():
        logging.getLogger(logger_name).setLevel(level)
    try:
        from loguru import logger as loguru_logger
    except ImportError:
        pass
    else:
        # Model libraries can install their own global console sink. Project
        # diagnostics already flow through the standard logging hierarchy.
        loguru_logger.remove()
    return log_path
