"""Unified, secret-safe logging for every backend-first process role."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
from typing import Final

import colorama

from src.backend_v2.redaction import redact_sensitive_text


LOG_LEVEL_ENV: Final = "SABER_V2_LOG_LEVEL"
_HANDLER_MARKER: Final = "_saber_v2_handler"
_NOISY_LOGGERS: Final = {
    "PIL": logging.WARNING,
    "alembic": logging.WARNING,
    "charset_normalizer": logging.WARNING,
    "filelock": logging.WARNING,
    "httpcore": logging.WARNING,
    "httpx": logging.WARNING,
    "matplotlib": logging.WARNING,
    "manga_ocr": logging.WARNING,
    "multipart": logging.WARNING,
    "openai": logging.WARNING,
    "sqlalchemy.engine": logging.WARNING,
    "torch": logging.WARNING,
    "transformers": logging.WARNING,
    "urllib3": logging.WARNING,
    "waitress": logging.WARNING,
}


class SecretSafeFormatter(logging.Formatter):
    """Redact credentials from the final message, including tracebacks."""

    def format(self, record: logging.LogRecord) -> str:
        return redact_sensitive_text(
            super().format(record),
            redact_paths=False,
        )


class ColoredSecretSafeFormatter(SecretSafeFormatter):
    """Apply level colors to the complete redacted console line."""

    _COLORS: Final = {
        logging.DEBUG: colorama.Fore.CYAN,
        logging.INFO: colorama.Fore.GREEN,
        logging.WARNING: colorama.Fore.YELLOW,
        logging.ERROR: colorama.Fore.RED,
        logging.CRITICAL: colorama.Fore.RED + colorama.Style.BRIGHT,
    }

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        color = self._COLORS.get(record.levelno)
        if color is None:
            return rendered
        return f"{color}{rendered}{colorama.Style.RESET_ALL}"


def _console_level(explicit: str | None) -> int:
    raw = (explicit or os.environ.get(LOG_LEVEL_ENV, "INFO")).strip().upper()
    level = getattr(logging, raw, None)
    if not isinstance(level, int):
        return logging.INFO
    return level


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
    role_label = normalized_role.upper()
    format_template = (
        "%(asctime)s [%(levelname)s] "
        f"[{role_label}:%(process)d] %(name)s - %(message)s"
    )
    console_formatter = ColoredSecretSafeFormatter(
        format_template,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_formatter = SecretSafeFormatter(
        format_template,
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
    console = _mark(logging.StreamHandler(sys.stdout))
    console.setLevel(_console_level(console_level))
    console.setFormatter(console_formatter)
    root.addHandler(console)

    file_handler = _mark(
        RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    root.addHandler(file_handler)

    for logger_name, level in _NOISY_LOGGERS.items():
        logging.getLogger(logger_name).setLevel(level)
    return log_path
