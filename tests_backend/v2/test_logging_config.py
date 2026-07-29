from __future__ import annotations

import logging
from pathlib import Path
import sys

import colorama
from flask import Flask

from src.backend_v2.api.app import _install_request_logging
from src.backend_v2.logging_config import (
    ColoredSecretSafeFormatter,
    SecretSafeFormatter,
    configure_backend_logging,
)


def test_secret_safe_formatter_redacts_messages_and_tracebacks() -> None:
    formatter = SecretSafeFormatter("%(levelname)s %(message)s")
    try:
        raise RuntimeError("upstream password=trace-sensitive")
    except RuntimeError:
        exception_info = sys.exc_info()
    record = logging.LogRecord(
        "test",
        logging.ERROR,
        __file__,
        1,
        "request failed: api_key=sk-sensitive Authorization: Bearer token-value",
        (),
        exception_info,
    )

    rendered = formatter.format(record)

    assert "sk-sensitive" not in rendered
    assert "token-value" not in rendered
    assert "trace-sensitive" not in rendered
    assert rendered.count("[REDACTED]") >= 3


def test_colored_formatter_colors_complete_lines_by_severity() -> None:
    formatter = ColoredSecretSafeFormatter("%(levelname)s %(message)s")

    warning = formatter.format(
        logging.LogRecord(
            "test",
            logging.WARNING,
            __file__,
            1,
            "warning message",
            (),
            None,
        )
    )
    error = formatter.format(
        logging.LogRecord(
            "test",
            logging.ERROR,
            __file__,
            1,
            "error message",
            (),
            None,
        )
    )

    assert warning == (
        f"{colorama.Fore.YELLOW}WARNING warning message{colorama.Style.RESET_ALL}"
    )
    assert error == (
        f"{colorama.Fore.RED}ERROR error message{colorama.Style.RESET_ALL}"
    )


def test_backend_logging_writes_info_to_console_and_debug_to_rotating_file(
    tmp_path: Path,
    capsys,
) -> None:
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    try:
        log_path = configure_backend_logging(
            role="worker",
            data_root=tmp_path,
            console_level="INFO",
        )
        logger = logging.getLogger("saber.test")
        logger.debug("debug detail")
        logger.info("safe info api_key=sk-do-not-log")
        for handler in root.handlers:
            handler.flush()

        console = capsys.readouterr().out
        file_text = log_path.read_text(encoding="utf-8")
        assert "[WORKER:" in console
        assert "safe info" in console
        assert colorama.Fore.GREEN in console
        assert colorama.Style.RESET_ALL in console
        assert "sk-do-not-log" not in console
        assert "debug detail" not in console
        assert "debug detail" in file_text
        assert "safe info" in file_text
        assert "sk-do-not-log" not in file_text
        assert "\x1b[" not in file_text
    finally:
        for handler in list(root.handlers):
            if handler not in original_handlers:
                root.removeHandler(handler)
                handler.close()
        for handler in original_handlers:
            if handler not in root.handlers:
                root.addHandler(handler)
        root.setLevel(original_level)


def test_api_request_logging_records_timing_without_query_values(caplog) -> None:
    app = Flask(__name__)
    _install_request_logging(app)

    @app.get("/ping")
    def ping() -> str:
        return "pong"

    with caplog.at_level(logging.INFO, logger="saber.api.http"):
        response = app.test_client().get("/ping?api_key=sk-query-secret")

    assert response.status_code == 200
    assert response.headers["X-Response-Time"].endswith("s")
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "HTTP GET /ping -> 200" in messages
    assert "sk-query-secret" not in messages
