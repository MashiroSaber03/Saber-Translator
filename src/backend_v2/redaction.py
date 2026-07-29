"""Defensive redaction for values that may cross a persistence/API boundary.

Credentials are stored as immutable server-side snapshots, but third-party
clients and plugins can still include request headers or keys in exception
messages.  Persisted failures and events therefore need a final scrub even
when the normal data flow never places a secret in their payload.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import re
from typing import Any


REDACTED = "[REDACTED]"

_SENSITIVE_KEY = re.compile(
    r"(?:"
    r"api[_-]?key|secret(?:[_-]?key)?|client[_-]?secret|"
    r"access[_-]?token|refresh[_-]?token|auth(?:orization)?|"
    r"password|passwd|cookie|session[_-]?token"
    r")",
    re.IGNORECASE,
)
_AUTHORIZATION = re.compile(
    r"(?i)\b(authorization|x-authorization)\b"
    r"(\s*[:=]\s*)(bearer|basic|token)?(\s*)"
    r"([^\s,;\"']+)"
)
_LABELED_SECRET = re.compile(
    r"""(?ix)
    \b(
        api[_-]?key|secret(?:[_-]?key)?|client[_-]?secret|
        access[_-]?token|refresh[_-]?token|password|passwd|
        cookie|session[_-]?token
    )
    (\s*["']?\s*[:=]\s*["']?)
    ([^"'\s,;}]+)
    """
)
_WINDOWS_PATH = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:[A-Z]:\\|\\\\)[^\r\n\"'<>|]*"
)
_PRIVATE_POSIX_PATH = re.compile(
    r"(?<![A-Za-z0-9])/(?:home|Users|tmp|private/tmp|var/tmp)/"
    r"[^\s\"'<>]*"
)


def secret_values_from_json(secret_json: str | None) -> tuple[str, ...]:
    """Return scalar values from one credential snapshot for exact scrubbing."""

    if not secret_json:
        return ()
    try:
        value = json.loads(secret_json)
    except (TypeError, ValueError):
        return ()
    values: set[str] = set()

    def visit(current: object) -> None:
        if isinstance(current, Mapping):
            for child in current.values():
                visit(child)
        elif isinstance(current, Sequence) and not isinstance(
            current,
            (str, bytes, bytearray),
        ):
            for child in current:
                visit(child)
        elif isinstance(current, str) and len(current) >= 4:
            values.add(current)

    visit(value)
    return tuple(sorted(values, key=len, reverse=True))


def redact_sensitive_text(
    value: object,
    *,
    secret_values: Sequence[str] = (),
    redact_paths: bool = True,
) -> str:
    """Scrub known credentials, labeled secrets, auth headers and local paths."""

    text = str(value)
    for secret in sorted(
        {item for item in secret_values if isinstance(item, str) and len(item) >= 4},
        key=len,
        reverse=True,
    ):
        text = text.replace(secret, REDACTED)
    text = _AUTHORIZATION.sub(
        lambda match: (
            f"{match.group(1)}{match.group(2)}"
            f"{match.group(3) or ''}{match.group(4)}{REDACTED}"
        ),
        text,
    )
    text = _LABELED_SECRET.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}",
        text,
    )
    if redact_paths:
        text = _WINDOWS_PATH.sub("[LOCAL_PATH]", text)
        text = _PRIVATE_POSIX_PATH.sub("[LOCAL_PATH]", text)
    return text


def redact_sensitive_value(
    value: Any,
    *,
    secret_values: Sequence[str] = (),
    redact_paths: bool = True,
) -> Any:
    """Recursively scrub a JSON-compatible value without mutating the input."""

    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            if _SENSITIVE_KEY.search(key_text):
                redacted[key_text] = REDACTED
            else:
                redacted[key_text] = redact_sensitive_value(
                    child,
                    secret_values=secret_values,
                    redact_paths=redact_paths,
                )
        return redacted
    if isinstance(value, list):
        return [
            redact_sensitive_value(
                child,
                secret_values=secret_values,
                redact_paths=redact_paths,
            )
            for child in value
        ]
    if isinstance(value, tuple):
        return [
            redact_sensitive_value(
                child,
                secret_values=secret_values,
                redact_paths=redact_paths,
            )
            for child in value
        ]
    if isinstance(value, str):
        return redact_sensitive_text(
            value,
            secret_values=secret_values,
            redact_paths=redact_paths,
        )
    return value
