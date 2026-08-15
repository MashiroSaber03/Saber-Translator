"""Shared validation and error helpers for the v2 HTTP routes."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from flask import jsonify, request


def json_body(
    *,
    allowed_keys: Collection[str],
    optional: bool = False,
) -> dict[str, Any]:
    body = request.get_json(silent=True)
    if body is None and optional and not request.get_data(cache=True):
        return {}
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    unknown = set(body) - set(allowed_keys)
    if unknown:
        raise ValueError(
            "unknown request fields: " + ", ".join(sorted(unknown))
        )
    return body


def validate_multipart_fields(
    *,
    allowed_form_keys: Collection[str] = (),
    allowed_file_keys: Collection[str] = (),
) -> None:
    unknown_form = set(request.form) - set(allowed_form_keys)
    unknown_files = set(request.files) - set(allowed_file_keys)
    unknown = [
        *(f"form.{key}" for key in sorted(unknown_form)),
        *(f"file.{key}" for key in sorted(unknown_files)),
    ]
    if unknown:
        raise ValueError("unknown multipart fields: " + ", ".join(unknown))


def required_string(body: Mapping[str, object], key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def required_integer(
    body: Mapping[str, object],
    key: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = body.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{key} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{key} must be at most {maximum}")
    return value


def required_boolean(body: Mapping[str, object], key: str) -> bool:
    value = body.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def integer_value(
    value: object,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or (
        isinstance(value, float) and not value.is_integer()
    ):
        raise ValueError(f"{name} must be an integer")
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return parsed


def require_idempotency_key() -> str:
    value = request.headers.get("Idempotency-Key", "")
    if not value or len(value) > 200:
        raise ValueError(
            "Idempotency-Key is required and must be at most 200 characters"
        )
    return value


def error_response(
    code: str,
    message: str,
    status: int,
    *,
    details: Mapping[str, object] | None = None,
):
    error: dict[str, object] = {"code": code, "message": message}
    if details:
        error["details"] = dict(details)
    return jsonify({"error": error}), status
