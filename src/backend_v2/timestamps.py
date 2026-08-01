"""UTC timestamp conventions shared by v2 persistence modules."""

from __future__ import annotations

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Return the naive UTC value used by SQLite timestamp columns."""

    return datetime.now(timezone.utc).replace(tzinfo=None)


def iso_utc(value: datetime | str | None) -> str | None:
    if value is None or isinstance(value, str):
        return value
    return value.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
