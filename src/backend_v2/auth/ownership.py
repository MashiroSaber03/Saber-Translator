"""Small request/worker ownership context used by shared repositories."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator

from flask import g, has_request_context

from src.backend_v2.auth.constants import LOCAL_USER_ID


_worker_owner_id: ContextVar[str | None] = ContextVar(
    "saber_worker_owner_id", default=None
)


def effective_owner_id() -> str:
    """Return the authenticated request owner, worker owner, or local profile owner."""

    worker_owner = _worker_owner_id.get()
    if worker_owner:
        return worker_owner
    if has_request_context():
        identity = getattr(g, "saber_identity", None)
        user_id = getattr(identity, "user_id", None)
        if isinstance(user_id, str) and user_id:
            return user_id
    return LOCAL_USER_ID


@contextmanager
def owner_scope(owner_user_id: str) -> Iterator[None]:
    """Propagate a claimed job/operation owner through worker-side services."""

    token = _worker_owner_id.set(owner_user_id)
    try:
        yield
    finally:
        _worker_owner_id.reset(token)
