"""Request-scoped authenticated identity helpers."""

from __future__ import annotations

from flask import g

from src.backend_v2.auth.repository import SessionIdentity


def current_identity() -> SessionIdentity:
    identity = getattr(g, "saber_identity", None)
    if not isinstance(identity, SessionIdentity):
        raise RuntimeError("authenticated request identity is unavailable")
    return identity


def current_user_id() -> str:
    return current_identity().user_id


def current_user_role() -> str:
    return current_identity().role


def require_admin() -> SessionIdentity:
    identity = current_identity()
    if identity.role != "admin":
        raise PermissionError("administrator access is required")
    return identity
