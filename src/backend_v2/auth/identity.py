"""Lightweight request identity shared by authentication consumers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SessionIdentity:
    user_id: str
    username: str
    role: str
    session_token_hash: str
    csrf_token_hash: str
