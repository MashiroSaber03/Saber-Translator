"""Flask authentication gate shared by every public API blueprint."""

from __future__ import annotations

import hashlib

from flask import Flask, Response, g, jsonify, request

from src.backend_v2.auth.constants import LOCAL_USER_ID, LOCAL_USERNAME, SESSION_COOKIE_NAME
from src.backend_v2.auth.repository import AuthRepository, SessionIdentity
from src.backend_v2.runtime_profile import RuntimeProfile


_PUBLIC_API_PATHS = frozenset(
    {
        "/api/v2/health",
        "/api/v2/system/capabilities",
        "/api/v2/auth/login",
        "/api/v2/auth/register",
        "/api/v2/auth/recover",
    }
)
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def _error(code: str, message: str, status: int) -> tuple[Response, int]:
    return jsonify({"error": {"code": code, "message": message}}), status


def install_authentication(
    app: Flask,
    *,
    repository: AuthRepository,
    profile: RuntimeProfile,
) -> None:
    @app.before_request
    def authenticate_request():
        if not request.path.startswith("/api/v2"):
            return None

        if not profile.requires_auth:
            local_hash = hashlib.sha256(b"local-session").hexdigest()
            g.saber_identity = SessionIdentity(
                user_id=LOCAL_USER_ID,
                username=LOCAL_USERNAME,
                role="admin",
                session_token_hash=local_hash,
                csrf_token_hash=local_hash,
            )
            return None

        token = request.cookies.get(SESSION_COOKIE_NAME, "")
        identity = repository.authenticate(token)
        if identity is not None:
            g.saber_identity = identity

        if request.path in _PUBLIC_API_PATHS:
            return None
        if identity is None:
            return _error("authentication_required", "请先登录", 401)
        if request.method not in _SAFE_METHODS:
            csrf = request.headers.get("X-CSRF-Token", "")
            if not repository.validate_csrf(identity, csrf):
                return _error("csrf_failed", "请求安全令牌无效，请刷新页面重试", 403)
        return None
