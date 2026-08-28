"""Account and minimal administrator HTTP routes."""

from __future__ import annotations

from collections.abc import Mapping

from flask import Blueprint, Response, jsonify, request
from sqlalchemy import Engine

from src.backend_v2.api.request_helpers import (
    json_body,
    required_boolean,
    required_string,
)
from src.backend_v2.auth.constants import CSRF_COOKIE_NAME, SESSION_COOKIE_NAME
from src.backend_v2.auth.credential_broker import (
    CredentialLeaseClient,
    CredentialLeaseUnavailable,
    credential_reference,
)
from src.backend_v2.auth.context import current_identity, current_user_id, require_admin
from src.backend_v2.auth.repository import AuthError, AuthRepository
from src.backend_v2.auth.rate_limit import AuthRateLimited, FailedAuthLimiter
from src.backend_v2.public_policy import PublicUserPolicyRepository
from src.backend_v2.runtime_profile import RuntimeProfile
from src.backend_v2.scheduling_policy import POLICY_KEYS, SchedulingPolicyRepository
from src.backend_v2.settings.validation import validate_credential_secret


def create_auth_blueprint(*, engine: Engine, profile: RuntimeProfile) -> Blueprint:
    if not profile.requires_auth:
        raise ValueError("authentication routes require the public profile")
    blueprint = Blueprint("auth_v2", __name__, url_prefix="/api/v2")
    repository = AuthRepository(engine)
    public_policy = PublicUserPolicyRepository(engine)
    scheduling_policy = SchedulingPolicyRepository(engine)
    limiter = FailedAuthLimiter()

    @blueprint.errorhandler(AuthError)
    def auth_error(error: AuthError):
        status = 401 if error.code in {
            "invalid_credentials",
            "account_disabled",
            "invalid_recovery_code",
        } else 409 if error.code == "username_taken" else 422
        return jsonify({"error": {"code": error.code, "message": str(error)}}), status

    @blueprint.errorhandler(PermissionError)
    def forbidden(error: PermissionError):
        return jsonify(
            {"error": {"code": "admin_required", "message": str(error)}}
        ), 403

    @blueprint.errorhandler(CredentialLeaseUnavailable)
    def credential_lease_unavailable(error: CredentialLeaseUnavailable):
        return jsonify(
            {"error": {"code": "credential_lease_unavailable", "message": str(error)}}
        ), 503

    @blueprint.errorhandler(AuthRateLimited)
    def auth_rate_limited(error: AuthRateLimited):
        response = jsonify(
            {"error": {"code": "auth_rate_limited", "message": str(error)}}
        )
        response.status_code = 429
        response.headers["Retry-After"] = "600"
        return response

    @blueprint.errorhandler(ValueError)
    def validation(error: ValueError):
        return jsonify(
            {"error": {"code": "validation_error", "message": str(error)}}
        ), 422

    def client_ip() -> str:
        return request.remote_addr or "unknown"

    def attempt_key(username: object) -> str:
        return str(username).strip().lower()[:64]

    def require_browser_credentials() -> CredentialLeaseClient:
        if not profile.browser_credentials:
            raise AuthError("browser_credentials_disabled", "当前运行模式不使用浏览器密钥")
        return CredentialLeaseClient.from_environment()

    def session_response(
        user: dict[str, object],
        *,
        recovery_codes: list[str] | None = None,
    ) -> Response:
        token, csrf = repository.create_session(str(user["id"]))
        body: dict[str, object] = {
            "user": user,
            "csrfToken": csrf,
            **repository.usage(str(user["id"])),
        }
        if recovery_codes is not None:
            body["recoveryCodes"] = recovery_codes
        response = jsonify(body)
        response.set_cookie(
            SESSION_COOKIE_NAME,
            token,
            max_age=7 * 24 * 60 * 60,
            secure=True,
            httponly=True,
            samesite="Lax",
            path="/",
        )
        response.set_cookie(
            CSRF_COOKIE_NAME,
            csrf,
            max_age=7 * 24 * 60 * 60,
            secure=True,
            httponly=False,
            samesite="Lax",
            path="/",
        )
        return response

    @blueprint.post("/auth/register")
    def register() -> Response:
        body = json_body(allowed_keys={"username", "password", "inviteCode"})
        username = required_string(body, "username")
        key = attempt_key(username)
        ip = client_ip()
        limiter.check(route="register", client_ip=ip, username=key)
        try:
            invite_code = body.get("inviteCode")
            if invite_code is not None and not isinstance(invite_code, str):
                raise AuthError("invalid_invite", "邀请码格式无效")
            user, recovery = repository.register(
                username=username,
                password=required_string(body, "password"),
                invite_code=invite_code,
            )
        except AuthError:
            limiter.record_failure(route="register", client_ip=ip, username=key)
            raise
        limiter.clear_user(route="register", client_ip=ip, username=key)
        return session_response(user, recovery_codes=recovery)

    @blueprint.post("/auth/login")
    def login() -> Response:
        body = json_body(allowed_keys={"username", "password"})
        username = required_string(body, "username")
        key = attempt_key(username)
        ip = client_ip()
        limiter.check(route="login", client_ip=ip, username=key)
        try:
            user = repository.verify_password(
                username,
                required_string(body, "password"),
            )
        except AuthError:
            limiter.record_failure(route="login", client_ip=ip, username=key)
            raise
        limiter.clear_user(route="login", client_ip=ip, username=key)
        return session_response(user)

    @blueprint.post("/auth/logout")
    def logout() -> Response:
        repository.delete_session(request.cookies.get(SESSION_COOKIE_NAME, ""))
        response = jsonify({"status": "ok"})
        response.delete_cookie(SESSION_COOKIE_NAME, path="/")
        response.delete_cookie(CSRF_COOKIE_NAME, path="/")
        return response

    @blueprint.get("/auth/me")
    def me() -> Response:
        identity = current_identity()
        return jsonify(
            {
                "user": {
                    "id": identity.user_id,
                    "username": identity.username,
                    "role": identity.role,
                },
                **repository.usage(identity.user_id),
            }
        )

    @blueprint.post("/auth/recover")
    def recover() -> Response:
        body = json_body(
            allowed_keys={"username", "recoveryCode", "newPassword"}
        )
        username = required_string(body, "username")
        key = attempt_key(username)
        ip = client_ip()
        limiter.check(route="recover", client_ip=ip, username=key)
        try:
            repository.recover(
                username,
                required_string(body, "recoveryCode"),
                required_string(body, "newPassword"),
            )
        except AuthError:
            limiter.record_failure(route="recover", client_ip=ip, username=key)
            raise
        limiter.clear_user(route="recover", client_ip=ip, username=key)
        return jsonify({"status": "ok"})

    @blueprint.post("/auth/change-password")
    def change_password() -> Response:
        body = json_body(allowed_keys={"currentPassword", "newPassword"})
        repository.change_password(
            current_user_id(),
            required_string(body, "currentPassword"),
            required_string(body, "newPassword"),
        )
        response = jsonify({"status": "ok"})
        response.delete_cookie(SESSION_COOKIE_NAME, path="/")
        response.delete_cookie(CSRF_COOKIE_NAME, path="/")
        return response

    @blueprint.put("/browser-credentials/<domain>/<provider>")
    def put_browser_credential(domain: str, provider: str) -> Response:
        body = json_body(allowed_keys={"secret"})
        secret = body.get("secret")
        if not isinstance(secret, Mapping):
            raise AuthError("invalid_credential", "密钥内容必须是对象")
        try:
            validated = validate_credential_secret(domain, provider, dict(secret))
            require_browser_credentials().put(
                current_user_id(), domain, provider, validated
            )
        except ValueError as exc:
            raise AuthError("invalid_credential", "密钥内容无效") from exc
        return jsonify(
            {
                "status": "loaded",
                "credentialReference": credential_reference(domain, provider),
            }
        )

    @blueprint.delete("/browser-credentials/<domain>/<provider>")
    def delete_browser_credential(domain: str, provider: str) -> Response:
        require_browser_credentials().delete(current_user_id(), domain, provider)
        return jsonify({"status": "deleted"})

    @blueprint.get("/admin/users")
    def list_users() -> Response:
        require_admin()
        return jsonify({"users": repository.list_users()})

    @blueprint.patch("/admin/users/<user_id>/status")
    def update_user_status(user_id: str) -> Response:
        require_admin()
        body = json_body(allowed_keys={"status"})
        repository.set_user_status(user_id, required_string(body, "status"))
        return jsonify({"status": "ok"})

    @blueprint.get("/admin/asset-quota")
    def get_asset_quota() -> Response:
        require_admin()
        return jsonify({"assetQuotaBytes": repository.asset_quota()})

    @blueprint.patch("/admin/asset-quota")
    def update_asset_quota() -> Response:
        require_admin()
        body = json_body(allowed_keys={"assetQuotaBytes"})
        value = body.get("assetQuotaBytes")
        if isinstance(value, bool) or not isinstance(value, int):
            raise AuthError("invalid_quota", "资产额度必须是正整数")
        return jsonify({"assetQuotaBytes": repository.set_asset_quota(value)})

    @blueprint.get("/admin/registration-policy")
    def get_registration_policy() -> Response:
        require_admin()
        return jsonify(
            {
                "registrationRequiresInvite": (
                    repository.registration_requires_invite()
                )
            }
        )

    @blueprint.patch("/admin/registration-policy")
    def update_registration_policy() -> Response:
        require_admin()
        body = json_body(allowed_keys={"registrationRequiresInvite"})
        value = required_boolean(body, "registrationRequiresInvite")
        return jsonify(
            {
                "registrationRequiresInvite": (
                    repository.set_registration_requires_invite(value)
                )
            }
        )

    @blueprint.get("/admin/public-user-policy")
    def get_public_user_policy() -> Response:
        require_admin()
        return jsonify(public_policy.load())

    @blueprint.patch("/admin/public-user-policy")
    def update_public_user_policy() -> Response:
        require_admin()
        body = json_body(allowed_keys={"features", "models", "settings"})
        try:
            saved = public_policy.save(body)
        except ValueError as error:
            raise AuthError("invalid_public_user_policy", str(error)) from error
        return jsonify(saved)

    @blueprint.get("/admin/scheduling-policy")
    def get_scheduling_policy() -> Response:
        require_admin()
        return jsonify(scheduling_policy.overview())

    @blueprint.patch("/admin/scheduling-policy")
    def update_scheduling_policy() -> Response:
        require_admin()
        body = json_body(allowed_keys=POLICY_KEYS)
        try:
            scheduling_policy.save(body)
        except ValueError as error:
            raise AuthError("invalid_scheduling_policy", str(error)) from error
        return jsonify(scheduling_policy.overview())

    @blueprint.get("/admin/invites")
    def list_invites() -> Response:
        require_admin()
        return jsonify({"invites": repository.list_invites()})

    @blueprint.post("/admin/invites")
    def create_invite() -> Response:
        admin = require_admin()
        return jsonify(repository.create_invite(admin.user_id)), 201

    @blueprint.delete("/admin/invites/<invite_id>")
    def revoke_invite(invite_id: str) -> Response:
        require_admin()
        repository.revoke_invite(invite_id)
        return jsonify({"status": "ok"})

    @blueprint.post("/admin/users/<user_id>/recovery-code")
    def create_recovery_code(user_id: str) -> Response:
        require_admin()
        return jsonify({"recoveryCode": repository.create_recovery_code(user_id)})

    return blueprint
