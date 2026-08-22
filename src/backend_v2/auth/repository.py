"""Small SQLite-backed account, session, invite, and quota repository."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import hashlib
import re
import secrets
import string
from typing import Any
import uuid

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from sqlalchemy import Engine, and_, case, delete, func, insert, select, update

from src.backend_v2.storage.database import immediate_transaction
from src.backend_v2.storage.schema import (
    CURRENT_JOB_STATUSES,
    DEFAULT_ASSET_QUOTA_BYTES,
    assets,
    invite_codes,
    jobs,
    platform_config,
    recovery_codes,
    sessions,
    users,
)
from src.backend_v2.storage.seeding import seed_user_records_in_connection
from src.backend_v2.timestamps import iso_utc, utcnow


USERNAME_RE = re.compile(r"^[a-z0-9_-]{3,32}$")
PASSWORD_MIN_LENGTH = 10
PASSWORD_MAX_LENGTH = 128
SESSION_DAYS = 7
INVITE_DAYS = 7
RECOVERY_CODE_COUNT = 8
_PASSWORDS = PasswordHasher()
_CODE_ALPHABET = string.ascii_uppercase + string.digits


class AuthError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class SessionIdentity:
    user_id: str
    username: str
    role: str
    session_token_hash: str
    csrf_token_hash: str


def normalize_username(value: str) -> str:
    normalized = value.strip().lower()
    if not USERNAME_RE.fullmatch(normalized):
        raise AuthError(
            "invalid_username",
            "用户名必须为 3-32 位小写字母、数字、下划线或连字符",
        )
    return normalized


def validate_password(value: str) -> str:
    if not isinstance(value, str) or not (
        PASSWORD_MIN_LENGTH <= len(value) <= PASSWORD_MAX_LENGTH
    ):
        raise AuthError(
            "invalid_password",
            f"密码长度必须为 {PASSWORD_MIN_LENGTH}-{PASSWORD_MAX_LENGTH} 位",
        )
    return value


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_code(value: str) -> str:
    return value.strip().upper()


def _new_code(groups: int = 4, width: int = 4) -> str:
    return "-".join(
        "".join(secrets.choice(_CODE_ALPHABET) for _ in range(width))
        for _ in range(groups)
    )


class AuthRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def create_admin(self, username: str, password: str) -> dict[str, Any]:
        username = normalize_username(username)
        password = validate_password(password)
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            existing = connection.execute(
                select(users.c.id).where(users.c.username == username)
            ).scalar_one_or_none()
            if existing is not None:
                raise AuthError("username_taken", "用户名已存在")
            user_id = str(uuid.uuid4())
            connection.execute(
                insert(users).values(
                    id=user_id,
                    username=username,
                    password_hash=_PASSWORDS.hash(password),
                    role="admin",
                    status="active",
                    created_at=now,
                    updated_at=now,
                )
            )
            seed_user_records_in_connection(connection, user_id)
        return {"id": user_id, "username": username, "role": "admin"}

    def create_invite(self, created_by_user_id: str) -> dict[str, Any]:
        code = _new_code()
        now = utcnow()
        expires_at = now + timedelta(days=INVITE_DAYS)
        with self.engine.begin() as connection:
            connection.execute(
                insert(invite_codes).values(
                    code_hash=_hash(code),
                    created_by_user_id=created_by_user_id,
                    expires_at=expires_at,
                    created_at=now,
                )
            )
        return {"code": code, "expiresAt": iso_utc(expires_at)}

    def list_invites(self) -> list[dict[str, Any]]:
        now = utcnow()
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(
                    invite_codes.c.code_hash,
                    invite_codes.c.expires_at,
                    invite_codes.c.used_at,
                    invite_codes.c.revoked_at,
                    invite_codes.c.created_at,
                    users.c.username.label("used_by_username"),
                )
                .outerjoin(users, users.c.id == invite_codes.c.used_by_user_id)
                .order_by(invite_codes.c.created_at.desc())
            ).mappings()
            return [
                {
                    "id": str(row["code_hash"])[:12],
                    "status": (
                        "revoked"
                        if row["revoked_at"] is not None
                        else "used"
                        if row["used_at"] is not None
                        else "expired"
                        if row["expires_at"] <= now
                        else "active"
                    ),
                    "expiresAt": iso_utc(row["expires_at"]),
                    "usedAt": iso_utc(row["used_at"]),
                    "usedBy": row["used_by_username"],
                    "createdAt": iso_utc(row["created_at"]),
                }
                for row in rows
            ]

    def revoke_invite(self, invite_id: str) -> None:
        now = utcnow()
        with self.engine.begin() as connection:
            rows = list(
                connection.execute(
                    select(invite_codes.c.code_hash).where(
                        invite_codes.c.code_hash.like(f"{invite_id}%")
                    )
                ).scalars()
            )
            if len(rows) != 1:
                raise AuthError("invite_not_found", "邀请码不存在")
            connection.execute(
                update(invite_codes)
                .where(invite_codes.c.code_hash == rows[0])
                .values(revoked_at=now)
            )

    def register(
        self,
        *,
        username: str,
        password: str,
        invite_code: str | None = None,
    ) -> tuple[dict[str, Any], list[str]]:
        username = normalize_username(username)
        password = validate_password(password)
        now = utcnow()
        raw_recovery_codes = [_new_code(groups=3) for _ in range(RECOVERY_CODE_COUNT)]
        with immediate_transaction(self.engine) as connection:
            requires_invite = bool(
                connection.execute(
                    select(platform_config.c.registration_requires_invite).where(
                        platform_config.c.singleton_id == 1
                    )
                ).scalar_one()
            )
            invite_hash: str | None = None
            if requires_invite:
                if not isinstance(invite_code, str) or not invite_code.strip():
                    raise AuthError("invite_required", "当前注册必须使用邀请码")
                invite_hash = _hash(_normalize_code(invite_code))
                invite = connection.execute(
                    select(
                        invite_codes.c.used_at,
                        invite_codes.c.revoked_at,
                        invite_codes.c.expires_at,
                    ).where(invite_codes.c.code_hash == invite_hash)
                ).mappings().one_or_none()
                if (
                    invite is None
                    or invite["used_at"] is not None
                    or invite["revoked_at"] is not None
                    or invite["expires_at"] <= now
                ):
                    raise AuthError("invalid_invite", "邀请码无效、已使用或已过期")
            if connection.execute(
                select(users.c.id).where(users.c.username == username)
            ).scalar_one_or_none() is not None:
                raise AuthError("username_taken", "用户名已存在")
            user_id = str(uuid.uuid4())
            connection.execute(
                insert(users).values(
                    id=user_id,
                    username=username,
                    password_hash=_PASSWORDS.hash(password),
                    role="user",
                    status="active",
                    created_at=now,
                    updated_at=now,
                )
            )
            connection.execute(
                insert(recovery_codes),
                [
                    {
                        "id": str(uuid.uuid4()),
                        "user_id": user_id,
                        "code_hash": _hash(code),
                        "created_at": now,
                    }
                    for code in raw_recovery_codes
                ],
            )
            if invite_hash is not None:
                connection.execute(
                    update(invite_codes)
                    .where(invite_codes.c.code_hash == invite_hash)
                    .values(used_by_user_id=user_id, used_at=now)
                )
            seed_user_records_in_connection(connection, user_id)
        return (
            {"id": user_id, "username": username, "role": "user"},
            raw_recovery_codes,
        )

    def verify_password(self, username: str, password: str) -> dict[str, Any]:
        try:
            username = normalize_username(username)
        except AuthError as exc:
            raise AuthError("invalid_credentials", "用户名或密码错误") from exc
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    users.c.id,
                    users.c.username,
                    users.c.password_hash,
                    users.c.role,
                    users.c.status,
                ).where(users.c.username == username)
            ).mappings().one_or_none()
        if row is None or row["password_hash"] is None:
            raise AuthError("invalid_credentials", "用户名或密码错误")
        if row["status"] != "active":
            raise AuthError("account_disabled", "账号已被禁用")
        try:
            valid = _PASSWORDS.verify(str(row["password_hash"]), password)
        except (VerifyMismatchError, InvalidHashError):
            valid = False
        if not valid:
            raise AuthError("invalid_credentials", "用户名或密码错误")
        if _PASSWORDS.check_needs_rehash(str(row["password_hash"])):
            with self.engine.begin() as connection:
                connection.execute(
                    update(users)
                    .where(users.c.id == row["id"])
                    .values(password_hash=_PASSWORDS.hash(password), updated_at=utcnow())
                )
        return {"id": row["id"], "username": row["username"], "role": row["role"]}

    def create_session(self, user_id: str) -> tuple[str, str]:
        token = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(32)
        now = utcnow()
        with self.engine.begin() as connection:
            connection.execute(
                insert(sessions).values(
                    token_hash=_hash(token),
                    user_id=user_id,
                    csrf_token_hash=_hash(csrf),
                    expires_at=now + timedelta(days=SESSION_DAYS),
                    created_at=now,
                )
            )
        return token, csrf

    def authenticate(self, token: str) -> SessionIdentity | None:
        if not token:
            return None
        token_hash = _hash(token)
        now = utcnow()
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    sessions.c.token_hash,
                    sessions.c.csrf_token_hash,
                    sessions.c.expires_at,
                    users.c.id,
                    users.c.username,
                    users.c.role,
                    users.c.status,
                )
                .join(users, users.c.id == sessions.c.user_id)
                .where(sessions.c.token_hash == token_hash)
            ).mappings().one_or_none()
        if row is None:
            return None
        if row["expires_at"] <= now or row["status"] != "active":
            with self.engine.begin() as connection:
                connection.execute(
                    delete(sessions).where(sessions.c.token_hash == token_hash)
                )
            return None
        return SessionIdentity(
            user_id=str(row["id"]),
            username=str(row["username"]),
            role=str(row["role"]),
            session_token_hash=token_hash,
            csrf_token_hash=str(row["csrf_token_hash"]),
        )

    def validate_csrf(self, identity: SessionIdentity, token: str) -> bool:
        return bool(token) and secrets.compare_digest(
            identity.csrf_token_hash,
            _hash(token),
        )

    def delete_session(self, token: str) -> None:
        if not token:
            return
        with self.engine.begin() as connection:
            connection.execute(
                delete(sessions).where(sessions.c.token_hash == _hash(token))
            )

    def revoke_user_sessions(self, user_id: str) -> None:
        with self.engine.begin() as connection:
            connection.execute(delete(sessions).where(sessions.c.user_id == user_id))

    def recover(self, username: str, recovery_code: str, new_password: str) -> None:
        username = normalize_username(username)
        new_password = validate_password(new_password)
        code_hash = _hash(_normalize_code(recovery_code))
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            row = connection.execute(
                select(users.c.id, recovery_codes.c.id.label("recovery_id"))
                .join(recovery_codes, recovery_codes.c.user_id == users.c.id)
                .where(
                    users.c.username == username,
                    users.c.status == "active",
                    recovery_codes.c.code_hash == code_hash,
                    recovery_codes.c.used_at.is_(None),
                )
            ).mappings().one_or_none()
            if row is None:
                raise AuthError("invalid_recovery_code", "恢复码无效或已使用")
            connection.execute(
                update(recovery_codes)
                .where(recovery_codes.c.id == row["recovery_id"])
                .values(used_at=now)
            )
            connection.execute(
                update(users)
                .where(users.c.id == row["id"])
                .values(password_hash=_PASSWORDS.hash(new_password), updated_at=now)
            )
            connection.execute(delete(sessions).where(sessions.c.user_id == row["id"]))

    def change_password(self, user_id: str, current: str, new: str) -> None:
        new = validate_password(new)
        with self.engine.connect() as connection:
            row = connection.execute(
                select(users.c.username).where(users.c.id == user_id)
            ).scalar_one()
        self.verify_password(str(row), current)
        with self.engine.begin() as connection:
            connection.execute(
                update(users)
                .where(users.c.id == user_id)
                .values(password_hash=_PASSWORDS.hash(new), updated_at=utcnow())
            )
            connection.execute(delete(sessions).where(sessions.c.user_id == user_id))

    def usage(self, user_id: str) -> dict[str, int]:
        with self.engine.connect() as connection:
            quota_bytes = int(
                connection.execute(
                    select(platform_config.c.asset_quota_bytes).where(
                        platform_config.c.singleton_id == 1
                    )
                ).scalar_one()
            )
            used = int(
                connection.execute(
                    select(func.coalesce(func.sum(assets.c.byte_size), 0)).where(
                        assets.c.owner_user_id == user_id,
                        assets.c.integrity_status == "ok",
                    )
                ).scalar_one()
            )
        return {
            "assetUsageBytes": used,
            "assetQuotaBytes": quota_bytes or DEFAULT_ASSET_QUOTA_BYTES,
        }

    def list_users(self) -> list[dict[str, Any]]:
        asset_usage = (
            select(
                assets.c.owner_user_id.label("owner_user_id"),
                func.sum(assets.c.byte_size).label("usage"),
            )
            .where(assets.c.integrity_status == "ok")
            .group_by(assets.c.owner_user_id)
            .subquery()
        )
        job_stats = (
            select(
                jobs.c.owner_user_id.label("owner_user_id"),
                func.sum(
                    case((jobs.c.status.in_(CURRENT_JOB_STATUSES), 1), else_=0)
                ).label("active_count"),
                func.sum(case((jobs.c.status == "queued", 1), else_=0)).label(
                    "queued_count"
                ),
                func.sum(
                    case((jobs.c.status == "interrupted", 1), else_=0)
                ).label("interrupted_count"),
                func.sum(
                    case((jobs.c.status == "completed", 1), else_=0)
                ).label("completed_count"),
                func.sum(
                    case(
                        (
                            jobs.c.status.in_(("completed_with_errors", "failed")),
                            1,
                        ),
                        else_=0,
                    )
                ).label("issue_count"),
                func.max(
                    case(
                        (jobs.c.status.in_(CURRENT_JOB_STATUSES), jobs.c.kind),
                        else_=None,
                    )
                ).label("current_kind"),
                func.max(
                    case(
                        (
                            jobs.c.status.in_(CURRENT_JOB_STATUSES),
                            func.coalesce(jobs.c.started_at, jobs.c.created_at),
                        ),
                        else_=None,
                    )
                ).label("current_started_at"),
                func.max(jobs.c.updated_at).label("last_task_at"),
            )
            .group_by(jobs.c.owner_user_id)
            .subquery()
        )
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(
                    users.c.id,
                    users.c.username,
                    users.c.role,
                    users.c.status,
                    users.c.created_at,
                    func.coalesce(asset_usage.c.usage, 0).label("usage"),
                    func.coalesce(job_stats.c.active_count, 0).label("active_count"),
                    func.coalesce(job_stats.c.queued_count, 0).label("queued_count"),
                    func.coalesce(job_stats.c.interrupted_count, 0).label(
                        "interrupted_count"
                    ),
                    func.coalesce(job_stats.c.completed_count, 0).label(
                        "completed_count"
                    ),
                    func.coalesce(job_stats.c.issue_count, 0).label("issue_count"),
                    job_stats.c.current_kind,
                    job_stats.c.current_started_at,
                    job_stats.c.last_task_at,
                )
                .outerjoin(asset_usage, asset_usage.c.owner_user_id == users.c.id)
                .outerjoin(job_stats, job_stats.c.owner_user_id == users.c.id)
                .where(users.c.password_hash.is_not(None))
                .order_by(users.c.created_at)
            ).mappings()
            asset_quota = int(
                connection.execute(
                    select(platform_config.c.asset_quota_bytes).where(
                        platform_config.c.singleton_id == 1
                    )
                ).scalar_one()
            )
            return [
                {
                    "id": str(row["id"]),
                    "username": str(row["username"]),
                    "role": str(row["role"]),
                    "status": str(row["status"]),
                    "assetUsageBytes": int(row["usage"]),
                    "assetQuotaBytes": asset_quota,
                    "createdAt": iso_utc(row["created_at"]),
                    "taskStatus": (
                        "active"
                        if int(row["active_count"]) > 0
                        else "queued"
                        if int(row["queued_count"]) > 0
                        else "interrupted"
                        if int(row["interrupted_count"]) > 0
                        else "idle"
                    ),
                    "activeTaskCount": int(row["active_count"]),
                    "queuedTaskCount": int(row["queued_count"]),
                    "interruptedTaskCount": int(row["interrupted_count"]),
                    "completedTaskCount": int(row["completed_count"]),
                    "issueTaskCount": int(row["issue_count"]),
                    "currentTaskKind": (
                        str(row["current_kind"])
                        if row["current_kind"] is not None
                        else None
                    ),
                    "currentTaskStartedAt": (
                        iso_utc(row["current_started_at"])
                        if row["current_started_at"] is not None
                        else None
                    ),
                    "lastTaskAt": (
                        iso_utc(row["last_task_at"])
                        if row["last_task_at"] is not None
                        else None
                    ),
                }
                for row in rows
            ]

    def asset_quota(self) -> int:
        with self.engine.connect() as connection:
            return int(
                connection.execute(
                    select(platform_config.c.asset_quota_bytes).where(
                        platform_config.c.singleton_id == 1
                    )
                ).scalar_one()
            )

    def registration_requires_invite(self) -> bool:
        with self.engine.connect() as connection:
            return bool(
                connection.execute(
                    select(platform_config.c.registration_requires_invite).where(
                        platform_config.c.singleton_id == 1
                    )
                ).scalar_one()
            )

    def set_registration_requires_invite(self, value: bool) -> bool:
        if not isinstance(value, bool):
            raise AuthError("invalid_registration_policy", "注册设置必须是布尔值")
        with self.engine.begin() as connection:
            connection.execute(
                update(platform_config)
                .where(platform_config.c.singleton_id == 1)
                .values(registration_requires_invite=value, updated_at=utcnow())
            )
        return value

    def set_asset_quota(self, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise AuthError("invalid_quota", "资产额度必须是正整数")
        with self.engine.begin() as connection:
            connection.execute(
                update(platform_config)
                .where(platform_config.c.singleton_id == 1)
                .values(asset_quota_bytes=value, updated_at=utcnow())
            )
        return value

    def set_user_status(self, user_id: str, status: str) -> None:
        if status not in {"active", "disabled"}:
            raise AuthError("invalid_status", "用户状态无效")
        with immediate_transaction(self.engine) as connection:
            row = connection.execute(
                select(users.c.role).where(users.c.id == user_id)
            ).scalar_one_or_none()
            if row is None:
                raise AuthError("user_not_found", "用户不存在")
            if row == "admin" and status == "disabled":
                raise AuthError("admin_disable_forbidden", "不能禁用管理员账号")
            connection.execute(
                update(users)
                .where(users.c.id == user_id)
                .values(status=status, updated_at=utcnow())
            )
            if status == "disabled":
                connection.execute(delete(sessions).where(sessions.c.user_id == user_id))

    def create_recovery_code(self, user_id: str) -> str:
        code = _new_code(groups=3)
        with self.engine.begin() as connection:
            if connection.execute(
                select(users.c.id).where(users.c.id == user_id)
            ).scalar_one_or_none() is None:
                raise AuthError("user_not_found", "用户不存在")
            connection.execute(
                insert(recovery_codes).values(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    code_hash=_hash(code),
                    created_at=utcnow(),
                )
            )
        return code
