"""Launcher-owned, loopback-only memory store for public BYOK credentials."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
import secrets
import threading
import time
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlsplit
from urllib.request import Request, urlopen


BROKER_URL_ENV = "SABER_V2_CREDENTIAL_BROKER_URL"
BROKER_TOKEN_ENV = "SABER_V2_CREDENTIAL_BROKER_TOKEN"
DEFAULT_LEASE_SECONDS = 7 * 24 * 60 * 60
MAX_BODY_BYTES = 64 * 1024
_IDENTIFIER = re.compile(r"^[a-zA-Z0-9_.-]{1,96}$")
_REFERENCE_PREFIX = "browser:"


class CredentialLeaseUnavailable(LookupError):
    """The browser key is missing, expired, or the launcher broker restarted."""


def credential_reference(domain: str, provider: str) -> str:
    _validate_identifier(domain, "domain")
    _validate_identifier(provider, "provider")
    return f"{_REFERENCE_PREFIX}{domain}:{provider}"


def parse_credential_reference(value: str) -> tuple[str, str] | None:
    if not value.startswith(_REFERENCE_PREFIX):
        return None
    parts = value[len(_REFERENCE_PREFIX) :].split(":")
    if len(parts) != 2:
        raise CredentialLeaseUnavailable("浏览器密钥引用无效，请重新保存密钥")
    domain, provider = parts
    _validate_identifier(domain, "domain")
    _validate_identifier(provider, "provider")
    return domain, provider


def _validate_identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"invalid credential {label}")
    return value


def _validated_secret(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value or len(value) > 16:
        raise ValueError("credential secret must be a non-empty object")
    result: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not _IDENTIFIER.fullmatch(raw_key):
            raise ValueError("credential secret field is invalid")
        if not isinstance(raw_value, str) or len(raw_value) > 16_384:
            raise ValueError("credential secret value is invalid")
        result[raw_key] = raw_value
    return result


@dataclass(slots=True)
class _Lease:
    secret: dict[str, str]
    expires_at: float


class CredentialLeaseBroker:
    """A tiny HTTP broker shared only by Launcher child processes."""

    def __init__(self, *, lease_seconds: int = DEFAULT_LEASE_SECONDS) -> None:
        self.token = secrets.token_urlsafe(48)
        self._lease_seconds = lease_seconds
        self._leases: dict[tuple[str, str, str], _Lease] = {}
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("credential broker is not running")
        return f"http://127.0.0.1:{self._server.server_port}"

    def start(self) -> None:
        if self._server is not None:
            return
        broker = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "SaberCredentialBroker/1"

            def log_message(self, _format: str, *_args: object) -> None:
                return

            def _reply(self, status: int, payload: Mapping[str, Any]) -> None:
                body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _authorized(self) -> bool:
                expected = f"Bearer {broker.token}"
                supplied = self.headers.get("Authorization", "")
                return secrets.compare_digest(expected, supplied)

            def _key(self) -> tuple[str, str, str] | None:
                path = urlsplit(self.path).path
                prefix = "/v1/credentials/"
                if not path.startswith(prefix):
                    return None
                parts = [unquote(part) for part in path[len(prefix) :].split("/")]
                if len(parts) != 3:
                    return None
                try:
                    return tuple(
                        _validate_identifier(part, "path") for part in parts
                    )  # type: ignore[return-value]
                except ValueError:
                    return None

            def _prepare(self) -> tuple[str, str, str] | None:
                if not self._authorized():
                    self._reply(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                    return None
                key = self._key()
                if key is None:
                    self._reply(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return key

            def do_PUT(self) -> None:  # noqa: N802
                key = self._prepare()
                if key is None:
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= MAX_BODY_BYTES:
                        raise ValueError("credential body size is invalid")
                    payload = json.loads(self.rfile.read(length))
                    secret = _validated_secret(payload.get("secret"))
                except (ValueError, json.JSONDecodeError, AttributeError):
                    self._reply(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "invalid"})
                    return
                with broker._lock:
                    broker._leases[key] = _Lease(
                        secret=secret,
                        expires_at=time.monotonic() + broker._lease_seconds,
                    )
                self._reply(HTTPStatus.OK, {"status": "loaded"})

            def do_GET(self) -> None:  # noqa: N802
                key = self._prepare()
                if key is None:
                    return
                with broker._lock:
                    lease = broker._leases.get(key)
                    if lease is not None and lease.expires_at <= time.monotonic():
                        del broker._leases[key]
                        lease = None
                    secret = dict(lease.secret) if lease is not None else None
                if secret is None:
                    self._reply(HTTPStatus.NOT_FOUND, {"error": "missing"})
                    return
                self._reply(HTTPStatus.OK, {"secret": secret})

            def do_DELETE(self) -> None:  # noqa: N802
                key = self._prepare()
                if key is None:
                    return
                with broker._lock:
                    broker._leases.pop(key, None)
                self._reply(HTTPStatus.OK, {"status": "deleted"})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="saber-credential-broker",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        server, thread = self._server, self._thread
        self._server = None
        self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=2)
        with self._lock:
            self._leases.clear()


class CredentialLeaseClient:
    def __init__(self, url: str, token: str) -> None:
        parsed = urlsplit(url)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port:
            raise ValueError("credential broker must be a loopback HTTP endpoint")
        self.url = url.rstrip("/")
        self.token = token

    @classmethod
    def from_environment(cls) -> "CredentialLeaseClient":
        url = os.environ.get(BROKER_URL_ENV, "")
        token = os.environ.get(BROKER_TOKEN_ENV, "")
        if not url or not token:
            raise CredentialLeaseUnavailable("密钥内存服务未启动，请重启 Saber")
        return cls(url, token)

    def _request(
        self,
        method: str,
        owner_user_id: str,
        domain: str,
        provider: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        for value, label in (
            (owner_user_id, "owner"),
            (domain, "domain"),
            (provider, "provider"),
        ):
            _validate_identifier(value, label)
        path = "/".join(quote(value, safe="") for value in (owner_user_id, domain, provider))
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.url}/v1/credentials/{path}",
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=3) as response:
                raw = response.read()
                return json.loads(raw) if raw else None
        except HTTPError as exc:
            if exc.code == HTTPStatus.NOT_FOUND:
                raise CredentialLeaseUnavailable(
                    "浏览器中的 API Key 尚未加载或已失效，请打开设置重新保存"
                ) from exc
            raise CredentialLeaseUnavailable("密钥内存服务暂时不可用") from exc
        except (OSError, URLError, ValueError) as exc:
            raise CredentialLeaseUnavailable("密钥内存服务暂时不可用") from exc

    def put(
        self,
        owner_user_id: str,
        domain: str,
        provider: str,
        secret: Mapping[str, str],
    ) -> None:
        self._request(
            "PUT", owner_user_id, domain, provider, {"secret": _validated_secret(secret)}
        )

    def delete(self, owner_user_id: str, domain: str, provider: str) -> None:
        self._request("DELETE", owner_user_id, domain, provider)

    def resolve(
        self, owner_user_id: str, domain: str, provider: str
    ) -> dict[str, str]:
        payload = self._request("GET", owner_user_id, domain, provider)
        if not isinstance(payload, Mapping):
            raise CredentialLeaseUnavailable("浏览器密钥响应无效")
        return _validated_secret(payload.get("secret"))


def resolve_credential_reference(value: str, owner_user_id: str) -> dict[str, str] | None:
    parsed = parse_credential_reference(value)
    if parsed is None:
        return None
    domain, provider = parsed
    return CredentialLeaseClient.from_environment().resolve(
        owner_user_id, domain, provider
    )
