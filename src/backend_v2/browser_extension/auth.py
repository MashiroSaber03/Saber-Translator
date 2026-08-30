"""Authentication and scoped result capabilities for the browser extension."""

from __future__ import annotations

from dataclasses import dataclass, field
import base64
import hashlib
import hmac
import time


BROWSER_EXTENSION_ENABLED_ENV = "SABER_BROWSER_EXTENSION_ENABLED"
BROWSER_EXTENSION_TOKEN_ENV = "SABER_BROWSER_EXTENSION_TOKEN"
CAPABILITY_TTL_SECONDS = 15 * 60


@dataclass(frozen=True, slots=True)
class BrowserExtensionAccess:
    enabled: bool
    token: str = field(repr=False)

    def __post_init__(self) -> None:
        if self.enabled and not 32 <= len(self.token) <= 200:
            raise ValueError(
                "enabled browser extension integration requires a "
                "32-200 character token"
            )

    def valid_token(self, candidate: str) -> bool:
        return bool(self.enabled and self.token and candidate) and hmac.compare_digest(
            self.token,
            candidate,
        )

    def sign_result(
        self,
        *,
        session_id: str,
        browser_page_id: str,
        asset_id: str,
        expires_at: int | None = None,
    ) -> tuple[int, str]:
        if not self.enabled or not self.token:
            raise RuntimeError("browser extension integration is disabled")
        expiry = expires_at or int(time.time()) + CAPABILITY_TTL_SECONDS
        payload = self._capability_payload(
            session_id=session_id,
            browser_page_id=browser_page_id,
            asset_id=asset_id,
            expires_at=expiry,
        )
        digest = hmac.new(
            self.token.encode("utf-8"),
            payload,
            hashlib.sha256,
        ).digest()
        signature = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
        return expiry, signature

    def verify_result(
        self,
        *,
        session_id: str,
        browser_page_id: str,
        asset_id: str,
        expires_at: int,
        signature: str,
    ) -> bool:
        if expires_at < int(time.time()):
            return False
        try:
            expected_expiry, expected = self.sign_result(
                session_id=session_id,
                browser_page_id=browser_page_id,
                asset_id=asset_id,
                expires_at=expires_at,
            )
        except RuntimeError:
            return False
        return expected_expiry == expires_at and hmac.compare_digest(
            expected,
            signature,
        )

    @staticmethod
    def _capability_payload(
        *,
        session_id: str,
        browser_page_id: str,
        asset_id: str,
        expires_at: int,
    ) -> bytes:
        return (
            f"{session_id}\n{browser_page_id}\n{asset_id}\n{expires_at}"
        ).encode("utf-8")
