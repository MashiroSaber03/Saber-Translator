"""Fixed runtime profiles for local and public deployments."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re


PROFILE_ENV = "SABER_V2_PROFILE"
PUBLIC_HOST_ENV = "SABER_V2_PUBLIC_HOST"
PROFILE_NAMES = ("local", "public")
_PUBLIC_HOST = re.compile(r"^[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?$")


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    name: str
    requires_auth: bool
    browser_credentials: bool
    enforce_asset_quota: bool
    allow_plugins: bool
    allow_web_import: bool
    allow_local_providers: bool

_PROFILES = {
    "local": RuntimeProfile(
        name="local",
        requires_auth=False,
        browser_credentials=False,
        enforce_asset_quota=False,
        allow_plugins=True,
        allow_web_import=True,
        allow_local_providers=True,
    ),
    "public": RuntimeProfile(
        name="public",
        requires_auth=True,
        browser_credentials=True,
        enforce_asset_quota=True,
        allow_plugins=False,
        allow_web_import=False,
        allow_local_providers=False,
    ),
}


def resolve_runtime_profile(value: str | None = None) -> RuntimeProfile:
    normalized = (value or os.environ.get(PROFILE_ENV, "local")).strip().lower()
    try:
        return _PROFILES[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported runtime profile: {normalized!r}") from exc


def resolve_public_host(
    profile: RuntimeProfile,
    value: str | None = None,
) -> str | None:
    """Resolve the deployment-owned hostname required by the public profile."""

    if profile.name != "public":
        return None
    normalized = (value or os.environ.get(PUBLIC_HOST_ENV, "")).strip().lower()
    normalized = normalized.rstrip(".")
    if not normalized:
        raise ValueError(f"{PUBLIC_HOST_ENV} is required for the public profile")
    if not _PUBLIC_HOST.fullmatch(normalized) or ".." in normalized:
        raise ValueError(f"{PUBLIC_HOST_ENV} must contain one hostname without a scheme or port")
    return normalized


def validate_profile_bind_host(profile: RuntimeProfile, host: str) -> str:
    """Keep the public origin private to the deployment machine."""

    normalized = host.strip().lower()
    if profile.name == "public" and normalized not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise ValueError("the public profile must bind to a loopback host")
    return host
