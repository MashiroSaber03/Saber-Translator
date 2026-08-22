"""Fixed runtime profiles for local and public deployments."""

from __future__ import annotations

from dataclasses import dataclass
import os


PROFILE_ENV = "SABER_V2_PROFILE"
PROFILE_NAMES = ("local", "public")


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    name: str
    requires_auth: bool
    browser_credentials: bool
    enforce_asset_quota: bool
    allow_plugins: bool
    allow_web_import: bool
    allow_local_providers: bool
    public_host: str | None = None

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
        public_host="saber.mashirosaber.work",
    ),
}


def resolve_runtime_profile(value: str | None = None) -> RuntimeProfile:
    normalized = (value or os.environ.get(PROFILE_ENV, "local")).strip().lower()
    try:
        return _PROFILES[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported runtime profile: {normalized!r}") from exc
