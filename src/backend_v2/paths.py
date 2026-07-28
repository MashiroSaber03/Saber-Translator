"""Canonical v2 runtime path resolution.

Only Launcher should resolve the production data root.  API and Worker receive
the resulting absolute path explicitly.  Direct role startup is reserved for
tests and still uses this deterministic resolver.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys


DATA_ROOT_ENV = "SABER_V2_DATA_ROOT"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_data_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    candidate = explicit or os.environ.get(DATA_ROOT_ENV)
    if candidate:
        return Path(candidate).expanduser().resolve()

    if getattr(sys, "frozen", False):
        local_app_data = os.environ.get("LOCALAPPDATA")
        if not local_app_data:
            raise RuntimeError("LOCALAPPDATA is required for packaged v2 startup")
        return (Path(local_app_data) / "SaberTranslator" / "data-v2").resolve()

    return (project_root() / "data-v2").resolve()


def ensure_data_root(path: Path) -> Path:
    resolved = path.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    for relative in (
        "objects",
        "temp/staging",
        "temp/imports",
        "temp/jobs",
        "temp/web-import",
        "chroma",
        "plugins",
        "runtime",
    ):
        (resolved / relative).mkdir(parents=True, exist_ok=True)
    return resolved


def data_root_fingerprint(path: Path) -> str:
    normalized = os.path.normcase(str(path.resolve()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
