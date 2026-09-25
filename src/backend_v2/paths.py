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


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_data_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "data-v2"

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
        "logs",
    ):
        (resolved / relative).mkdir(parents=True, exist_ok=True)
    return resolved


def data_root_fingerprint(path: Path) -> str:
    normalized = os.path.normcase(str(path.resolve()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
