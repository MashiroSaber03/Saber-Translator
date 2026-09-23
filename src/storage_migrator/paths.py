"""Filesystem spelling only; never persist Windows extended paths as identities."""

import os
from pathlib import Path


def filesystem_path(path: Path) -> Path:
    if os.name != "nt":
        return path
    absolute = os.path.abspath(path)
    if absolute.startswith("\\\\?\\"):
        return Path(absolute)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)
