"""Independent application and storage versions from the packaged manifest."""

import json
from pathlib import Path
import re


def parse_version(value: str) -> tuple[int, int, int]:
    if not isinstance(value, str) or not re.fullmatch(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", value):
        raise ValueError(f"Invalid version: {value!r}; expected major.minor.patch")
    return tuple(int(part) for part in value.split("."))


_manifest = json.loads((Path(__file__).resolve().parents[1] / "version.json").read_text(encoding="utf-8"))
APP_VERSION = _manifest["version"]
STORAGE_VERSION = _manifest["storageVersion"]
parse_version(APP_VERSION)
parse_version(STORAGE_VERSION)
