"""The packaged release version is also the required storage version."""

import json
from pathlib import Path
import re


def parse_version(value: str) -> tuple[int, int, int]:
    if not isinstance(value, str) or not re.fullmatch(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", value):
        raise ValueError(f"Invalid release version: {value!r}; expected major.minor.patch")
    return tuple(int(part) for part in value.split("."))


APP_VERSION = json.loads((Path(__file__).resolve().parents[1] / "version.json").read_text(encoding="utf-8"))["version"]
parse_version(APP_VERSION)
