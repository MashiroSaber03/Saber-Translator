"""Canonical serialization used by durable v2 records and fingerprints."""

from __future__ import annotations

import json


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
