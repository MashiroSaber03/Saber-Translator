"""Geometry derived from the current editable bubble box."""

from __future__ import annotations

import math
from collections.abc import Sequence


def rotated_box_polygon(
    coords: Sequence[int],
    rotation_angle: int | float,
) -> list[list[int]]:
    """Return the four corners of ``coords`` rotated around its center."""

    x1, y1, x2, y2 = coords
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    half_width = (x2 - x1) / 2.0
    half_height = (y2 - y1) / 2.0
    radians = math.radians(float(rotation_angle))
    cosine = math.cos(radians)
    sine = math.sin(radians)

    return [
        [
            int(round(center_x + dx * cosine - dy * sine)),
            int(round(center_y + dx * sine + dy * cosine)),
        ]
        for dx, dy in (
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height),
        )
    ]
