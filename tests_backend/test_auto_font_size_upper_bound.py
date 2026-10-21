"""Regression tests for the dynamic upper bound of automatic font sizing.

These tests verify that the old hard-coded 80px cap no longer prevents the
algorithm from choosing an appropriate size for large bubbles. They do not
assert a fixed expected size: the result always depends on the bubble geometry,
the text length, and the text direction.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.rendering import calculate_auto_font_size


def test_auto_font_size_is_not_capped_at_eighty() -> None:
    """A bubble that used to hit the hard cap may now grow beyond 80px."""

    size = calculate_auto_font_size(
        "这是一段测试文字内容",
        2000,
        2000,
        "horizontal",
    )
    assert size > 80


def test_auto_font_size_can_exceed_the_old_cap_on_large_bubbles() -> None:
    """8K is only an example: the old 80px cap should no longer bind.

    This does not claim that 160 is a lower bound for 8K pages. The actual
    size is still chosen from the bubble geometry and the text length.
    """

    size = calculate_auto_font_size("测试", 8000, 8000, "horizontal")
    assert size > 80


def test_auto_font_size_respects_the_geometric_upper_bound() -> None:
    """The result must never exceed the safe upper bound derived from geometry."""

    width, height = 1200, 900
    size = calculate_auto_font_size("短句", width, height, "horizontal")
    assert size <= min(width, height / 1.05) + 1


def test_auto_font_size_keeps_small_bubbles_small() -> None:
    size = calculate_auto_font_size("测", 40, 40, "horizontal")
    assert size <= 40


def test_explicit_max_size_still_works_for_compatibility() -> None:
    size = calculate_auto_font_size(
        "测试",
        8000,
        8000,
        "horizontal",
        max_size=80,
    )
    assert size == 80
