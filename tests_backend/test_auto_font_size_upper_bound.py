"""Regression tests for the dynamic upper bound of automatic font sizing."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.rendering import calculate_auto_font_size


def test_auto_font_size_is_not_capped_at_eighty() -> None:
    size = calculate_auto_font_size(
        "这是一段测试文字内容",
        2000,
        2000,
        "horizontal",
    )
    assert size > 80


def test_auto_font_size_scales_with_high_resolution_bubbles() -> None:
    size = calculate_auto_font_size("测试", 8000, 8000, "horizontal")
    assert size > 160


def test_auto_font_size_never_exceeds_bubble_geometry() -> None:
    width, height = 1200, 900
    size = calculate_auto_font_size("短句", width, height, "horizontal")
    assert size <= max(width, height)


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
