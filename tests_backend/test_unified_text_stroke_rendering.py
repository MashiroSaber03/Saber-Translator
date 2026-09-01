from __future__ import annotations

import numpy as np
from PIL import Image
import pytest

from src.core.config_models import BubbleState
from src.core.rendering import render_bubbles_unified
from src.shared import constants


@pytest.mark.parametrize(
    ("text_direction", "text", "coords", "rotation_angle", "position_offset"),
    [
        (
            "horizontal",
            "测试测试",
            (90, 100, 390, 220),
            0,
            {"x": -13.25, "y": 7.75},
        ),
        (
            "vertical",
            "测试测试",
            (160, 60, 320, 280),
            0,
            {"x": 8.5, "y": -9.25},
        ),
        (
            "horizontal",
            "测试测试",
            (90, 100, 390, 220),
            18,
            {"x": 0.5, "y": -0.25},
        ),
        (
            "vertical",
            "测试测试",
            (160, 60, 320, 280),
            -18,
            {"x": -0.5, "y": 0.25},
        ),
    ],
)
def test_stroke_never_covers_another_character_fill(
    text_direction: str,
    text: str,
    coords: tuple[int, int, int, int],
    rotation_angle: float,
    position_offset: dict[str, float],
) -> None:
    fill_only = Image.new("RGB", (480, 340), "white")
    unified_stroke = Image.new("RGB", (480, 340), "white")
    base = {
        "translated_text": text,
        "coords": coords,
        "font_size": 64,
        "font_family": constants.DEFAULT_FONT_RELATIVE_PATH,
        "text_direction": text_direction,
        "text_color": "#000000",
        "stroke_color": "#FF0000",
        "stroke_width": 16,
        "rotation_angle": rotation_angle,
        "position_offset": position_offset,
    }
    try:
        render_bubbles_unified(
            fill_only,
            [BubbleState(**base, stroke_enabled=False)],
        )
        render_bubbles_unified(
            unified_stroke,
            [BubbleState(**base, stroke_enabled=True)],
        )

        fill_pixels = np.asarray(fill_only)
        stroked_pixels = np.asarray(unified_stroke)
        solid_text = np.all(fill_pixels < 8, axis=2)
        assert solid_text.any()
        assert np.all(stroked_pixels[solid_text] < 8)
    finally:
        fill_only.close()
        unified_stroke.close()


@pytest.mark.parametrize("text_direction", ["horizontal", "vertical"])
def test_zero_width_stroke_keeps_the_fill_only_rendering(text_direction: str) -> None:
    disabled = Image.new("RGB", (320, 240), "white")
    zero_width = Image.new("RGB", (320, 240), "white")
    base = {
        "translated_text": "测试",
        "coords": (60, 40, 260, 200),
        "font_size": 48,
        "font_family": constants.DEFAULT_FONT_RELATIVE_PATH,
        "text_direction": text_direction,
        "text_color": "#000000",
        "stroke_color": "#FF0000",
        "stroke_width": 0,
    }
    try:
        render_bubbles_unified(
            disabled,
            [BubbleState(**base, stroke_enabled=False)],
        )
        render_bubbles_unified(
            zero_width,
            [BubbleState(**base, stroke_enabled=True)],
        )

        assert np.array_equal(np.asarray(disabled), np.asarray(zero_width))
    finally:
        disabled.close()
        zero_width.close()
