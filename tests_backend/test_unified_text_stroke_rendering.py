from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
import pytest

from src.core.config_models import BubbleState
from src.core.rendering import _draw_bubble_text_pass, get_font, render_bubbles_unified
from src.shared import constants


@pytest.mark.parametrize(
    ("text_direction", "text", "rotation_angle"),
    [
        ("horizontal", "测试TEST", 0),
        ("vertical", "测试", 0),
        ("horizontal", "测试TEST", 18),
        ("vertical", "测试", -18),
        ("vertical", "<H>AB</H>", 0),
        ("vertical", "……", 0),
        ("vertical", "⁉", 0),
        ("vertical", "ー", 0),
    ],
)
def test_decimal_stroke_widths_produce_progressively_wider_strokes(
    text_direction: str, text: str, rotation_angle: float,
) -> None:
    red_coverage = []
    for width in (0, 0.1, 0.2, 0.5, 1, 1.1, 1.2):
        with Image.new("RGB", (480, 340), "white") as image:
            render_bubbles_unified(image, [BubbleState(
                translated_text=text,
                coords=(100, 60, 380, 280),
                font_size=48,
                font_family=constants.DEFAULT_FONT_RELATIVE_PATH,
                text_direction=text_direction,
                text_color="#000000",
                stroke_enabled=True,
                stroke_color="#FF0000",
                stroke_width=width,
                rotation_angle=rotation_angle,
            )])
            pixels = np.asarray(image, dtype=np.int32)
            red_coverage.append(int((pixels[:, :, 0] - pixels[:, :, 1]).sum()))

    assert red_coverage[0] == 0
    assert all(after > before for before, after in zip(red_coverage, red_coverage[1:])), red_coverage


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


@pytest.mark.parametrize("rotation_angle", [0, 18, -35, 90])
@pytest.mark.parametrize("stroke_width", [0, 0.5, 6])
@pytest.mark.parametrize("alignment", ["start", "center", "end"])
@pytest.mark.parametrize("direction,text", [
    ("horizontal", "测试测试"),
    ("vertical", "测试测试"),
    ("vertical", "<H>AB</H>……⁉ー测试"),
])
def test_overflow_matches_unclipped_full_page_render(direction, text, stroke_width, rotation_angle, alignment):
    """整页直接绘制作为参照，覆盖框外正文、描边与特殊字符贴图。"""
    state = BubbleState(
        translated_text=text, coords=(440, 440, 560, 560), font_size=80,
        font_family=constants.DEFAULT_FONT_RELATIVE_PATH,
        text_direction=direction, text_color="#000000",
        stroke_enabled=stroke_width > 0, stroke_color="#FF0000",
        stroke_width=stroke_width, rotation_angle=rotation_angle,
        position_offset={"x": -13.25, "y": 7.75},
        inline_align=alignment, block_align=alignment,
    )
    font = get_font(state.font_family, state.font_size)
    direct = not stroke_width and not rotation_angle
    with Image.new("RGB" if direct else "RGBA", (1000, 1000),
                   "white" if direct else (0, 0, 0, 0)) as reference_layer:
        draw = ImageDraw.Draw(reference_layer)
        for fill, width in ((state.stroke_color, stroke_width), (state.text_color, 0)):
            if fill == state.stroke_color and not stroke_width:
                continue
            _draw_bubble_text_pass(
                draw, text, font, state, 426.75, 447.75, 120, 120,
                bubble_index=0, fill=fill, stroke_width=width,
            )
        with reference_layer.rotate(
            -rotation_angle, center=(486.75, 507.75),
            resample=Image.Resampling.BICUBIC,
        ) as rotated, Image.new("RGB", (1000, 1000), "white") as expected, \
                Image.new("RGB", (1000, 1000), "white") as actual:
            expected.paste(rotated, (0, 0), None if direct else rotated)
            render_bubbles_unified(actual, [state])
            expected_pixels = np.asarray(expected).astype(np.int16)
            actual_pixels = np.asarray(actual).astype(np.int16)
            ink = np.any(expected_pixels < 240, axis=2)
            outside = ink.copy()
            outside[437:578, 416:557] = False
            assert outside.any(), "用例必须确实包含旧图层边界以外的文字"
            # 仿射变换的浮点舍入允许少量抗锯齿差异，但不能丢失文字。
            difference = np.abs(actual_pixels - expected_pixels)
            assert difference[ink].mean() < 1
            assert np.count_nonzero(np.any(difference > 16, axis=2)) < ink.sum() * 0.01


@pytest.mark.parametrize("angle", [-30, 30, 90])
@pytest.mark.parametrize("stroke_width", [0, 0.5])
def test_rotation_preserves_text_that_enters_page_from_outside(angle, stroke_width):
    """先完整旋转再裁到页面；不能提前丢掉原本位于页面外的文字。"""
    state = BubbleState(
        translated_text="<H>AB</H>……⁉ー测试", coords=(240, 240, 360, 360),
        font_size=80, font_family=constants.DEFAULT_FONT_RELATIVE_PATH,
        text_direction="vertical", text_color="#000000",
        stroke_enabled=stroke_width > 0, stroke_color="#FF0000",
        stroke_width=stroke_width, rotation_angle=angle,
        inline_align="center", block_align="center",
    )
    font = get_font(state.font_family, state.font_size)
    with Image.new("RGBA", (1200, 1200)) as reference:
        draw = ImageDraw.Draw(reference)
        passes = [(state.stroke_color, stroke_width)] if stroke_width else []
        passes.append((state.text_color, 0))
        for fill, width in passes:
            _draw_bubble_text_pass(
                draw, state.translated_text, font, state, 540, 540, 120, 120,
                bubble_index=0, fill=fill, stroke_width=width,
            )
        assert reference.getbbox()[0] < 300  # 未旋转文字确实超出了最终页面。
        with reference.rotate(-angle, resample=Image.Resampling.BICUBIC) as rotated, \
                rotated.crop((300, 300, 900, 900)) as cropped, \
                Image.new("RGB", (600, 600), "white") as expected, \
                Image.new("RGB", (600, 600), "white") as actual:
            expected.paste(cropped, (0, 0), cropped)
            render_bubbles_unified(actual, [state])
            expected_pixels = np.asarray(expected).astype(np.int16)
            difference = np.abs(np.asarray(actual).astype(np.int16) - expected_pixels)
            ink = np.any(expected_pixels < 240, axis=2)
            assert ink.any()
            assert difference[ink].mean() < 1
            assert np.count_nonzero(np.any(difference > 16, axis=2)) < ink.sum() * 0.01
