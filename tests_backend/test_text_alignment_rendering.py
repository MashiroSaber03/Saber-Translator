from __future__ import annotations

from PIL import Image, ImageChops, ImageDraw
import pytest

from src.core.config_models import BubbleState
from src.core.rendering import (
    draw_multiline_text_horizontal,
    draw_multiline_text_vertical,
    get_font,
)
from src.shared import constants


CANVAS_SIZE = (300, 230)
BOX_X = 50
BOX_Y = 30
BOX_WIDTH = 180
BOX_HEIGHT = 150
FONT_SIZE = 28


def _ink_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    background = Image.new("RGB", image.size, "white")
    bbox = ImageChops.difference(image, background).getbbox()
    assert bbox is not None
    return bbox


def _render_horizontal(
    *,
    inline_align: str,
    block_align: str,
) -> tuple[int, int, int, int]:
    image = Image.new("RGB", CANVAS_SIZE, "white")
    draw_multiline_text_horizontal(
        ImageDraw.Draw(image),
        "TEST",
        get_font(constants.DEFAULT_FONT_RELATIVE_PATH, FONT_SIZE),
        BOX_X,
        BOX_Y,
        BOX_WIDTH,
        BOX_HEIGHT,
        fill="#000000",
        stroke_enabled=False,
        font_family_path=constants.DEFAULT_FONT_RELATIVE_PATH,
        inline_align=inline_align,
        block_align=block_align,
    )
    return _ink_bbox(image)


def _render_vertical(
    *,
    inline_align: str,
    block_align: str,
) -> tuple[int, int, int, int]:
    image = Image.new("RGB", CANVAS_SIZE, "white")
    draw_multiline_text_vertical(
        ImageDraw.Draw(image),
        "测试",
        get_font(constants.DEFAULT_FONT_RELATIVE_PATH, FONT_SIZE),
        BOX_X + BOX_WIDTH,
        BOX_Y,
        BOX_WIDTH,
        BOX_HEIGHT,
        fill="#000000",
        stroke_enabled=False,
        font_family_path=constants.DEFAULT_FONT_RELATIVE_PATH,
        inline_align=inline_align,
        block_align=block_align,
    )
    return _ink_bbox(image)


def test_horizontal_alignment_axes_move_independently() -> None:
    inline_positions = [
        _render_horizontal(inline_align=align, block_align="start")[0]
        for align in ("start", "center", "end")
    ]
    block_positions = [
        _render_horizontal(inline_align="start", block_align=align)[1]
        for align in ("start", "center", "end")
    ]

    assert inline_positions[0] < inline_positions[1] < inline_positions[2]
    assert block_positions[0] < block_positions[1] < block_positions[2]


def test_vertical_alignment_axes_move_independently() -> None:
    inline_positions = [
        _render_vertical(inline_align=align, block_align="start")[1]
        for align in ("start", "center", "end")
    ]
    block_positions = [
        _render_vertical(inline_align="start", block_align=align)[0]
        for align in ("start", "center", "end")
    ]

    assert inline_positions[0] < inline_positions[1] < inline_positions[2]
    assert block_positions[0] > block_positions[1] > block_positions[2]


def test_bubble_state_round_trip_keeps_both_alignment_axes() -> None:
    state = BubbleState(inline_align="end", block_align="center")

    restored = BubbleState.from_dict(state.to_dict())

    assert restored.inline_align == "end"
    assert restored.block_align == "center"


def test_legacy_single_axis_alignment_is_not_accepted() -> None:
    payload = BubbleState().to_dict()
    payload["textAlign"] = payload.pop("inlineAlign")

    with pytest.raises(ValueError, match="unknown bubble payload fields: textAlign"):
        BubbleState.from_dict(payload)
