import os
import sys
import unittest

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.core.rendering import (
    draw_multiline_text_vertical,
    get_char_ink_offset,
    get_font,
    process_text_for_vertical,
)


def _top_ink_y(image: Image.Image) -> int | None:
    array = np.array(image.convert("L"))
    y_positions, _x_positions = np.where(array < 250)
    if len(y_positions) == 0:
        return None
    return int(y_positions.min())


def _render_vertical_top_y(char: str, font_path: str, font_size: int = 48, start_y: int = 49) -> int:
    font = get_font(font_path, font_size)
    image = Image.new("L", (160, 160), 255)
    draw = ImageDraw.Draw(image)
    draw_multiline_text_vertical(
        draw,
        char,
        font,
        x=100,
        y=start_y,
        max_height=100,
        font_family_path=font_path,
    )
    top_y = _top_ink_y(image)
    if top_y is None:
        raise AssertionError(f"竖排渲染未产生可见像素: {ascii(char)}")
    return top_y


def _render_direct_top_y(char: str, font_path: str, font_size: int = 48, start_y: int = 49) -> int:
    font = get_font(font_path, font_size)
    image = Image.new("L", (160, 160), 255)
    draw = ImageDraw.Draw(image)
    draw.text((40, start_y), char, font=font, fill=0)
    top_y = _top_ink_y(image)
    if top_y is None:
        raise AssertionError(f"直接渲染未产生可见像素: {ascii(char)}")
    return top_y


def _render_compound_reference_top_y(chars: str, font_path: str, font_size: int = 48, start_y: int = 49) -> int:
    tops = [_render_direct_top_y(char, font_path, font_size=font_size, start_y=start_y) for char in chars]
    return min(tops)


def _render_vertical_bbox(text: str, font_path: str, font_size: int = 48, start_y: int = 49) -> tuple[int, int, int, int]:
    font = get_font(font_path, font_size)
    image = Image.new("RGB", (260, 260), "white")
    draw = ImageDraw.Draw(image)
    draw_multiline_text_vertical(
        draw,
        text,
        font,
        x=180,
        y=start_y,
        max_height=180,
        font_family_path=font_path,
    )
    array = np.array(image.convert("L"))
    y_positions, x_positions = np.where(array < 250)
    if len(y_positions) == 0:
        raise AssertionError(f"竖排渲染未产生可见像素: {ascii(text)}")
    return int(x_positions.min()), int(y_positions.min()), int(x_positions.max()), int(y_positions.max())


class VerticalPunctuationRenderingTests(unittest.TestCase):
    FONTS = (
        os.path.join("src", "app", "static", "fonts", "msyh.ttc"),
        os.path.join("src", "app", "static", "fonts", "Arial_Unicode.ttf"),
    )

    def test_vertical_question_and_exclamation_marks_do_not_jump_far_above_direct_draw_position(self) -> None:
        for font_path in self.FONTS:
            with self.subTest(font=font_path, char="？"):
                vertical_top = _render_vertical_top_y("？", font_path)
                direct_top = _render_direct_top_y("？", font_path)
                self.assertGreaterEqual(
                    vertical_top - direct_top,
                    -2,
                    f"{font_path} 中的竖排问号被额外上提过多: vertical={vertical_top}, direct={direct_top}",
                )

            with self.subTest(font=font_path, char="！"):
                vertical_top = _render_vertical_top_y("！", font_path)
                direct_top = _render_direct_top_y("！", font_path)
                self.assertGreaterEqual(
                    vertical_top - direct_top,
                    -2,
                    f"{font_path} 中的竖排感叹号被额外上提过多: vertical={vertical_top}, direct={direct_top}",
                )

    def test_vertical_comma_still_receives_low_punctuation_lift(self) -> None:
        for font_path in self.FONTS:
            with self.subTest(font=font_path):
                vertical_top = _render_vertical_top_y("，", font_path)
                direct_top = _render_direct_top_y("，", font_path)
                self.assertLessEqual(
                    vertical_top - direct_top,
                    -12,
                    f"{font_path} 中的竖排逗号不应失去低位标点校正: vertical={vertical_top}, direct={direct_top}",
                )

    def test_vertical_compound_punctuation_does_not_jump_far_above_constituent_glyphs(self) -> None:
        cases = (
            ("!?", "!?"),
            ("！？", "!?"),
            ("?!", "?!"),
            ("？！", "?!"),
            ("!!", "!!"),
            ("！！", "!!"),
            ("??", "??"),
            ("？？", "??"),
        )

        for font_path in self.FONTS:
            for text, reference_chars in cases:
                with self.subTest(font=font_path, text=text):
                    vertical_top = _render_vertical_top_y(text, font_path)
                    reference_top = _render_compound_reference_top_y(reference_chars, font_path)
                    self.assertGreaterEqual(
                        vertical_top - reference_top,
                        -2,
                        (
                            f"{font_path} 中的竖排组合标点 {text} 被额外上提过多: "
                            f"vertical={vertical_top}, reference={reference_top}"
                        ),
                    )

    def test_process_text_for_vertical_normalizes_fullwidth_compound_punctuation_to_ordered_compound_symbols(self) -> None:
        self.assertEqual(process_text_for_vertical("!?"), "⁉")
        self.assertEqual(process_text_for_vertical("！？"), "⁉")
        self.assertEqual(process_text_for_vertical("?!"), "⁈")
        self.assertEqual(process_text_for_vertical("？！"), "⁈")
        self.assertEqual(process_text_for_vertical("!!"), "‼")
        self.assertEqual(process_text_for_vertical("！！"), "‼")
        self.assertEqual(process_text_for_vertical("??"), "⁇")
        self.assertEqual(process_text_for_vertical("？？"), "⁇")

    def test_single_vertical_linear_punctuation_aligns_to_single_cjk_visual_center(self) -> None:
        font_size = 48
        start_y = 49
        line_height_unit = font_size + 1
        cases = ("...", "…", "—", "―")

        for font_path in self.FONTS:
            font = get_font(font_path, font_size)
            ref_ink_offset_y = get_char_ink_offset("我", font)[1]
            cjk_ink_center_in_unit = line_height_unit / 2 + ref_ink_offset_y
            expected_center = start_y + cjk_ink_center_in_unit

            for text in cases:
                with self.subTest(font=font_path, text=text):
                    _left, top_y, _right, bottom_y = _render_vertical_bbox(text, font_path, font_size=font_size, start_y=start_y)
                    rendered_center = (top_y + bottom_y) / 2.0
                    self.assertAlmostEqual(
                        rendered_center,
                        expected_center,
                        delta=3.0,
                        msg=(
                            f"{font_path} 中的竖排单个线性标点 {text} 未对齐中文正文的视觉中心: "
                            f"rendered_center={rendered_center}, expected_center={expected_center}"
                        ),
                    )

    def test_vertical_ellipsis_block_aligns_to_cjk_visual_center(self) -> None:
        font_size = 48
        start_y = 49
        line_height_unit = font_size + 1

        for font_path in self.FONTS:
            with self.subTest(font=font_path):
                font = get_font(font_path, font_size)
                _left, top_y, _right, bottom_y = _render_vertical_bbox("......", font_path, font_size=font_size, start_y=start_y)
                rendered_center = (top_y + bottom_y) / 2.0

                ref_ink_offset_y = get_char_ink_offset("我", font)[1]
                cjk_ink_center_in_unit = line_height_unit / 2 + ref_ink_offset_y
                expected_center = start_y + ((2 - 1) / 2) * line_height_unit + cjk_ink_center_in_unit

                self.assertAlmostEqual(
                    rendered_center,
                    expected_center,
                    delta=3.0,
                    msg=(
                        f"{font_path} 中的竖排省略号块未对齐中文正文的视觉中心: "
                        f"rendered_center={rendered_center}, expected_center={expected_center}"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
