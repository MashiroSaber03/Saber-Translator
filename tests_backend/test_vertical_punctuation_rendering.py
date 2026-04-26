import os
import sys
import unittest

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.core.rendering import draw_multiline_text_vertical, get_font, process_text_for_vertical


def _top_ink_y(image: Image.Image) -> int | None:
    array = np.array(image)
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


if __name__ == "__main__":
    unittest.main()
