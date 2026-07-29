import json
import os
import unittest
from pathlib import Path

from src.core.config_models import BubbleState
from src.shared.text_style_defaults import get_text_style_factory_defaults


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TextStyleFactoryDefaultsTests(unittest.TestCase):
    def test_loader_returns_exact_bundled_factory_defaults(self) -> None:
        defaults_path = (
            PROJECT_ROOT / "src" / "shared" / "text_style_defaults_factory.json"
        )
        expected = json.loads(defaults_path.read_text(encoding="utf-8"))

        self.assertEqual(get_text_style_factory_defaults(), expected)

    def test_factory_defaults_enable_auto_font_size(self) -> None:
        self.assertTrue(get_text_style_factory_defaults()["autoFontSize"])

    def test_factory_defaults_are_returned_as_deep_copies(self) -> None:
        first = get_text_style_factory_defaults()
        first["fontSize"] = 999

        self.assertNotEqual(
            first["fontSize"],
            get_text_style_factory_defaults()["fontSize"],
        )


class BubbleStateDefaultsTests(unittest.TestCase):
    def test_bubble_state_defaults_match_factory_fallbacks(self) -> None:
        defaults = get_text_style_factory_defaults()
        state = BubbleState()

        self.assertEqual(state.font_size, defaults["fontSize"])
        self.assertEqual(
            state.font_family,
            os.path.join(
                "src",
                "backend_v2",
                "resources",
                defaults["fontFamily"].replace("/", os.sep),
            ),
        )
        self.assertEqual(state.text_direction, "vertical")
        self.assertEqual(state.auto_text_direction, "vertical")
        self.assertEqual(state.text_color, defaults["textColor"])
        self.assertEqual(state.fill_color, defaults["fillColor"])
        self.assertEqual(state.inpaint_method, defaults["inpaintMethod"])
        self.assertEqual(state.stroke_enabled, defaults["strokeEnabled"])
        self.assertEqual(state.stroke_color, defaults["strokeColor"])
        self.assertEqual(state.stroke_width, defaults["strokeWidth"])
        self.assertEqual(state.line_spacing, defaults["lineSpacing"])
        self.assertEqual(state.text_align, defaults["textAlign"])


if __name__ == "__main__":
    unittest.main()
