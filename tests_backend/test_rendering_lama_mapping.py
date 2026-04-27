import unittest
from unittest import mock

from PIL import Image

from src.core.config_models import BubbleState
from src.core.rendering import render_single_bubble_unified


class RenderingLamaMappingTests(unittest.TestCase):
    def test_render_single_bubble_unified_maps_litelama_to_lama_method_and_model(self) -> None:
        image = Image.new("RGB", (200, 120), "white")
        bubble_states = [
            BubbleState(
                translated_text="",
                coords=(20, 20, 100, 80),
                fill_color="#FFFFFF",
                inpaint_method="litelama",
            )
        ]

        with mock.patch("src.core.inpainting.inpaint_bubbles", return_value=(image.copy(), image.copy())) as inpaint_mock, \
             mock.patch("src.core.rendering.render_bubbles_unified", return_value=image.copy()):
            render_single_bubble_unified(
                image=image,
                bubble_states=bubble_states,
                bubble_index=0,
                use_clean_background=False,
            )

        inpaint_mock.assert_called_once()
        _, kwargs = inpaint_mock.call_args
        self.assertEqual(kwargs["method"], "lama")
        self.assertEqual(kwargs["lama_model"], "litelama")
