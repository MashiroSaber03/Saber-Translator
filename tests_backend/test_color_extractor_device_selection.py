import os
import sys
import unittest
from unittest import mock

from PIL import Image


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import src.core.color_extractor as color_extractor
from src.interfaces.ocr_48px.interface import Model48pxOCR


class FakeColorExtractor:
    def __init__(self):
        self.is_initialized = False
        self.initialize_calls = []

    def initialize(self, device):
        self.initialize_calls.append(device)
        self.is_initialized = True
        return True

    def extract_colors(self, _image, bubble_coords, _textlines_per_bubble):
        return [
            color_extractor.ColorExtractionResult((1, 2, 3), (4, 5, 6), 0.9)
            for _ in bubble_coords
        ]


class ColorExtractorDeviceSelectionTests(unittest.TestCase):
    def test_extract_bubble_colors_prefers_cuda_when_available(self):
        fake_extractor = FakeColorExtractor()
        image = Image.new("RGB", (8, 8), color="white")

        with mock.patch.object(
            color_extractor,
            "get_color_extractor",
            return_value=fake_extractor,
        ), mock.patch("torch.cuda.is_available", return_value=True):
            results = color_extractor.extract_bubble_colors(image, [(0, 0, 4, 4)])

        self.assertEqual(fake_extractor.initialize_calls, ["cuda"])
        self.assertEqual(
            results,
            [{"fg_color": [1, 2, 3], "bg_color": [4, 5, 6], "confidence": 0.9}],
        )

    def test_48px_initialize_switches_existing_model_to_requested_device(self):
        handler = Model48pxOCR()
        handler.initialized = True
        handler.device = "cpu"
        handler.model = mock.Mock()

        result = handler.initialize("cuda")

        self.assertTrue(result)
        handler.model.to.assert_called_once_with("cuda")
        self.assertEqual(handler.device, "cuda")


if __name__ == "__main__":
    unittest.main()
