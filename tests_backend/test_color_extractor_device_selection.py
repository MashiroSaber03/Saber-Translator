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

    def test_color_extractor_reloads_48px_after_worker_releases_model(self):
        extractor = color_extractor.ColorExtractor()
        released_handler = mock.Mock()
        released_handler.initialized = False
        released_handler.model = None
        extractor._ocr_handler = released_handler
        extractor._initialized = True
        extractor._device = "cpu"

        loaded_handler = mock.Mock()
        loaded_handler.initialized = False
        loaded_handler.model = None

        def initialize(_device):
            loaded_handler.initialized = True
            loaded_handler.model = object()
            return True

        loaded_handler.initialize.side_effect = initialize
        loaded_handler.extract_colors_for_bubbles.return_value = [
            mock.Mock(fg_color=(7, 8, 9), bg_color=(240, 241, 242), confidence=0.8)
        ]
        image = Image.new("RGB", (8, 8), color="white")

        with mock.patch.object(
            color_extractor,
            "get_color_extractor",
            return_value=extractor,
        ), mock.patch(
            "src.interfaces.ocr_48px.get_48px_ocr_handler",
            return_value=loaded_handler,
        ):
            results = color_extractor.extract_bubble_colors(
                image,
                [(0, 0, 4, 4)],
                device="cpu",
            )

        loaded_handler.initialize.assert_called_once_with("cpu")
        self.assertEqual(
            results,
            [
                {
                    "fg_color": [7, 8, 9],
                    "bg_color": [240, 241, 242],
                    "confidence": 0.8,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
