import os
import sys
import unittest
from unittest import mock

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.interfaces.ocr_48px.interface import Model48pxOCR


class Fake48pxModel:
    def __init__(self, width_to_output):
        self.width_to_output = width_to_output
        self.calls = []

    def infer_beam_batch_tensor(self, tensor, img_widths, beams_k=5, max_seq_length=255):
        self.calls.append(list(img_widths))
        predictions = []
        for width in img_widths:
            token_index, prob = self.width_to_output[width]
            char_indices = torch.tensor([token_index, 2], dtype=torch.long)
            fg_pred = torch.zeros((2, 3), dtype=torch.float32)
            bg_pred = torch.zeros((2, 3), dtype=torch.float32)
            fg_ind_pred = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
            bg_ind_pred = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
            predictions.append((char_indices, float(prob), fg_pred, bg_pred, fg_ind_pred, bg_ind_pred))
        return predictions


class Ocr48pxBatchingTests(unittest.TestCase):
    def _create_handler(self, width_to_output):
        handler = Model48pxOCR()
        handler.initialized = True
        handler.device = "cpu"
        handler.dictionary = [
            "<PAD>",
            "<S>",
            "</S>",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
        ]
        handler.model = Fake48pxModel(width_to_output)
        return handler

    @staticmethod
    def _create_polygon(width):
        return [[0, 0], [width, 0], [width, 10], [0, 10]]

    @staticmethod
    def _fake_region_from_polygon(_image, pts, _direction, target_height=48):
        width = max(int(round(float(pts[1][0] - pts[0][0]))), 1)
        return np.zeros((target_height, width, 3), dtype=np.uint8)

    def test_recognize_textlines_with_details_batches_lines_and_preserves_input_order(self):
        handler = self._create_handler({
            30: (3, 0.91),
            10: (4, 0.82),
            20: (5, 0.73),
        })
        image = Image.new("RGB", (64, 64), color="white")
        textlines = [
            {"polygon": self._create_polygon(30), "direction": "h"},
            {"polygon": [[0, 0], [1, 1]], "direction": "h"},
            {"polygon": self._create_polygon(10), "direction": "h"},
            {"polygon": self._create_polygon(20), "direction": "h"},
        ]

        with mock.patch(
            "src.interfaces.ocr_48px.interface.get_transformed_region",
            side_effect=self._fake_region_from_polygon,
        ):
            results = handler.recognize_textlines_with_details(image, textlines)

        self.assertEqual([result.text for result in results], ["A", "", "B", "C"])
        self.assertEqual(handler.model.calls, [[10, 20, 30]])

    def test_recognize_textlines_with_details_splits_into_multiple_chunks(self):
        width_to_output = {
            width: (3 + index, 0.8)
            for index, width in enumerate(range(1, 18))
        }
        handler = self._create_handler(width_to_output)
        image = Image.new("RGB", (64, 64), color="white")
        textlines = [
            {"polygon": self._create_polygon(width), "direction": "h"}
            for width in range(17, 0, -1)
        ]

        with mock.patch(
            "src.interfaces.ocr_48px.interface.get_transformed_region",
            side_effect=self._fake_region_from_polygon,
        ):
            results = handler.recognize_textlines_with_details(image, textlines)

        self.assertEqual(len(results), 17)
        self.assertEqual(len(handler.model.calls), 2)
        self.assertEqual(handler.model.calls[0], list(range(1, 17)))
        self.assertEqual(handler.model.calls[1], [17])

    def test_recognize_text_with_details_batches_textlines_and_aggregates_bubbles(self):
        handler = self._create_handler({
            30: (3, 0.9),
            10: (4, 0.5),
            20: (5, 0.7),
        })
        image = Image.new("RGB", (80, 80), color="white")
        bubble_coords = [(0, 0, 40, 40), (40, 0, 80, 40)]
        textlines_per_bubble = [
            [
                {"polygon": self._create_polygon(30), "direction": "h"},
                {"polygon": [[0, 0], [1, 1]], "direction": "h"},
                {"polygon": self._create_polygon(10), "direction": "h"},
            ],
            [
                {"polygon": self._create_polygon(20), "direction": "h"},
            ],
        ]

        with mock.patch(
            "src.interfaces.ocr_48px.interface.get_transformed_region",
            side_effect=self._fake_region_from_polygon,
        ):
            results = handler.recognize_text_with_details(image, bubble_coords, textlines_per_bubble)

        self.assertEqual([result.text for result in results], ["A B", "C"])
        self.assertAlmostEqual(results[0].confidence or 0.0, 0.7, places=6)
        self.assertAlmostEqual(results[1].confidence or 0.0, 0.7, places=6)
        self.assertEqual(handler.model.calls, [[10, 20, 30]])

    def test_recognize_text_with_details_batches_simple_crop_fallback(self):
        handler = self._create_handler({
            30: (3, 0.9),
            10: (4, 0.6),
            20: (5, 0.7),
        })
        image = Image.new("RGB", (60, 48), color="white")
        bubble_coords = [(0, 0, 30, 48), (30, 0, 40, 48), (40, 0, 60, 48)]

        results = handler.recognize_text_with_details(image, bubble_coords, None)

        self.assertEqual([result.text for result in results], ["A", "B", "C"])
        self.assertEqual(handler.model.calls, [[10, 20, 30]])


if __name__ == "__main__":
    unittest.main()
