import unittest
from unittest import mock

from PIL import Image

from src.core.config_models import BubbleState
from src.core.ocr import recognize_ocr_results_in_bubbles
from src.core.ocr_types import create_ocr_result, create_ocr_textline_result

class HybridOcrCoreTests(unittest.TestCase):
    def test_backend_bubble_state_round_trips_textlines(self) -> None:
        state = BubbleState.from_dict({
            "coords": [1, 2, 3, 4],
            "textlines": [
                {
                    "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "direction": "h",
                    "confidence": 0.7,
                }
            ],
            "ocrResult": {
                "text": "原文",
                "confidence": 0.8,
                "confidenceSupported": True,
                "engine": "48px_ocr",
                "primaryEngine": "48px_ocr",
                "fallbackUsed": False,
            },
        })

        payload = state.to_dict()

        self.assertEqual(payload["textlines"], [
            {
                "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "direction": "h",
                "confidence": 0.7,
            }
        ])
        self.assertEqual(payload["ocrResult"]["text"], "原文")

    def test_plain_manga_ocr_no_longer_uses_48px_composite_confidence(self) -> None:
        with mock.patch("src.core.ocr.get_manga_ocr_instance", return_value=object()), \
             mock.patch("src.core.ocr.recognize_japanese_text", return_value="こんにちは"), \
             mock.patch("src.interfaces.ocr_48px.get_48px_ocr_handler", side_effect=AssertionError("48px should not be used")):
            results = recognize_ocr_results_in_bubbles(
                Image.new("RGB", (16, 16), color="white"),
                [(0, 0, 16, 16)],
                ocr_engine="manga_ocr",
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "こんにちは")
        self.assertIsNone(results[0].confidence)
        self.assertFalse(results[0].confidence_supported)

    def test_supported_hybrid_combo_uses_specialized_adapter(self) -> None:
        mocked_results = [
            create_ocr_result(
                "混合结果",
                "manga_ocr",
                confidence=0.42,
                confidence_supported=True,
                primary_engine="48px_ocr",
                fallback_used=True,
            )
        ]

        with mock.patch("src.core.ocr.recognize_manga_48_hybrid", return_value=mocked_results) as hybrid_mock:
            results = recognize_ocr_results_in_bubbles(
                Image.new("RGB", (32, 32), color="white"),
                [(0, 0, 16, 16)],
                ocr_engine="48px_ocr",
                enable_hybrid_ocr=True,
                secondary_ocr_engine="manga_ocr",
                textlines_per_bubble=[[{"polygon": [[0, 0], [8, 0], [8, 8], [0, 8]], "direction": "h"}]],
                hybrid_ocr_threshold=0.2,
            )

        self.assertEqual(results, mocked_results)
        hybrid_mock.assert_called_once()

    def test_unsupported_hybrid_combo_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "仅支持 MangaOCR / 48px OCR 组合"):
            recognize_ocr_results_in_bubbles(
                Image.new("RGB", (16, 16), color="white"),
                [(0, 0, 16, 16)],
                ocr_engine="manga_ocr",
                enable_hybrid_ocr=True,
                secondary_ocr_engine="paddle_ocr",
            )

    def test_specialized_hybrid_path_uses_48px_textline_ocr_not_color_extraction(self) -> None:
        fake_handler = mock.Mock()
        fake_handler.initialize.return_value = True
        fake_handler.recognize_textlines_with_details.return_value = [
            create_ocr_textline_result(
                "",
                "48px_ocr",
                confidence=0.1,
                confidence_supported=True,
                primary_engine="48px_ocr",
                polygon=[[0, 0], [8, 0], [8, 8], [0, 8]],
                direction="h",
            )
        ]
        fake_handler.extract_colors_for_bubbles = mock.Mock()

        with mock.patch("src.core.ocr_hybrid_manga_48.get_48px_ocr_handler", return_value=fake_handler), \
             mock.patch("src.core.ocr_hybrid_manga_48.recognize_japanese_text", return_value="后备识别"), \
             mock.patch("src.core.ocr_hybrid_manga_48.torch.cuda.is_available", return_value=False):
            results = recognize_ocr_results_in_bubbles(
                Image.new("RGB", (16, 16), color="white"),
                [(0, 0, 16, 16)],
                ocr_engine="48px_ocr",
                enable_hybrid_ocr=True,
                secondary_ocr_engine="manga_ocr",
                textlines_per_bubble=[[{"polygon": [[0, 0], [8, 0], [8, 8], [0, 8]], "direction": "h"}]],
                hybrid_ocr_threshold=0.2,
            )

        self.assertEqual(results[0].text, "后备识别")
        self.assertTrue(results[0].fallback_used)
        fake_handler.recognize_textlines_with_details.assert_called_once()
        fake_handler.extract_colors_for_bubbles.assert_not_called()

    def test_specialized_hybrid_path_batches_48px_textlines_across_bubbles(self) -> None:
        fake_handler = mock.Mock()
        fake_handler.initialize.return_value = True
        fake_handler.recognize_textlines_with_details.return_value = [
            create_ocr_textline_result(
                "甲",
                "48px_ocr",
                confidence=0.9,
                confidence_supported=True,
                primary_engine="48px_ocr",
                polygon=[[0, 0], [4, 0], [4, 4], [0, 4]],
                direction="h",
            ),
            create_ocr_textline_result(
                "乙",
                "48px_ocr",
                confidence=0.8,
                confidence_supported=True,
                primary_engine="48px_ocr",
                polygon=[[10, 0], [14, 0], [14, 4], [10, 4]],
                direction="h",
            ),
            create_ocr_textline_result(
                "丙",
                "48px_ocr",
                confidence=0.7,
                confidence_supported=True,
                primary_engine="48px_ocr",
                polygon=[[10, 5], [14, 5], [14, 9], [10, 9]],
                direction="h",
            ),
        ]

        bubble_textlines = [
            [{"polygon": [[0, 0], [4, 0], [4, 4], [0, 4]], "direction": "h"}],
            [
                {"polygon": [[10, 0], [14, 0], [14, 4], [10, 4]], "direction": "h"},
                {"polygon": [[10, 5], [14, 5], [14, 9], [10, 9]], "direction": "h"},
            ],
        ]

        with mock.patch("src.core.ocr_hybrid_manga_48.get_48px_ocr_handler", return_value=fake_handler), \
             mock.patch("src.core.ocr_hybrid_manga_48.torch.cuda.is_available", return_value=False):
            results = recognize_ocr_results_in_bubbles(
                Image.new("RGB", (32, 16), color="white"),
                [(0, 0, 8, 8), (8, 0, 16, 16)],
                ocr_engine="48px_ocr",
                enable_hybrid_ocr=True,
                secondary_ocr_engine="manga_ocr",
                textlines_per_bubble=bubble_textlines,
                hybrid_ocr_threshold=0.2,
            )

        self.assertEqual([result.text for result in results], ["甲", "乙 丙"])
        fake_handler.recognize_textlines_with_details.assert_called_once()
        called_args, called_kwargs = fake_handler.recognize_textlines_with_details.call_args
        self.assertEqual(called_args[1], bubble_textlines[0] + bubble_textlines[1])
        self.assertEqual(called_kwargs["primary_engine"], "48px_ocr")
