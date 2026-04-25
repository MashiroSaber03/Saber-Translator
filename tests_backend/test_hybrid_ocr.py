import base64
import io
import os
import sys
import unittest
from unittest import mock

from flask import Flask
from PIL import Image


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.app.api.translation.parallel_routes import parallel_bp
from src.app.api.translation import routes as translation_routes
from src.core.config_models import BubbleState
from src.core.ocr import recognize_ocr_results_in_bubbles
from src.core.ocr_types import OcrResult, create_ocr_result, create_ocr_textline_result
from src.plugins.manager import get_plugin_manager


def create_tiny_png_base64() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color="white").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


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


class HybridOcrRouteContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Flask(__name__)
        self.app.register_blueprint(parallel_bp)
        self.app.register_blueprint(translation_routes.translate_bp)
        self.client = self.app.test_client()
        plugin_manager = get_plugin_manager(self.app)
        plugin_manager.plugins = {}
        plugin_manager.hooks = {}

    def test_parallel_ocr_returns_structured_results_and_legacy_texts(self) -> None:
        mocked_results = [
            OcrResult(
                text="こんにちは",
                confidence=0.88,
                confidence_supported=True,
                engine="manga_ocr",
                primary_engine="manga_ocr",
                fallback_used=False,
            )
        ]

        with mock.patch(
            "src.app.api.translation.parallel_routes.recognize_ocr_results_in_bubbles",
            return_value=mocked_results,
        ):
            response = self.client.post(
                "/api/parallel/ocr",
                json={
                    "image": create_tiny_png_base64(),
                    "bubble_coords": [[0, 0, 1, 1]],
                    "source_language": "japanese",
                    "ocr_engine": "manga_ocr",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["original_texts"], ["こんにちは"])
        self.assertEqual(payload["ocr_results"][0]["text"], "こんにちは")
        self.assertEqual(payload["ocr_results"][0]["confidence"], 0.88)
        self.assertEqual(payload["ocr_results"][0]["engine"], "manga_ocr")

    def test_parallel_render_preserves_textlines_and_ocr_metadata_in_bubble_states(self) -> None:
        with mock.patch(
            "src.app.api.translation.parallel_routes.render_bubbles_unified",
            side_effect=lambda image, states: image,
        ):
            response = self.client.post(
                "/api/parallel/render",
                json={
                    "clean_image": create_tiny_png_base64(),
                    "bubble_states": [
                        {
                            "coords": [0, 0, 8, 8],
                            "originalText": "原文",
                            "translatedText": "译文",
                            "textlines": [
                                {
                                    "polygon": [[0, 0], [4, 0], [4, 4], [0, 4]],
                                    "direction": "h",
                                    "confidence": 0.8,
                                }
                            ],
                            "ocrResult": {
                                "text": "原文",
                                "confidence": 0.91,
                                "confidenceSupported": True,
                                "engine": "48px_ocr",
                                "primaryEngine": "48px_ocr",
                                "fallbackUsed": False,
                            },
                            "colorConfidence": 0.66,
                        }
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        bubble_states = payload["bubble_states"]
        self.assertEqual(len(bubble_states), 1)
        self.assertEqual(
            bubble_states[0]["textlines"],
            [
                {
                    "polygon": [[0, 0], [4, 0], [4, 4], [0, 4]],
                    "direction": "h",
                    "confidence": 0.8,
                }
            ],
        )
        self.assertEqual(bubble_states[0]["ocrResult"]["engine"], "48px_ocr")
        self.assertEqual(bubble_states[0]["ocrResult"]["confidence"], 0.91)
        self.assertEqual(bubble_states[0]["colorConfidence"], 0.66)

    def test_single_bubble_ocr_returns_structured_result_and_legacy_text(self) -> None:
        mocked_results = [
            OcrResult(
                text="测试",
                confidence=0.91,
                confidence_supported=True,
                engine="48px_ocr",
                primary_engine="48px_ocr",
                fallback_used=False,
            )
        ]

        with mock.patch(
            "src.app.api.translation.routes.recognize_ocr_results_in_bubbles",
            return_value=mocked_results,
        ), mock.patch(
            "src.app.api.translation.routes.detect_textlines",
            return_value=[],
        ):
            response = self.client.post(
                "/api/ocr_single_bubble",
                json={
                    "bubble_image": create_tiny_png_base64(),
                    "ocr_engine": "48px_ocr",
                    "source_language": "japanese",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["text"], "测试")
        self.assertEqual(payload["ocr_result"]["text"], "测试")
        self.assertEqual(payload["ocr_result"]["confidence"], 0.91)
        self.assertEqual(payload["ocr_result"]["engine"], "48px_ocr")

    def test_single_bubble_route_reuses_provided_textlines(self) -> None:
        captured_kwargs = {}

        def _fake_recognize(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return [create_ocr_result("命中", "48px_ocr")]

        with mock.patch(
            "src.app.api.translation.routes.recognize_ocr_results_in_bubbles",
            side_effect=_fake_recognize,
        ) as recognize_mock, mock.patch(
            "src.app.api.translation.routes.detect_textlines"
        ) as detect_mock:
            response = self.client.post(
                "/api/ocr_single_bubble",
                json={
                    "image_data": create_tiny_png_base64(),
                    "bubble_coords": [2, 2, 6, 6],
                    "ocr_engine": "48px_ocr",
                    "bubble_textlines": [
                        {
                            "polygon": [[2, 2], [5, 2], [5, 5], [2, 5]],
                            "direction": "h",
                        }
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(recognize_mock.called)
        detect_mock.assert_not_called()
        payload = response.get_json()
        self.assertEqual(payload["textlines"], [
            {
                "polygon": [[0, 0], [3, 0], [3, 3], [0, 3]],
                "direction": "h",
                "confidence": 0.0,
            }
        ])
        self.assertEqual(
            captured_kwargs["textlines_per_bubble"],
            [[{"polygon": [[0, 0], [3, 0], [3, 3], [0, 3]], "direction": "h", "confidence": 0.0}]],
        )

    def test_single_bubble_route_detects_textlines_locally_for_supported_hybrid(self) -> None:
        captured_kwargs = {}
        detected_textlines = [{"polygon": [[0, 0], [4, 0], [4, 4], [0, 4]], "direction": "h", "confidence": 0.7}]

        def _fake_recognize(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return [create_ocr_result("命中", "manga_ocr", primary_engine="48px_ocr", fallback_used=True)]

        with mock.patch(
            "src.app.api.translation.routes.recognize_ocr_results_in_bubbles",
            side_effect=_fake_recognize,
        ), mock.patch(
            "src.app.api.translation.routes.detect_textlines",
            return_value=detected_textlines,
        ) as detect_mock:
            response = self.client.post(
                "/api/ocr_single_bubble",
                json={
                    "image_data": create_tiny_png_base64(),
                    "bubble_coords": [0, 0, 6, 6],
                    "ocr_engine": "48px_ocr",
                    "enable_hybrid_ocr": True,
                    "secondary_ocr_engine": "manga_ocr",
                    "text_detector": "default",
                    "enable_aux_yolo_detection": True,
                    "aux_yolo_conf_threshold": 0.5,
                    "aux_yolo_overlap_threshold": 0.2,
                    "enable_saber_yolo_refine": True,
                    "saber_yolo_refine_overlap_threshold": 50,
                },
            )

        self.assertEqual(response.status_code, 200)
        detect_mock.assert_called_once()
        self.assertEqual(captured_kwargs["textlines_per_bubble"], [detected_textlines])

    def test_single_bubble_unsupported_hybrid_combo_returns_400(self) -> None:
        response = self.client.post(
            "/api/ocr_single_bubble",
            json={
                "bubble_image": create_tiny_png_base64(),
                "ocr_engine": "manga_ocr",
                "enable_hybrid_ocr": True,
                "secondary_ocr_engine": "paddle_ocr",
                "bubble_textlines": [
                    {
                        "polygon": [[0, 0], [4, 0], [4, 4], [0, 4]],
                        "direction": "h",
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn("仅支持 MangaOCR / 48px OCR 组合", payload["error"])

    def test_single_bubble_ai_vision_missing_api_key_returns_400(self) -> None:
        response = self.client.post(
            "/api/ocr_single_bubble",
            json={
                "bubble_image": create_tiny_png_base64(),
                "ocr_engine": "ai_vision",
                "source_language": "japanese",
                "ai_vision_api_key": "",
                "ai_vision_model_name": "gpt-4.1-mini"
            },
        )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn("AI视觉OCR需要提供API Key", payload["error"])

    def test_parallel_ocr_plugin_can_modify_legacy_texts_and_structured_results(self) -> None:
        mocked_results = [
            OcrResult(
                text="原文",
                confidence=0.5,
                confidence_supported=True,
                engine="manga_ocr",
                primary_engine="manga_ocr",
                fallback_used=False,
            )
        ]

        plugin_manager = get_plugin_manager(self.app)
        hook_mock = mock.Mock(return_value=["插件改写"])
        plugin_manager.hooks["after_ocr"] = [("mock_plugin", hook_mock)]

        with mock.patch(
            "src.app.api.translation.parallel_routes.recognize_ocr_results_in_bubbles",
            return_value=mocked_results,
        ):
            response = self.client.post(
                "/api/parallel/ocr",
                json={
                    "image": create_tiny_png_base64(),
                    "bubble_coords": [[0, 0, 1, 1]],
                    "source_language": "japanese",
                    "ocr_engine": "manga_ocr",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["original_texts"], ["插件改写"])
        self.assertEqual(payload["ocr_results"][0]["text"], "插件改写")
        self.assertTrue(hook_mock.called)
        hook_args = hook_mock.call_args[0]
        self.assertEqual(hook_args[1], ["原文"])
        self.assertEqual(hook_args[3]["ocr_results"][0]["text"], "原文")
