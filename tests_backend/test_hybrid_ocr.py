import base64
import copy
import io
import os
import stat
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from flask import Flask
from PIL import Image


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.app.api.translation.parallel_routes import parallel_bp
from src.app.api.translation import routes as translation_routes
from src.app.api.system import system_bp
from src.core.config_models import BubbleState
from src.core.ocr import recognize_ocr_results_in_bubbles
from src.core.ocr_types import OcrResult, create_ocr_result, create_ocr_textline_result
from src.plugins.base import PluginBase
from src.plugins.manager import PluginManager, get_plugin_manager


def create_tiny_png_base64() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color="white").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class StepHookFixturePlugin(PluginBase):
    plugin_id = "fixture_plugin"
    display_name = "Fixture Plugin"
    plugin_version = "1.0"
    plugin_author = "Tests"
    plugin_description = "用于验证 v2 插件步骤钩子。"
    default_enabled = True
    supported_steps = (
        "detect",
        "ocr",
        "translate",
        "ai_translate",
        "inpaint",
        "render",
    )
    supported_modes = (
        "standard",
        "hq",
        "proofread",
    )

    def __init__(self, plugin_manager, app=None):
        super().__init__(plugin_manager, app=app)
        self.seen_contexts = []

    def get_config_schema(self):
        return {
            "suffix": {
                "type": "text",
                "label": "Suffix",
                "default": "-cfg",
                "description": "追加到原文的测试后缀",
            }
        }

    def before_detect(self, context, payload):
        self.seen_contexts.append((context.step, context.mode))
        updated = dict(payload)
        updated["detector_type"] = "plugin_detector"
        return updated

    def after_detect(self, context, result):
        updated = copy.deepcopy(result)
        updated["bubble_coords"] = list(updated.get("bubble_coords", [])) + [[9, 9, 10, 10]]
        return updated

    def after_ocr(self, context, result):
        updated = copy.deepcopy(result)
        updated["original_texts"] = ["插件OCR"]
        if updated.get("ocr_results"):
            updated["ocr_results"][0]["text"] = "插件OCR"
        return updated

    def before_translate(self, context, payload):
        self.seen_contexts.append((context.step, context.mode))
        updated = dict(payload)
        suffix = self.config.get("suffix", "")
        updated["original_texts"] = [
            f"{text}{suffix}" for text in payload.get("original_texts", [])
        ]
        return updated

    def after_translate(self, context, result):
        updated = copy.deepcopy(result)
        updated["translated_texts"] = [
            f"{text}|after" for text in updated.get("translated_texts", [])
        ]
        return updated

    def after_ai_translate(self, context, result):
        self.seen_contexts.append((context.step, context.mode))
        updated = copy.deepcopy(result)
        updated["results"][0]["bubbles"][0]["translated"] += "|after_ai"
        return updated

    def after_inpaint(self, context, result):
        self.seen_contexts.append((context.step, context.mode))
        updated = copy.deepcopy(result)
        updated["clean_image"] = "plugin-clean-image"
        return updated

    def after_render(self, context, result):
        self.seen_contexts.append((context.step, context.mode))
        updated = copy.deepcopy(result)
        if "final_image" in updated:
            updated["final_image"] = "plugin-render-image"
        if "rendered_images" in updated and updated["rendered_images"]:
            updated["rendered_images"][0] = "plugin-render-image"
        return updated


class AliasMetadataPlugin(PluginBase):
    plugin_id = "alias_plugin"
    display_name = ""
    plugin_description = "  alias test  "
    supported_steps = ("aiTranslate", "ocr")
    supported_modes = ("removeText", "hq")
    failure_policy = "CONTINUE"


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

    def test_plugin_manager_can_load_external_plugin_without_root_package(self) -> None:
        with tempfile.TemporaryDirectory() as plugin_root:
            plugin_dir = os.path.join(plugin_root, "sample-plugin")
            os.makedirs(plugin_dir, exist_ok=True)
            with open(os.path.join(plugin_dir, "__init__.py"), "w", encoding="utf-8") as file:
                file.write("")
            with open(os.path.join(plugin_dir, "plugin.py"), "w", encoding="utf-8") as file:
                file.write(
                    "from src.plugins.base import PluginBase\n"
                    "class TempPlugin(PluginBase):\n"
                    "    plugin_id = 'temp_plugin'\n"
                    "    display_name = 'Temp Plugin'\n"
                    "    supported_steps = ('ocr',)\n"
                    "    supported_modes = ('standard',)\n"
                )

            manager = PluginManager(plugin_dirs=[plugin_root])
            manager.discover_and_load_plugins()

            plugin = manager.get_plugin("temp_plugin")
            self.assertIsNotNone(plugin)
            self.assertEqual(plugin.display_name, "Temp Plugin")

    def test_remove_plugin_deletes_orphaned_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as config_root:
            manager = PluginManager(plugin_dirs=[])
            manager.plugin_config_dir = config_root
            plugin = StepHookFixturePlugin(manager)
            manager.register_plugin_instance(plugin, source_path="tests://fixture_plugin", enabled=True)
            self.assertTrue(manager.save_plugin_config("fixture_plugin", {"suffix": "-saved"}))

            config_path = os.path.join(config_root, "fixture_plugin.json")
            self.assertTrue(os.path.exists(config_path))

            manager.remove_plugin("fixture_plugin")

            self.assertFalse(os.path.exists(config_path))

    def test_get_plugin_manager_updates_app_reference_for_loaded_plugins(self) -> None:
        first_app = Flask("first")
        second_app = Flask("second")

        manager = get_plugin_manager(first_app)
        manager.reset_for_testing()
        plugin = StepHookFixturePlugin(manager, app=first_app)
        manager.register_plugin_instance(plugin, source_path="tests://fixture_plugin", enabled=True)

        updated_manager = get_plugin_manager(second_app)

        self.assertIs(updated_manager.app, second_app)
        self.assertIs(updated_manager.get_plugin("fixture_plugin").app, second_app)

    def test_plugin_metadata_validation_accepts_alias_step_and_mode_names(self) -> None:
        manager = PluginManager(plugin_dirs=[])
        plugin = AliasMetadataPlugin(manager)
        plugin.validate_metadata()

        self.assertEqual(plugin.display_name, "alias_plugin")
        self.assertEqual(plugin.plugin_description, "alias test")
        self.assertEqual(plugin.supported_steps, ("ai_translate", "ocr"))
        self.assertEqual(plugin.supported_modes, ("remove_text", "hq"))
        self.assertEqual(plugin.failure_policy, "continue")


class HybridOcrRouteContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Flask(__name__)
        self.app.register_blueprint(parallel_bp)
        self.app.register_blueprint(translation_routes.translate_bp)
        self.app.register_blueprint(system_bp)
        self.client = self.app.test_client()
        plugin_manager = get_plugin_manager(self.app)
        plugin_manager.reset_for_testing()

    def tearDown(self) -> None:
        get_plugin_manager().reset_for_testing()

    def register_fixture_plugin(self) -> StepHookFixturePlugin:
        plugin_manager = get_plugin_manager(self.app)
        plugin = StepHookFixturePlugin(plugin_manager, app=self.app)
        plugin_manager.register_plugin_instance(
            plugin,
            source_path="tests://fixture_plugin",
            enabled=True,
        )
        return plugin

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

    def test_plugins_api_returns_v2_metadata_and_keyed_schema(self) -> None:
        self.register_fixture_plugin()

        list_response = self.client.get("/api/plugins")
        self.assertEqual(list_response.status_code, 200)
        list_payload = list_response.get_json()
        self.assertTrue(list_payload["success"])
        self.assertEqual(len(list_payload["plugins"]), 1)

        plugin = list_payload["plugins"][0]
        self.assertEqual(plugin["id"], "fixture_plugin")
        self.assertEqual(plugin["display_name"], "Fixture Plugin")
        self.assertTrue(plugin["enabled"])
        self.assertTrue(plugin["default_enabled"])
        self.assertTrue(plugin["has_config"])
        self.assertEqual(plugin["supported_steps"], ["detect", "ocr", "translate", "ai_translate", "inpaint", "render"])
        self.assertEqual(plugin["supported_modes"], ["standard", "hq", "proofread"])

        schema_response = self.client.get("/api/plugins/fixture_plugin/config_schema")
        self.assertEqual(schema_response.status_code, 200)
        schema_payload = schema_response.get_json()
        self.assertTrue(schema_payload["success"])
        self.assertEqual(schema_payload["schema"]["suffix"]["default"], "-cfg")

        config_response = self.client.get("/api/plugins/fixture_plugin/config")
        self.assertEqual(config_response.status_code, 200)
        config_payload = config_response.get_json()
        self.assertEqual(config_payload["config"]["suffix"], "-cfg")

    def test_delete_plugin_api_removes_plugin_directory_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as plugin_root:
            plugin_dir = os.path.join(plugin_root, "fixture_plugin_dir")
            os.makedirs(plugin_dir, exist_ok=True)

            plugin_manager = get_plugin_manager(self.app)
            plugin_manager.reset_for_testing()
            plugin_manager.plugin_config_dir = plugin_root

            plugin = StepHookFixturePlugin(plugin_manager, app=self.app)
            plugin_manager.register_plugin_instance(
                plugin,
                source_path=plugin_dir,
                enabled=True,
            )
            plugin_manager.save_plugin_config("fixture_plugin", {"suffix": "-saved"})

            response = self.client.delete("/api/plugins/fixture_plugin")

            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertTrue(payload["success"])
            self.assertFalse(os.path.exists(plugin_dir))
            self.assertFalse(os.path.exists(os.path.join(plugin_root, "fixture_plugin.json")))
            self.assertIsNone(plugin_manager.get_plugin("fixture_plugin"))

    def test_delete_plugin_api_cleans_runtime_when_directory_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as plugin_root:
            missing_plugin_dir = os.path.join(plugin_root, "missing_plugin_dir")

            plugin_manager = get_plugin_manager(self.app)
            plugin_manager.reset_for_testing()
            plugin_manager.plugin_config_dir = plugin_root

            plugin = StepHookFixturePlugin(plugin_manager, app=self.app)
            plugin_manager.register_plugin_instance(
                plugin,
                source_path=missing_plugin_dir,
                enabled=True,
            )
            plugin_manager.save_plugin_config("fixture_plugin", {"suffix": "-saved"})

            response = self.client.delete("/api/plugins/fixture_plugin")

            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertTrue(payload["success"])
            self.assertIn("目录不存在", payload["message"])
            self.assertFalse(os.path.exists(os.path.join(plugin_root, "fixture_plugin.json")))
            self.assertIsNone(plugin_manager.get_plugin("fixture_plugin"))

    def test_delete_plugin_api_removes_readonly_pycache_directory_on_windows(self) -> None:
        with tempfile.TemporaryDirectory() as plugin_root:
            plugin_dir = os.path.join(plugin_root, "fixture_plugin_dir")
            pycache_dir = os.path.join(plugin_dir, "__pycache__")
            os.makedirs(pycache_dir, exist_ok=True)
            pycache_file = os.path.join(pycache_dir, "plugin.cpython-312.pyc")
            with open(pycache_file, "wb") as handle:
                handle.write(b"stub")

            os.chmod(pycache_file, stat.S_IREAD)
            os.chmod(pycache_dir, stat.S_IREAD)

            plugin_manager = get_plugin_manager(self.app)
            plugin_manager.reset_for_testing()
            plugin_manager.plugin_config_dir = plugin_root

            plugin = StepHookFixturePlugin(plugin_manager, app=self.app)
            plugin_manager.register_plugin_instance(
                plugin,
                source_path=plugin_dir,
                enabled=True,
            )

            response = self.client.delete("/api/plugins/fixture_plugin")

            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertTrue(payload["success"])
            self.assertFalse(os.path.exists(plugin_dir))

    def test_parallel_detect_hooks_can_modify_request_and_response(self) -> None:
        self.register_fixture_plugin()
        captured_kwargs = {}

        def _fake_detect(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return {
                "coords": [[0, 0, 1, 1]],
                "angles": [0],
                "polygons": [[[0, 0], [1, 0], [1, 1], [0, 1]]],
                "auto_directions": ["v"],
                "textlines_per_bubble": [],
                "raw_mask": None,
            }

        with mock.patch(
            "src.app.api.translation.parallel_routes.get_bubble_detection_result_with_auto_directions",
            side_effect=_fake_detect,
        ):
            response = self.client.post(
                "/api/parallel/detect",
                json={
                    "image": create_tiny_png_base64(),
                    "detector_type": "default",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured_kwargs["detector_type"], "plugin_detector")
        payload = response.get_json()
        self.assertEqual(payload["bubble_coords"], [[0, 0, 1, 1], [9, 9, 10, 10]])

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

        self.register_fixture_plugin()

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
        self.assertEqual(payload["original_texts"], ["插件OCR"])
        self.assertEqual(payload["ocr_results"][0]["text"], "插件OCR")

    def test_single_bubble_ocr_plugin_can_modify_response_with_v2_shape(self) -> None:
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

        self.register_fixture_plugin()

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
                    "ocr_engine": "manga_ocr",
                    "source_language": "japanese",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["text"], "插件OCR")
        self.assertEqual(payload["ocr_result"]["text"], "插件OCR")

    def test_parallel_translate_hooks_can_modify_request_and_response(self) -> None:
        self.register_fixture_plugin()
        captured_texts = []

        def _fake_translate(texts, **kwargs):
            captured_texts.extend(texts)
            return [f"translated:{text}" for text in texts]

        with mock.patch(
            "src.app.api.translation.parallel_routes.translate_text_list",
            side_effect=_fake_translate,
        ):
            response = self.client.post(
                "/api/parallel/translate",
                json={
                    "original_texts": ["hello"],
                    "target_language": "zh",
                    "source_language": "japanese",
                    "model_provider": "openai",
                    "api_key": "test-key",
                    "model_name": "gpt-4.1-mini",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured_texts, ["hello-cfg"])
        payload = response.get_json()
        self.assertEqual(payload["translated_texts"], ["translated:hello-cfg|after"])

    def test_single_text_translate_hooks_use_canonical_v2_shape(self) -> None:
        self.register_fixture_plugin()
        captured = {}

        def _fake_translate_single_text(**kwargs):
            captured["text"] = kwargs["text"]
            return "译文"

        with mock.patch.object(
            translation_routes,
            "translate_single_text",
            side_effect=_fake_translate_single_text,
        ):
            response = self.client.post(
                "/api/translate_single_text",
                json={
                    "original_text": "hello",
                    "target_language": "zh",
                    "model_provider": "siliconflow",
                    "api_key": "test-key",
                    "model_name": "test-model",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["text"], "hello-cfg")
        payload = response.get_json()
        self.assertEqual(payload["translated_text"], "译文|after")

    def test_hq_translate_batch_after_hook_can_modify_results(self) -> None:
        plugin = self.register_fixture_plugin()
        mocked_result = SimpleNamespace(
            raw_content='[{"imageIndex":0,"bubbles":[{"bubbleIndex":0,"translated":"译文"}]}]',
            parsed=[{
                "imageIndex": 0,
                "bubbles": [{
                    "bubbleIndex": 0,
                    "translated": "译文",
                }],
            }],
        )

        with mock.patch(
            "src.app.api.translation.routes._hq_executor.execute",
            return_value=mocked_result,
        ):
            response = self.client.post(
                "/api/hq_translate_batch",
                json={
                    "provider": "siliconflow",
                    "api_key": "test-key",
                    "model_name": "gpt-4.1-mini",
                    "jsonData": [{
                        "imageIndex": 0,
                        "bubbles": [{
                            "bubbleIndex": 0,
                            "original": "原文",
                            "translated": "",
                            "textDirection": "vertical",
                        }],
                    }],
                    "imageBase64Array": [create_tiny_png_base64()],
                    "target_language": "zh",
                    "prompt": "translate",
                    "systemPrompt": "system",
                    "isProofreading": False,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["results"][0]["bubbles"][0]["translated"], "译文|after_ai")
        self.assertIn(("ai_translate", "hq"), plugin.seen_contexts)

    def test_re_render_image_route_uses_render_plugin_hooks(self) -> None:
        self.register_fixture_plugin()

        with mock.patch(
            "src.app.api.translation.routes.re_render_with_states",
            return_value=Image.new("RGB", (8, 8), color="white"),
        ):
            response = self.client.post(
                "/api/re_render_image",
                json={
                    "image": create_tiny_png_base64(),
                    "clean_image": create_tiny_png_base64(),
                    "bubble_texts": ["译文"],
                    "bubble_coords": [[0, 0, 4, 4]],
                    "bubble_states": [{
                        "coords": [0, 0, 4, 4],
                        "translatedText": "译文",
                    }],
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["rendered_image"], "plugin-render-image")

    def test_re_render_single_bubble_returns_effective_bubble_states(self) -> None:
        self.register_fixture_plugin()

        with mock.patch(
            "src.app.api.translation.routes.render_single_bubble",
            return_value=Image.new("RGB", (8, 8), color="white"),
        ):
            response = self.client.post(
                "/api/re_render_single_bubble",
                json={
                    "image": create_tiny_png_base64(),
                    "clean_image": create_tiny_png_base64(),
                    "bubble_index": 0,
                    "all_texts": ["译文"],
                    "bubble_coords": [[0, 0, 4, 4]],
                    "bubble_states": [{
                        "coords": [0, 0, 4, 4],
                        "translatedText": "译文",
                        "fontSize": 18,
                    }],
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["rendered_image"], "plugin-render-image")
        self.assertEqual(payload["bubble_states"][0]["translatedText"], "译文")
        self.assertEqual(payload["bubble_states"][0]["fontSize"], 18)

    def test_parallel_inpaint_route_uses_inpaint_plugin_hooks(self) -> None:
        self.register_fixture_plugin()

        with mock.patch(
            "src.app.api.translation.parallel_routes.inpaint_bubbles",
            return_value=(Image.new("RGB", (8, 8), color="white"), None),
        ):
            response = self.client.post(
                "/api/parallel/inpaint",
                json={
                    "image": create_tiny_png_base64(),
                    "bubble_coords": [[0, 0, 4, 4]],
                    "method": "solid",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["clean_image"], "plugin-clean-image")

    def test_parallel_inpaint_noop_result_still_runs_after_hook(self) -> None:
        self.register_fixture_plugin()

        response = self.client.post(
            "/api/parallel/inpaint",
            json={
                "image": create_tiny_png_base64(),
                "bubble_coords": [],
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["clean_image"], "plugin-clean-image")

    def test_parallel_render_route_uses_render_plugin_hooks(self) -> None:
        self.register_fixture_plugin()

        with mock.patch(
            "src.app.api.translation.parallel_routes.render_bubbles_unified",
            side_effect=lambda image, states: image,
        ):
            response = self.client.post(
                "/api/parallel/render",
                json={
                    "clean_image": create_tiny_png_base64(),
                    "bubble_states": [{
                        "coords": [0, 0, 4, 4],
                        "translatedText": "译文",
                    }],
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["final_image"], "plugin-render-image")

    def test_parallel_render_noop_result_still_runs_after_hook(self) -> None:
        self.register_fixture_plugin()

        response = self.client.post(
            "/api/parallel/render",
            json={
                "clean_image": create_tiny_png_base64(),
                "bubble_states": [],
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["final_image"], "plugin-render-image")

    def test_inpaint_single_bubble_route_uses_inpaint_plugin_hooks(self) -> None:
        self.register_fixture_plugin()

        with mock.patch("src.app.api.translation.routes.LAMA_AVAILABLE", True), \
             mock.patch(
                 "src.core.inpainting.inpaint_bubbles",
                 return_value=(Image.new("RGB", (8, 8), color="white"), None),
             ):
            response = self.client.post(
                "/api/inpaint_single_bubble",
                json={
                    "image_data": create_tiny_png_base64(),
                    "bubble_coords": [0, 0, 4, 4],
                    "method": "lama",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["inpainted_image"], "plugin-clean-image")

    def test_apply_settings_to_all_images_route_uses_render_plugin_hooks(self) -> None:
        self.register_fixture_plugin()

        with mock.patch(
            "src.app.api.translation.routes.re_render_text_in_bubbles",
            return_value=Image.new("RGB", (8, 8), color="white"),
        ):
            response = self.client.post(
                "/api/apply_settings_to_all_images",
                json={
                    "all_images": [create_tiny_png_base64()],
                    "all_clean_images": [create_tiny_png_base64()],
                    "all_texts": [["译文"]],
                    "all_bubble_coords": [[[0, 0, 4, 4]]],
                    "fontSize": 18,
                    "fontFamily": "fonts/Arial.ttf",
                    "textDirection": "vertical",
                    "textColor": "#000000",
                    "rotationAngle": 0,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["rendered_images"][0], "plugin-render-image")
