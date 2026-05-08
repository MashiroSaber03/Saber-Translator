import os
import sys
import types
import unittest
import importlib.util
from unittest import mock

from flask import Flask


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda *_args, **_kwargs: {}
    yaml_stub.safe_dump = lambda *_args, **_kwargs: ""
    sys.modules["yaml"] = yaml_stub

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class _OpenAI:
        def __init__(self, *args, **kwargs):
            pass

    openai_stub.OpenAI = _OpenAI
    sys.modules["openai"] = openai_stub

if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
        synchronize=lambda: None,
        memory_allocated=lambda: 0,
        memory_reserved=lambda: 0,
        get_device_name=lambda _index=0: "stub-gpu",
        get_device_properties=lambda _index=0: types.SimpleNamespace(total_memory=0),
    )
    torch_stub.hub = types.SimpleNamespace(set_dir=lambda *_args, **_kwargs: None)
    sys.modules["torch"] = torch_stub


class TranslationRouteConstraintTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_modules = {}
        self._stubbed_module_names = [
            "src.core.rendering",
            "src.core.translation",
            "src.interfaces.lama_interface",
            "src.core.config_models",
            "src.core.ocr",
            "src.core.detection",
            "src.core.ocr_hybrid_manga_48",
            "src.plugins.http_helpers",
            "src.plugins.manager",
            "isolated_translation_pkg",
            "isolated_translation_pkg.routes",
        ]
        for module_name in self._stubbed_module_names:
            self._original_modules[module_name] = sys.modules.get(module_name)

        rendering_stub = types.ModuleType("src.core.rendering")
        rendering_stub.re_render_text_in_bubbles = lambda *args, **kwargs: None
        rendering_stub.render_single_bubble = lambda *args, **kwargs: None
        rendering_stub.re_render_with_states = lambda *args, **kwargs: None
        sys.modules["src.core.rendering"] = rendering_stub

        core_translation_stub = types.ModuleType("src.core.translation")
        core_translation_stub.translate_single_text = lambda *args, **kwargs: ""
        core_translation_stub._enforce_rpm_limit = lambda *args, **kwargs: None
        sys.modules["src.core.translation"] = core_translation_stub

        lama_stub = types.ModuleType("src.interfaces.lama_interface")
        lama_stub.LAMA_AVAILABLE = False
        sys.modules["src.interfaces.lama_interface"] = lama_stub

        config_models_stub = types.ModuleType("src.core.config_models")
        config_models_stub.BubbleState = type("BubbleState", (), {})
        config_models_stub.bubble_states_to_api_response = lambda *_args, **_kwargs: []
        sys.modules["src.core.config_models"] = config_models_stub

        ocr_stub = types.ModuleType("src.core.ocr")
        ocr_stub.recognize_ocr_results_in_bubbles = lambda *args, **kwargs: []
        sys.modules["src.core.ocr"] = ocr_stub

        detection_stub = types.ModuleType("src.core.detection")
        detection_stub.detect_textlines = lambda *args, **kwargs: []
        sys.modules["src.core.detection"] = detection_stub

        hybrid_stub = types.ModuleType("src.core.ocr_hybrid_manga_48")
        hybrid_stub.validate_manga_48_hybrid_combo = lambda *args, **kwargs: None
        sys.modules["src.core.ocr_hybrid_manga_48"] = hybrid_stub

        plugins_stub = types.ModuleType("src.plugins.manager")
        plugins_stub.apply_before_step_hooks = lambda _step, payload, **_kwargs: payload
        plugins_stub.apply_after_step_hooks = lambda _step, result, **_kwargs: result
        sys.modules["src.plugins.manager"] = plugins_stub

        http_helpers_stub = types.ModuleType("src.plugins.http_helpers")
        http_helpers_stub.resolve_plugin_request_context = (
            lambda data, *, default_mode, default_scope: (
                data.get("translation_mode") or data.get("translationMode") or default_mode,
                data.get("translation_scope") or data.get("translationScope") or default_scope,
            )
        )
        http_helpers_stub.prepare_plugin_payload = (
            lambda _step, _route, data, *, default_mode, default_scope, metadata=None: (
                data,
                data.get("translation_mode") or data.get("translationMode") or default_mode,
                data.get("translation_scope") or data.get("translationScope") or default_scope,
            )
        )
        http_helpers_stub.run_before_step_hooks = lambda _step, _route, data, **_kwargs: data
        http_helpers_stub.finalize_plugin_result = lambda _step, _route, result, **_kwargs: result
        http_helpers_stub.run_before_pipeline_hooks = lambda payload, **_kwargs: payload
        http_helpers_stub.run_after_pipeline_hooks = lambda result, **_kwargs: result
        sys.modules["src.plugins.http_helpers"] = http_helpers_stub

        package_dir = os.path.join(PROJECT_ROOT, "src", "app", "api", "translation")
        init_path = os.path.join(package_dir, "__init__.py")
        spec = importlib.util.spec_from_file_location(
            "isolated_translation_pkg",
            init_path,
            submodule_search_locations=[package_dir],
        )
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules["isolated_translation_pkg"] = module
        spec.loader.exec_module(module)

        self.translation_module = sys.modules["isolated_translation_pkg.routes"]

        self.app = Flask(__name__)
        self.app.register_blueprint(module.translate_bp)
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        for module_name in self._stubbed_module_names:
            original = self._original_modules.get(module_name)
            if original is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original

    def test_translate_single_text_restores_non_translate_placeholders_and_returns_warnings(self) -> None:
        captured = {}

        def fake_translate_single_text(**kwargs):
            captured["text"] = kwargs["text"]
            return kwargs["text"]

        with mock.patch.object(self.translation_module, "translate_single_text", side_effect=fake_translate_single_text):
            response = self.client.post(
                "/api/translate_single_text",
                json={
                    "original_text": "Alice <keep>",
                    "target_language": "zh",
                    "model_provider": "siliconflow",
                    "api_key": "test-key",
                    "model_name": "test-model",
                    "glossary_settings": {
                        "enabled": True,
                        "entries": [
                            {"source": "Alice", "target": "爱丽丝", "note": "", "matchMode": "text"}
                        ],
                    },
                    "non_translate_settings": {
                        "enabled": True,
                        "entries": [
                            {"pattern": "<keep>", "note": "占位符", "matchMode": "text"}
                        ],
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["translated_text"], "Alice <keep>")
        self.assertIn("__SABER_NTL_", captured["text"])
        self.assertEqual(payload["warnings"][0]["expectedTarget"], "爱丽丝")

    def test_hq_translate_batch_restores_non_translate_placeholders_in_results(self) -> None:
        captured_request = {}

        def fake_execute(request, **kwargs):
            captured_request["messages"] = request.messages
            return types.SimpleNamespace(
                raw_content='{"images":[]}',
                parsed=[
                    {
                        "imageIndex": 0,
                        "bubbles": [
                            {"bubbleIndex": 0, "translated": "爱丽丝 __SABER_NTL_0001__"}
                        ],
                    }
                ],
            )

        with mock.patch.object(self.translation_module._hq_executor, "execute", side_effect=fake_execute):
            response = self.client.post(
                "/api/hq_translate_batch",
                json={
                    "provider": "siliconflow",
                    "api_key": "test-key",
                    "model_name": "test-model",
                    "jsonData": [
                        {
                            "imageIndex": 0,
                            "bubbles": [
                                {
                                    "bubbleIndex": 0,
                                    "original": "Alice <keep>",
                                    "translated": "",
                                    "textDirection": "vertical",
                                }
                            ],
                        }
                    ],
                    "imageBase64Array": ["ZmFrZQ=="],
                    "prompt": "请翻译",
                    "systemPrompt": "你是翻译助手",
                    "glossary_settings": {
                        "enabled": True,
                        "entries": [
                            {"source": "Alice", "target": "爱丽丝", "note": "", "matchMode": "text"}
                        ],
                    },
                    "non_translate_settings": {
                        "enabled": True,
                        "entries": [
                            {"pattern": "<keep>", "note": "占位符", "matchMode": "text"}
                        ],
                    },
                    "openai_options": {
                        "request": {"force_json_output": True},
                        "execution": {
                            "use_stream": False,
                            "rpm_limit": 0,
                            "transport_retries": 1,
                            "business_retries": 2,
                        },
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["results"][0]["bubbles"][0]["translated"], "爱丽丝 <keep>")
        user_content = captured_request["messages"][1]["content"]
        text_blocks = [item["text"] for item in user_content if isinstance(item, dict) and item.get("type") == "text"]
        joined_text = "\n".join(text_blocks)
        self.assertIn("###术语表", joined_text)
        self.assertIn("###禁翻表", joined_text)
        self.assertIn("###占位符保护规则", joined_text)
        self.assertIn("__SABER_NTL_0001__", joined_text)


if __name__ == "__main__":
    unittest.main()
