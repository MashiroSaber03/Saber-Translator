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


class HqTranslationRpmTests(unittest.TestCase):
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
        plugins_stub.apply_after_ocr_hooks = lambda _image, texts, _coords, _params: texts
        sys.modules["src.plugins.manager"] = plugins_stub

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

    def test_hq_translate_batch_applies_rpm_limit_from_request(self) -> None:
        with mock.patch.object(
            self.translation_module._hq_chat_transport,
            "complete",
            return_value='{"images":[]}',
        ), mock.patch.object(
            self.translation_module,
            "_enforce_rpm_limit",
        ) as rpm_mock:
            response = self.client.post(
                "/api/hq_translate_batch",
                json={
                    "provider": "siliconflow",
                    "api_key": "test-key",
                    "model_name": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "rpm_limit": 11,
                    "force_json_output": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        rpm_mock.assert_called()
        self.assertEqual(rpm_mock.call_args.args[0], 11)


if __name__ == "__main__":
    unittest.main()
