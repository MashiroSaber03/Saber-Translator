import json
import os
import sys
import tempfile
import types
import unittest
import importlib.util
from unittest import mock
from flask import Flask


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import yaml  # type: ignore # noqa: F401
except ModuleNotFoundError:
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda *_args, **_kwargs: {}
    yaml_stub.safe_dump = lambda *_args, **_kwargs: ""
    sys.modules["yaml"] = yaml_stub


from src.core.config_models import BubbleState
from src.shared import text_style_defaults
from src.shared.text_style_defaults import get_text_style_defaults


class TextStyleDefaultsLoaderTests(unittest.TestCase):
    def tearDown(self) -> None:
        text_style_defaults.load_text_style_defaults.cache_clear()
        if hasattr(text_style_defaults, "load_text_style_factory_defaults"):
            text_style_defaults.load_text_style_factory_defaults.cache_clear()

    def test_loader_returns_exact_shared_defaults(self) -> None:
        defaults_path = os.path.join(PROJECT_ROOT, "config", "text_style_defaults.json")
        with open(defaults_path, "r", encoding="utf-8") as file:
            expected_defaults = json.load(file)

        self.assertEqual(get_text_style_defaults(), expected_defaults)

    def test_loader_bootstraps_missing_defaults_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_defaults_path = os.path.join(temp_dir, "config", "text_style_defaults.json")
            factory_defaults_path = os.path.join(temp_dir, "shared", "text_style_defaults_factory.json")
            text_style_defaults.load_text_style_defaults.cache_clear()

            with open(os.path.join(PROJECT_ROOT, "src", "shared", "text_style_defaults_factory.json"), "r", encoding="utf-8") as file:
                factory_defaults = json.load(file)

            factory_defaults["fontSize"] = 41
            os.makedirs(os.path.dirname(factory_defaults_path), exist_ok=True)
            with open(factory_defaults_path, "w", encoding="utf-8") as file:
                json.dump(factory_defaults, file, indent=2, ensure_ascii=False)

            with mock.patch.object(text_style_defaults, "TEXT_STYLE_DEFAULTS_PATH", missing_defaults_path), \
                 mock.patch.object(text_style_defaults, "TEXT_STYLE_FACTORY_DEFAULTS_PATH", factory_defaults_path):
                defaults = text_style_defaults.get_text_style_defaults()

            self.assertTrue(os.path.exists(missing_defaults_path))
            with open(missing_defaults_path, "r", encoding="utf-8") as file:
                created_defaults = json.load(file)

            self.assertEqual(defaults, created_defaults)
            self.assertEqual(created_defaults, factory_defaults)

    def test_config_api_returns_current_defaults(self) -> None:
        module_path = os.path.join(PROJECT_ROOT, "src", "app", "api", "config_api.py")
        spec = importlib.util.spec_from_file_location("isolated_config_api", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)

        app = Flask(__name__)
        app.register_blueprint(module.config_bp)
        client = app.test_client()

        response = client.get("/api/config/text-style-defaults")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"success": True, "defaults": get_text_style_defaults()})

    def test_config_api_refreshes_defaults_after_file_change(self) -> None:
        module_path = os.path.join(PROJECT_ROOT, "src", "app", "api", "config_api.py")
        spec = importlib.util.spec_from_file_location("isolated_config_api", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)

        app = Flask(__name__)
        app.register_blueprint(module.config_bp)
        client = app.test_client()

        with tempfile.TemporaryDirectory() as temp_dir:
            defaults_path = os.path.join(temp_dir, "config", "text_style_defaults.json")
            os.makedirs(os.path.dirname(defaults_path), exist_ok=True)

            with open(os.path.join(PROJECT_ROOT, "config", "text_style_defaults.json"), "r", encoding="utf-8") as file:
                defaults = json.load(file)

            defaults["fontSize"] = 33
            with open(defaults_path, "w", encoding="utf-8") as file:
                json.dump(defaults, file, indent=2, ensure_ascii=False)

            text_style_defaults.load_text_style_defaults.cache_clear()
            with mock.patch.object(text_style_defaults, "TEXT_STYLE_DEFAULTS_PATH", defaults_path):
                response = client.get("/api/config/text-style-defaults")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json(), {"success": True, "defaults": defaults})

    def test_config_api_saves_updated_defaults(self) -> None:
        module_path = os.path.join(PROJECT_ROOT, "src", "app", "api", "config_api.py")
        spec = importlib.util.spec_from_file_location("isolated_config_api", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)

        app = Flask(__name__)
        app.register_blueprint(module.config_bp)
        client = app.test_client()

        with tempfile.TemporaryDirectory() as temp_dir:
            defaults_path = os.path.join(temp_dir, "config", "text_style_defaults.json")
            factory_defaults_path = os.path.join(temp_dir, "shared", "text_style_defaults_factory.json")
            os.makedirs(os.path.dirname(defaults_path), exist_ok=True)
            os.makedirs(os.path.dirname(factory_defaults_path), exist_ok=True)

            with open(os.path.join(PROJECT_ROOT, "config", "text_style_defaults.json"), "r", encoding="utf-8") as file:
                current_defaults = json.load(file)

            with open(factory_defaults_path, "w", encoding="utf-8") as file:
                json.dump(current_defaults, file, indent=2, ensure_ascii=False)
            with open(defaults_path, "w", encoding="utf-8") as file:
                json.dump(current_defaults, file, indent=2, ensure_ascii=False)

            updated_defaults = {**current_defaults, "fontSize": 34, "textColor": "#123456"}

            text_style_defaults.load_text_style_defaults.cache_clear()
            with mock.patch.object(text_style_defaults, "TEXT_STYLE_DEFAULTS_PATH", defaults_path), \
                 mock.patch.object(text_style_defaults, "TEXT_STYLE_FACTORY_DEFAULTS_PATH", factory_defaults_path):
                response = client.post("/api/config/text-style-defaults", json={"defaults": updated_defaults})

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json(), {"success": True, "defaults": updated_defaults})

            with open(defaults_path, "r", encoding="utf-8") as file:
                saved_defaults = json.load(file)

            self.assertEqual(saved_defaults, updated_defaults)

    def test_config_api_rejects_invalid_defaults_without_overwriting_file(self) -> None:
        module_path = os.path.join(PROJECT_ROOT, "src", "app", "api", "config_api.py")
        spec = importlib.util.spec_from_file_location("isolated_config_api", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)

        app = Flask(__name__)
        app.register_blueprint(module.config_bp)
        client = app.test_client()

        with tempfile.TemporaryDirectory() as temp_dir:
            defaults_path = os.path.join(temp_dir, "config", "text_style_defaults.json")
            factory_defaults_path = os.path.join(temp_dir, "shared", "text_style_defaults_factory.json")
            os.makedirs(os.path.dirname(defaults_path), exist_ok=True)
            os.makedirs(os.path.dirname(factory_defaults_path), exist_ok=True)

            with open(os.path.join(PROJECT_ROOT, "config", "text_style_defaults.json"), "r", encoding="utf-8") as file:
                current_defaults = json.load(file)

            with open(factory_defaults_path, "w", encoding="utf-8") as file:
                json.dump(current_defaults, file, indent=2, ensure_ascii=False)
            with open(defaults_path, "w", encoding="utf-8") as file:
                json.dump(current_defaults, file, indent=2, ensure_ascii=False)

            invalid_defaults = {**current_defaults, "strokeWidth": -1}

            text_style_defaults.load_text_style_defaults.cache_clear()
            with mock.patch.object(text_style_defaults, "TEXT_STYLE_DEFAULTS_PATH", defaults_path), \
                 mock.patch.object(text_style_defaults, "TEXT_STYLE_FACTORY_DEFAULTS_PATH", factory_defaults_path):
                response = client.post("/api/config/text-style-defaults", json={"defaults": invalid_defaults})

            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.get_json()["success"], False)

            with open(defaults_path, "r", encoding="utf-8") as file:
                saved_defaults = json.load(file)

            self.assertEqual(saved_defaults, current_defaults)

    def test_config_api_reset_restores_factory_defaults(self) -> None:
        module_path = os.path.join(PROJECT_ROOT, "src", "app", "api", "config_api.py")
        spec = importlib.util.spec_from_file_location("isolated_config_api", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)

        app = Flask(__name__)
        app.register_blueprint(module.config_bp)
        client = app.test_client()

        with tempfile.TemporaryDirectory() as temp_dir:
            defaults_path = os.path.join(temp_dir, "config", "text_style_defaults.json")
            factory_defaults_path = os.path.join(temp_dir, "shared", "text_style_defaults_factory.json")
            os.makedirs(os.path.dirname(defaults_path), exist_ok=True)
            os.makedirs(os.path.dirname(factory_defaults_path), exist_ok=True)

            with open(os.path.join(PROJECT_ROOT, "config", "text_style_defaults.json"), "r", encoding="utf-8") as file:
                current_defaults = json.load(file)

            factory_defaults = {**current_defaults, "fontSize": 44, "fillColor": "#EEEEEE"}
            current_defaults["fontSize"] = 19

            with open(factory_defaults_path, "w", encoding="utf-8") as file:
                json.dump(factory_defaults, file, indent=2, ensure_ascii=False)
            with open(defaults_path, "w", encoding="utf-8") as file:
                json.dump(current_defaults, file, indent=2, ensure_ascii=False)

            text_style_defaults.load_text_style_defaults.cache_clear()
            with mock.patch.object(text_style_defaults, "TEXT_STYLE_DEFAULTS_PATH", defaults_path), \
                 mock.patch.object(text_style_defaults, "TEXT_STYLE_FACTORY_DEFAULTS_PATH", factory_defaults_path):
                response = client.post("/api/config/text-style-defaults/reset")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json(), {"success": True, "defaults": factory_defaults})

            with open(defaults_path, "r", encoding="utf-8") as file:
                saved_defaults = json.load(file)

            self.assertEqual(saved_defaults, factory_defaults)


class BubbleStateDefaultsTests(unittest.TestCase):
    def test_bubble_state_defaults_match_canonical_text_style_defaults(self) -> None:
        defaults = get_text_style_defaults()
        state = BubbleState()

        self.assertEqual(state.font_size, defaults["fontSize"])
        self.assertEqual(
            state.font_family,
            os.path.join("src", "app", "static", defaults["fontFamily"].replace("/", os.sep)),
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
