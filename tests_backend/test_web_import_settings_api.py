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

if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda *_args, **_kwargs: {}
    yaml_stub.safe_dump = lambda *_args, **_kwargs: ""
    sys.modules["yaml"] = yaml_stub

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class _OpenAI:  # pragma: no cover - import stub only
        def __init__(self, *args, **kwargs):
            pass

    openai_stub.OpenAI = _OpenAI
    sys.modules["openai"] = openai_stub


class WebImportSettingsApiTests(unittest.TestCase):
    def setUp(self) -> None:
        module_path = os.path.join(PROJECT_ROOT, "src", "app", "api", "web_import_api.py")
        spec = importlib.util.spec_from_file_location("isolated_web_import_api", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)

        self.app = Flask(__name__)
        self.app.register_blueprint(module.web_import_bp)
        self.client = self.app.test_client()

    def test_get_settings_returns_empty_defaults_when_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch("src.shared.config_loader.CONFIG_DIR", temp_dir):
            response = self.client.get("/api/web-import/settings")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json(),
            {"success": True, "hasStoredSettings": False, "settings": {}, "providerConfigs": {"agent": {}}},
        )

    def test_save_settings_round_trips_and_normalizes_provider_ids(self) -> None:
        payload = {
            "settings": {
                "agent": {
                    "provider": "custom_openai",
                    "apiKey": "custom-key",
                    "modelName": "custom-model",
                    "customBaseUrl": "https://custom.example/v1",
                },
                "firecrawl": {
                    "apiKey": "fc-123",
                },
            },
            "providerConfigs": {
                "agent": {
                    "custom_openai": {
                        "apiKey": "custom-key",
                        "modelName": "custom-model",
                        "customBaseUrl": "https://custom.example/v1",
                    },
                    "deepseek": {
                        "apiKey": "deepseek-key",
                        "modelName": "deepseek-chat",
                        "customBaseUrl": "https://deepseek.example/v1",
                    },
                },
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch("src.shared.config_loader.CONFIG_DIR", temp_dir):
            post_response = self.client.post("/api/web-import/settings", json=payload)
            self.assertEqual(post_response.status_code, 200)
            self.assertEqual(post_response.get_json()["success"], True)

            saved_path = os.path.join(temp_dir, "web_import_settings.json")
            self.assertTrue(os.path.exists(saved_path))

            with open(saved_path, "r", encoding="utf-8") as file:
                saved_payload = json.load(file)

            self.assertEqual(saved_payload["settings"]["agent"]["provider"], "custom")
            self.assertIn("custom", saved_payload["providerConfigs"]["agent"])
            self.assertNotIn("custom_openai", saved_payload["providerConfigs"]["agent"])

            get_response = self.client.get("/api/web-import/settings")
            self.assertEqual(get_response.status_code, 200)
            response_json = get_response.get_json()

        self.assertEqual(response_json["settings"]["agent"]["provider"], "custom")
        self.assertEqual(response_json["hasStoredSettings"], True)
        self.assertEqual(response_json["providerConfigs"]["agent"]["custom"]["modelName"], "custom-model")
        self.assertEqual(response_json["providerConfigs"]["agent"]["deepseek"]["modelName"], "deepseek-chat")


if __name__ == "__main__":
    unittest.main()
