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


class TranslateWorkflowPreferencesApiTests(unittest.TestCase):
    def setUp(self) -> None:
        module_path = os.path.join(PROJECT_ROOT, "src", "app", "api", "config_api.py")
        spec = importlib.util.spec_from_file_location("isolated_config_api", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)

        self.app = Flask(__name__)
        self.app.register_blueprint(module.config_bp)
        self.client = self.app.test_client()

    def test_get_preferences_returns_defaults_when_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch("src.shared.config_loader.CONFIG_DIR", temp_dir):
            response = self.client.get("/api/config/translate-workflow-preferences")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {
            "success": True,
            "preferences": {
                "rememberWorkflowModeEnabled": False,
                "lastWorkflowMode": "translate-current",
            },
        })

    def test_save_preferences_round_trips_to_independent_file(self) -> None:
        payload = {
            "rememberWorkflowModeEnabled": True,
            "lastWorkflowMode": "clear-all",
        }

        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch("src.shared.config_loader.CONFIG_DIR", temp_dir):
            post_response = self.client.post(
                "/api/config/translate-workflow-preferences",
                json=payload,
            )
            self.assertEqual(post_response.status_code, 200)
            self.assertEqual(post_response.get_json()["success"], True)

            saved_path = os.path.join(temp_dir, "translate_workflow_preferences.json")
            self.assertTrue(os.path.exists(saved_path))
            with open(saved_path, "r", encoding="utf-8") as file:
                saved_payload = json.load(file)
            self.assertEqual(saved_payload, payload)

            get_response = self.client.get("/api/config/translate-workflow-preferences")
            self.assertEqual(get_response.status_code, 200)
            response_json = get_response.get_json()

        self.assertEqual(response_json["preferences"], payload)

    def test_save_preferences_normalizes_invalid_values(self) -> None:
        payload = {
            "rememberWorkflowModeEnabled": "yes",
            "lastWorkflowMode": "not-a-mode",
        }

        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch("src.shared.config_loader.CONFIG_DIR", temp_dir):
            response = self.client.post(
                "/api/config/translate-workflow-preferences",
                json=payload,
            )
            self.assertEqual(response.status_code, 200)
            saved_path = os.path.join(temp_dir, "translate_workflow_preferences.json")
            with open(saved_path, "r", encoding="utf-8") as file:
                saved_payload = json.load(file)

        self.assertEqual(saved_payload, {
            "rememberWorkflowModeEnabled": False,
            "lastWorkflowMode": "translate-current",
        })

    def test_get_preferences_normalizes_invalid_stored_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch("src.shared.config_loader.CONFIG_DIR", temp_dir):
            saved_path = os.path.join(temp_dir, "translate_workflow_preferences.json")
            with open(saved_path, "w", encoding="utf-8") as file:
                json.dump({
                    "rememberWorkflowModeEnabled": "bad",
                    "lastWorkflowMode": "unknown-mode",
                }, file)

            response = self.client.get("/api/config/translate-workflow-preferences")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {
            "success": True,
            "preferences": {
                "rememberWorkflowModeEnabled": False,
                "lastWorkflowMode": "translate-current",
            },
        })


if __name__ == "__main__":
    unittest.main()
