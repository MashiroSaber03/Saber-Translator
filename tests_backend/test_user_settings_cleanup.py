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


class UserSettingsCleanupTests(unittest.TestCase):
    def setUp(self) -> None:
        module_path = os.path.join(PROJECT_ROOT, "src", "app", "api", "config_api.py")
        spec = importlib.util.spec_from_file_location("isolated_config_api", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)

        self.app = Flask(__name__)
        self.app.register_blueprint(module.config_bp)
        self.client = self.app.test_client()

    def test_save_settings_strips_deprecated_hq_session_reset_fields(self) -> None:
        payload = {
            "settings": {
                "hqSessionReset": "3",
                "hqRpmLimit": "7",
                "proofreading": {
                    "enabled": True,
                    "rounds": [
                        {
                            "name": "第1轮",
                            "provider": "siliconflow",
                            "sessionReset": 2,
                            "rpmLimit": 9,
                        }
                    ],
                },
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch("src.shared.config_loader.CONFIG_DIR", temp_dir):
            response = self.client.post("/api/save_settings", json=payload)
            self.assertEqual(response.status_code, 200)

            settings_path = os.path.join(temp_dir, "user_settings.json")
            with open(settings_path, "r", encoding="utf-8") as file:
                saved = json.load(file)

        self.assertNotIn("hqSessionReset", saved)
        self.assertEqual(saved["hqRpmLimit"], "7")
        self.assertNotIn("sessionReset", saved["proofreading"]["rounds"][0])
        self.assertEqual(saved["proofreading"]["rounds"][0]["rpmLimit"], 9)

    def test_get_settings_ignores_deprecated_session_reset_fields_from_existing_file(self) -> None:
        existing_payload = {
            "hqSessionReset": "4",
            "hqRpmLimit": "8",
            "proofreading": {
                "enabled": True,
                "rounds": [
                    {
                        "name": "第1轮",
                        "provider": "siliconflow",
                        "sessionReset": 5,
                        "rpmLimit": 6,
                    }
                ],
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch("src.shared.config_loader.CONFIG_DIR", temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
            with open(os.path.join(temp_dir, "user_settings.json"), "w", encoding="utf-8") as file:
                json.dump(existing_payload, file, ensure_ascii=False, indent=2)

            response = self.client.get("/api/get_settings")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        settings = payload["settings"]
        self.assertNotIn("hqSessionReset", settings)
        self.assertEqual(settings["hqRpmLimit"], "8")
        self.assertNotIn("sessionReset", settings["proofreading"]["rounds"][0])
        self.assertEqual(settings["proofreading"]["rounds"][0]["rpmLimit"], 6)


if __name__ == "__main__":
    unittest.main()
