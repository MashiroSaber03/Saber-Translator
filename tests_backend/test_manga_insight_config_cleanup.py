import os
import sys
import types
import unittest
from unittest import mock


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda *_args, **_kwargs: {}
    yaml_stub.safe_dump = lambda *_args, **_kwargs: ""
    sys.modules["yaml"] = yaml_stub


class MangaInsightConfigCleanupTests(unittest.TestCase):
    def test_to_dict_omits_removed_runtime_only_fields(self) -> None:
        from src.core.manga_insight.config_models import MangaInsightConfig

        payload = MangaInsightConfig().to_dict()

        self.assertNotIn("max_retries", payload["vlm"])
        self.assertNotIn("max_images_per_request", payload["vlm"])
        self.assertNotIn("rpm_limit", payload["chat_llm"])
        self.assertNotIn("max_retries", payload["chat_llm"])
        self.assertNotIn("dimension", payload["embedding"])
        self.assertNotIn("max_retries", payload["embedding"])
        self.assertNotIn("enabled", payload["reranker"])
        self.assertNotIn("rpm_limit", payload["reranker"])
        self.assertNotIn("max_retries", payload["reranker"])
        self.assertNotIn("rpm_limit", payload["vlm"])
        self.assertNotIn("temperature", payload["vlm"])
        self.assertNotIn("force_json", payload["vlm"])
        self.assertNotIn("use_stream", payload["vlm"])
        self.assertNotIn("use_stream", payload["chat_llm"])

    def test_from_dict_ignores_removed_legacy_fields(self) -> None:
        from src.core.manga_insight.config_models import MangaInsightConfig

        config = MangaInsightConfig.from_dict(
            {
                "vlm": {
                    "provider": "gemini",
                    "api_key": "key",
                    "model": "gemini-2.0-flash",
                    "max_retries": 9,
                    "max_images_per_request": 4,
                    "rpm_limit": 12,
                    "temperature": 0.6,
                    "force_json": True,
                    "use_stream": False,
                },
                "chat_llm": {
                    "provider": "gemini",
                    "api_key": "key",
                    "model": "gemini-2.0-flash",
                    "rpm_limit": 123,
                    "max_retries": 6,
                    "use_stream": False,
                },
                "embedding": {
                    "provider": "openai",
                    "api_key": "key",
                    "model": "text-embedding-3-small",
                    "dimension": 3072,
                    "max_retries": 8,
                },
                "reranker": {
                    "provider": "jina",
                    "api_key": "key",
                    "model": "jina-reranker-v2-base-multilingual",
                    "enabled": False,
                    "rpm_limit": 12,
                    "max_retries": 7,
                },
            }
        )

        serialized = config.to_dict()
        self.assertNotIn("max_retries", serialized["vlm"])
        self.assertNotIn("max_images_per_request", serialized["vlm"])
        self.assertNotIn("rpm_limit", serialized["chat_llm"])
        self.assertNotIn("max_retries", serialized["chat_llm"])
        self.assertNotIn("dimension", serialized["embedding"])
        self.assertNotIn("max_retries", serialized["embedding"])
        self.assertNotIn("enabled", serialized["reranker"])
        self.assertNotIn("rpm_limit", serialized["reranker"])
        self.assertNotIn("max_retries", serialized["reranker"])
        self.assertNotIn("rpm_limit", serialized["vlm"])
        self.assertNotIn("temperature", serialized["vlm"])
        self.assertNotIn("force_json", serialized["vlm"])
        self.assertNotIn("use_stream", serialized["vlm"])
        self.assertNotIn("use_stream", serialized["chat_llm"])
        self.assertFalse(hasattr(config.vlm, "force_json"))
        self.assertFalse(hasattr(config.vlm, "use_stream"))
        self.assertFalse(hasattr(config.chat_llm, "use_stream"))

    def test_load_insight_config_migrates_legacy_openai_fields_and_rewrites_file(self) -> None:
        from src.core.manga_insight.config_utils import load_insight_config

        legacy_payload = {
            "vlm": {
                "provider": "custom",
                "api_key": "key",
                "model": "vlm-model",
                "base_url": "https://example.com/v1",
                "rpm_limit": 12,
                "temperature": 0.6,
                "force_json": True,
                "use_stream": False,
                "max_retries": 4,
            },
            "chat_llm": {
                "provider": "custom",
                "api_key": "key",
                "model": "chat-model",
                "base_url": "https://example.com/v1",
                "use_stream": False,
            },
        }

        with mock.patch(
            "src.core.manga_insight.config_utils.load_json_config",
            return_value=legacy_payload,
        ), mock.patch(
            "src.core.manga_insight.config_utils.save_json_config",
            return_value=True,
        ) as save_mock:
            config = load_insight_config()

        self.assertEqual(config.vlm.openai_options.execution.rpm_limit, 12)
        self.assertEqual(config.vlm.openai_options.request.temperature, 0.6)
        self.assertTrue(config.vlm.openai_options.request.force_json_output)
        self.assertFalse(config.vlm.openai_options.execution.use_stream)
        self.assertEqual(config.vlm.openai_options.execution.business_retries, 4)
        self.assertFalse(config.chat_llm.openai_options.execution.use_stream)
        save_mock.assert_called_once()
        saved_payload = save_mock.call_args.args[1]
        self.assertEqual(saved_payload["schema_version"], 2)
        self.assertEqual(saved_payload["vlm"]["openai_options"]["execution"]["rpm_limit"], 12)
        self.assertNotIn("force_json", saved_payload["vlm"])
        self.assertNotIn("use_stream", saved_payload["vlm"])
        self.assertNotIn("rpm_limit", saved_payload["vlm"])
        self.assertNotIn("temperature", saved_payload["vlm"])
        self.assertNotIn("use_stream", saved_payload["chat_llm"])
