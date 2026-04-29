import os
import sys
import types
import unittest


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
                },
                "chat_llm": {
                    "provider": "gemini",
                    "api_key": "key",
                    "model": "gemini-2.0-flash",
                    "rpm_limit": 123,
                    "max_retries": 6,
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
