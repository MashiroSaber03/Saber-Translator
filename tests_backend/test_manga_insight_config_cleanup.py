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

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class _OpenAI:  # pragma: no cover - import stub only
        def __init__(self, *args, **kwargs):
            pass

    openai_stub.OpenAI = _OpenAI
    sys.modules["openai"] = openai_stub


class MangaInsightConfigCleanupTests(unittest.TestCase):
    def test_vlm_prompt_builder_uses_updated_context_batch_fallback(self) -> None:
        from src.core.manga_insight.config_models import PromptsConfig
        from src.core.manga_insight.vlm_client import VLMClient

        client = VLMClient.__new__(VLMClient)
        client.prompts_config = PromptsConfig()

        prompt = client._build_batch_analysis_prompt(
            start_page=1,
            end_page=5,
            page_count=5,
            context={"previous_summary": "上一批剧情摘要"},
        )

        self.assertIn("前3批内容", prompt)

    def test_default_config_uses_updated_factory_defaults(self) -> None:
        from src.core.manga_insight.config_models import MangaInsightConfig

        config = MangaInsightConfig()

        self.assertEqual(config.vlm.openai_options.execution.rpm_limit, 0)
        self.assertEqual(config.vlm.openai_options.execution.transport_retries, 10)
        self.assertEqual(config.vlm.openai_options.execution.business_retries, 10)
        self.assertEqual(config.vlm.image_max_size, 1280)

        self.assertFalse(config.chat_llm.use_same_as_vlm)
        self.assertEqual(config.chat_llm.openai_options.execution.rpm_limit, 0)
        self.assertEqual(config.chat_llm.openai_options.execution.transport_retries, 10)
        self.assertEqual(config.chat_llm.openai_options.execution.business_retries, 10)

        self.assertEqual(config.analysis.batch.context_batch_count, 3)
        self.assertEqual(config.embedding.transport_retries, 10)
        self.assertEqual(config.embedding.business_retries, 10)
        self.assertEqual(config.embedding.timeout_seconds, 0)
        self.assertEqual(config.reranker.transport_retries, 10)
        self.assertEqual(config.reranker.business_retries, 10)
        self.assertEqual(config.reranker.timeout_seconds, 0)
        self.assertEqual(config.image_gen.transport_retries, 10)
        self.assertEqual(config.image_gen.business_retries, 10)
        self.assertEqual(config.image_gen.timeout_seconds, 0)

    def test_to_dict_omits_removed_runtime_only_fields(self) -> None:
        from src.core.manga_insight.config_models import MangaInsightConfig

        payload = MangaInsightConfig().to_dict()

        self.assertNotIn("max_retries", payload["vlm"])
        self.assertNotIn("max_images_per_request", payload["vlm"])
        self.assertNotIn("rpm_limit", payload["chat_llm"])
        self.assertNotIn("max_retries", payload["chat_llm"])
        self.assertNotIn("dimension", payload["embedding"])
        self.assertNotIn("max_retries", payload["embedding"])
        self.assertEqual(payload["embedding"]["transport_retries"], 10)
        self.assertEqual(payload["embedding"]["business_retries"], 10)
        self.assertEqual(payload["embedding"]["timeout_seconds"], 0)
        self.assertNotIn("enabled", payload["reranker"])
        self.assertNotIn("rpm_limit", payload["reranker"])
        self.assertNotIn("max_retries", payload["reranker"])
        self.assertEqual(payload["reranker"]["transport_retries"], 10)
        self.assertEqual(payload["reranker"]["business_retries"], 10)
        self.assertEqual(payload["reranker"]["timeout_seconds"], 0)
        self.assertNotIn("max_retries", payload["image_gen"])
        self.assertEqual(payload["image_gen"]["transport_retries"], 10)
        self.assertEqual(payload["image_gen"]["business_retries"], 10)
        self.assertEqual(payload["image_gen"]["timeout_seconds"], 0)
        self.assertNotIn("rpm_limit", payload["vlm"])
        self.assertNotIn("temperature", payload["vlm"])
        self.assertNotIn("force_json", payload["vlm"])
        self.assertNotIn("use_stream", payload["vlm"])
        self.assertNotIn("use_stream", payload["chat_llm"])

    def test_from_dict_ignores_removed_runtime_fields(self) -> None:
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
                    "transport_retries": 6,
                    "business_retries": 7,
                    "timeout_seconds": 0,
                },
                "reranker": {
                    "provider": "jina",
                    "api_key": "key",
                    "model": "jina-reranker-v2-base-multilingual",
                    "enabled": False,
                    "rpm_limit": 12,
                    "max_retries": 7,
                    "transport_retries": 3,
                    "business_retries": 4,
                    "timeout_seconds": 0,
                },
                "image_gen": {
                    "provider": "gpt2api",
                    "api_key": "key",
                    "model": "gpt-image-2",
                    "base_url": "https://gateway.example.com/v1",
                    "max_retries": 5,
                    "transport_retries": 6,
                    "business_retries": 7,
                    "timeout_seconds": 0,
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
        self.assertEqual(serialized["embedding"]["transport_retries"], 6)
        self.assertEqual(serialized["embedding"]["business_retries"], 7)
        self.assertEqual(serialized["embedding"]["timeout_seconds"], 0)
        self.assertNotIn("enabled", serialized["reranker"])
        self.assertNotIn("rpm_limit", serialized["reranker"])
        self.assertNotIn("max_retries", serialized["reranker"])
        self.assertEqual(serialized["reranker"]["transport_retries"], 3)
        self.assertEqual(serialized["reranker"]["business_retries"], 4)
        self.assertEqual(serialized["reranker"]["timeout_seconds"], 0)
        self.assertNotIn("max_retries", serialized["image_gen"])
        self.assertEqual(serialized["image_gen"]["transport_retries"], 6)
        self.assertEqual(serialized["image_gen"]["business_retries"], 7)
        self.assertEqual(serialized["image_gen"]["timeout_seconds"], 0)
        self.assertNotIn("rpm_limit", serialized["vlm"])
        self.assertNotIn("temperature", serialized["vlm"])
        self.assertNotIn("force_json", serialized["vlm"])
        self.assertNotIn("use_stream", serialized["vlm"])
        self.assertNotIn("use_stream", serialized["chat_llm"])
        self.assertFalse(hasattr(config.vlm, "force_json"))
        self.assertFalse(hasattr(config.vlm, "use_stream"))
        self.assertFalse(hasattr(config.chat_llm, "use_stream"))
