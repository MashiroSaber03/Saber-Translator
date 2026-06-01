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

    def test_load_insight_config_uses_updated_defaults_when_openai_options_are_missing(self) -> None:
        from src.core.manga_insight.config_utils import load_insight_config

        payload = {
            "vlm": {
                "provider": "gemini",
                "api_key": "key",
                "model": "gemini-2.0-flash",
            },
            "chat_llm": {
                "provider": "gemini",
                "api_key": "key",
                "model": "gemini-2.0-flash",
            },
        }

        with mock.patch(
            "src.core.manga_insight.config_utils.load_json_config",
            return_value=payload,
        ), mock.patch(
            "src.core.manga_insight.config_utils.save_json_config",
            return_value=True,
        ):
            config = load_insight_config()

        self.assertEqual(config.vlm.openai_options.execution.rpm_limit, 0)
        self.assertEqual(config.vlm.openai_options.execution.transport_retries, 10)
        self.assertEqual(config.vlm.openai_options.execution.business_retries, 10)
        self.assertEqual(config.chat_llm.openai_options.execution.rpm_limit, 0)
        self.assertEqual(config.chat_llm.openai_options.execution.transport_retries, 10)
        self.assertEqual(config.chat_llm.openai_options.execution.business_retries, 10)

    def test_load_insight_config_preserves_future_image_gen_provider_without_rewriting(self) -> None:
        from src.core.manga_insight.config_utils import load_insight_config

        payload = {
            "image_gen": {
                "provider": "future-image-provider",
                "api_key": "future-key",
                "model": "future-image-model",
                "base_url": "https://gateway.example.com/v1",
                "max_retries": 5,
            }
        }

        with mock.patch(
            "src.core.manga_insight.config_utils.load_json_config",
            return_value=payload,
        ), mock.patch(
            "src.core.manga_insight.config_utils.save_json_config",
            return_value=True,
        ) as save_mock:
            config = load_insight_config()

        self.assertEqual(config.image_gen.provider, "future-image-provider")
        self.assertEqual(config.image_gen.model, "future-image-model")
        self.assertEqual(config.image_gen.api_key, "future-key")
        self.assertEqual(config.image_gen.base_url, "https://gateway.example.com/v1")
        self.assertEqual(config.image_gen.transport_retries, 10)
        self.assertEqual(config.image_gen.business_retries, 5)
        self.assertEqual(config.image_gen.timeout_seconds, 0)
        if save_mock.called:
            saved_payload = save_mock.call_args.args[1]
            self.assertEqual(saved_payload["image_gen"]["provider"], "future-image-provider")
            self.assertEqual(saved_payload["image_gen"]["model"], "future-image-model")
            self.assertNotIn("max_retries", saved_payload["image_gen"])
            self.assertEqual(saved_payload["image_gen"]["transport_retries"], 10)
            self.assertEqual(saved_payload["image_gen"]["business_retries"], 5)
            self.assertEqual(saved_payload["image_gen"]["timeout_seconds"], 0)

    def test_save_insight_config_preserves_future_image_gen_provider(self) -> None:
        from src.core.manga_insight.config_utils import save_insight_config

        payload = {
            "image_gen": {
                "provider": "future-image-provider",
                "api_key": "future-key",
                "model": "future-image-model",
                "base_url": "https://gateway.example.com/v1",
                "transport_retries": 10,
                "business_retries": 3,
                "timeout_seconds": 0,
            }
        }

        with mock.patch(
            "src.core.manga_insight.config_utils.save_json_config",
            return_value=True,
        ) as save_mock:
            ok = save_insight_config(payload)

        self.assertTrue(ok)
        saved_payload = save_mock.call_args.args[1]
        self.assertEqual(saved_payload["image_gen"]["provider"], "future-image-provider")
        self.assertEqual(saved_payload["image_gen"]["model"], "future-image-model")
        self.assertEqual(saved_payload["image_gen"]["transport_retries"], 10)
        self.assertEqual(saved_payload["image_gen"]["business_retries"], 3)
        self.assertEqual(saved_payload["image_gen"]["timeout_seconds"], 0)

    def test_validate_config_rejects_negative_reranker_and_image_gen_runtime_values(self) -> None:
        from src.core.manga_insight.config_models import MangaInsightConfig
        from src.core.manga_insight.config_utils import validate_config

        config = MangaInsightConfig()
        config.reranker.transport_retries = -1
        config.reranker.business_retries = -2
        config.reranker.timeout_seconds = -3
        config.image_gen.transport_retries = -4
        config.image_gen.business_retries = -5
        config.image_gen.timeout_seconds = -6

        errors = validate_config(config)

        self.assertIn("Reranker transport_retries 不能为负数", errors)
        self.assertIn("Reranker business_retries 不能为负数", errors)
        self.assertIn("Reranker timeout_seconds 不能为负数", errors)
        self.assertIn("ImageGen transport_retries 不能为负数", errors)
        self.assertIn("ImageGen business_retries 不能为负数", errors)
        self.assertIn("ImageGen timeout_seconds 不能为负数", errors)

    def test_validate_config_warns_when_image_gen_provider_requires_model_but_model_is_empty(self) -> None:
        from src.core.manga_insight.config_models import MangaInsightConfig
        from src.core.manga_insight.config_utils import validate_config

        config = MangaInsightConfig()
        config.image_gen.provider = "newapi"
        config.image_gen.api_key = "image-key"
        config.image_gen.base_url = "https://newapi.example.com/v1"
        config.image_gen.model = ""

        issues = validate_config(config)

        self.assertIn("ImageGen 已选择服务商但未选择模型", issues)
