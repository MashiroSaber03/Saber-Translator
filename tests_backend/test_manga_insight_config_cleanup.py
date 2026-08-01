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

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class _OpenAI:  # pragma: no cover - import stub only
        def __init__(self, *args, **kwargs):
            pass

    openai_stub.OpenAI = _OpenAI
    sys.modules["openai"] = openai_stub


class MangaInsightConfigCleanupTests(unittest.TestCase):
    def test_vlm_prompt_builder_appends_updated_context_batch_count(self) -> None:
        from src.core.manga_insight.config_models import PromptsConfig
        from src.core.manga_insight.vlm_client import VLMClient

        client = VLMClient.__new__(VLMClient)
        client.prompts_config = PromptsConfig(
            batch_analysis=(
                "分析第 {start_page}-{end_page} 页，"
                "共 {page_count} 页。"
            )
        )

        prompt = client._build_batch_analysis_prompt(
            start_page=1,
            end_page=5,
            page_count=5,
            context={"previous_summary": "上一批剧情摘要"},
        )

        self.assertIn("前3批内容", prompt)

    def test_runtime_model_configs_use_current_defaults(self) -> None:
        from src.core.manga_insight.config_models import (
            ChatLLMConfig,
            EmbeddingConfig,
            ImageGenConfig,
            VLMConfig,
        )

        vlm = VLMConfig()
        chat = ChatLLMConfig()
        embedding = EmbeddingConfig()
        image_gen = ImageGenConfig()

        self.assertEqual(vlm.openai_options.execution.rpm_limit, 0)
        self.assertEqual(vlm.openai_options.execution.transport_retries, 10)
        self.assertEqual(vlm.openai_options.execution.business_retries, 10)
        self.assertEqual(vlm.image_max_size, 1280)
        self.assertFalse(chat.use_same_as_vlm)
        self.assertEqual(chat.openai_options.execution.rpm_limit, 0)
        self.assertEqual(chat.openai_options.execution.transport_retries, 10)
        self.assertEqual(chat.openai_options.execution.business_retries, 10)
        self.assertEqual(embedding.transport_retries, 10)
        self.assertEqual(embedding.business_retries, 10)
        self.assertEqual(embedding.timeout_seconds, 0)
        self.assertEqual(image_gen.transport_retries, 10)
        self.assertEqual(image_gen.business_retries, 10)
        self.assertEqual(image_gen.timeout_seconds, 0)

    def test_runtime_model_configs_ignore_removed_fields(self) -> None:
        from src.core.manga_insight.config_models import (
            ChatLLMConfig,
            EmbeddingConfig,
            ImageGenConfig,
            VLMConfig,
        )

        cases = (
            (
                VLMConfig,
                {"max_retries": 9, "rpm_limit": 12, "use_stream": False},
                {"max_retries", "rpm_limit", "use_stream"},
            ),
            (
                ChatLLMConfig,
                {"max_retries": 6, "rpm_limit": 123, "use_stream": False},
                {"max_retries", "rpm_limit", "use_stream"},
            ),
            (
                EmbeddingConfig,
                {"dimension": 3072, "max_retries": 8},
                {"dimension", "max_retries"},
            ),
            (
                ImageGenConfig,
                {"max_retries": 5},
                {"max_retries"},
            ),
        )

        for config_type, raw, removed in cases:
            with self.subTest(config=config_type.__name__):
                serialized = config_type.from_dict(raw).to_dict()
                self.assertTrue(removed.isdisjoint(serialized))
