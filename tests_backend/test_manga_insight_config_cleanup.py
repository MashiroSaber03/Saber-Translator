import unittest


class MangaInsightConfigCleanupTests(unittest.TestCase):
    def test_runtime_model_configs_use_current_defaults(self) -> None:
        from src.core.manga_insight.config_models import (
            ChatLLMConfig,
            EmbeddingConfig,
            ImageGenConfig,
            RerankerConfig,
            VLMConfig,
        )

        vlm = VLMConfig()
        chat = ChatLLMConfig()
        embedding = EmbeddingConfig()
        reranker = RerankerConfig()
        image_gen = ImageGenConfig()

        self.assertEqual(vlm.openai_options.execution.rpm_limit, 0)
        self.assertEqual(vlm.openai_options.execution.transport_retries, 1)
        self.assertEqual(vlm.openai_options.execution.business_retries, 0)
        self.assertEqual(vlm.image_max_size, 0)
        self.assertEqual(chat.openai_options.execution.rpm_limit, 0)
        self.assertEqual(chat.openai_options.execution.transport_retries, 1)
        self.assertEqual(chat.openai_options.execution.business_retries, 0)
        self.assertEqual(embedding.transport_retries, 1)
        self.assertEqual(embedding.business_retries, 0)
        self.assertEqual(embedding.timeout_seconds, 0)
        self.assertEqual(reranker.transport_retries, 1)
        self.assertEqual(reranker.business_retries, 0)
        self.assertEqual(reranker.timeout_seconds, 0)
        self.assertEqual(image_gen.transport_retries, 1)
        self.assertEqual(image_gen.business_retries, 0)
        self.assertEqual(image_gen.timeout_seconds, 0)

    def test_runtime_model_configs_reject_removed_or_partial_fields(self) -> None:
        from src.core.manga_insight.config_models import (
            ChatLLMConfig,
            EmbeddingConfig,
            ImageGenConfig,
            RerankerConfig,
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
                RerankerConfig,
                {"top_k": 5, "max_retries": 8},
                {"top_k", "max_retries"},
            ),
            (
                ImageGenConfig,
                {"max_retries": 5},
                {"max_retries"},
            ),
        )

        for config_type, raw, _removed in cases:
            with self.subTest(config=config_type.__name__), self.assertRaises(
                ValueError
            ):
                config_type.from_dict(raw)

    def test_frozen_provider_reader_accepts_current_keyless_provider(self) -> None:
        from src.backend_v2.insight.provider_runtime import (
            frozen_embedding_config,
        )

        config = frozen_embedding_config(
            {
                "embedding": {
                    "provider": "ollama",
                    "model_name": "bge-m3",
                    "custom_base_url": "",
                    "rpm_limit": 0,
                    "transport_retries": 1,
                    "business_retries": 0,
                    "timeout_seconds": 0,
                }
            }
        )

        self.assertEqual(config.api_key, "")
        self.assertEqual(config.model, "bge-m3")

    def test_frozen_provider_reader_rejects_retired_runtime_fields(self) -> None:
        from src.backend_v2.insight.provider_runtime import frozen_chat_config
        from src.backend_v2.insight.repository import InsightConflict

        with self.assertRaisesRegex(InsightConflict, "unknown timeout_seconds"):
            frozen_chat_config(
                {
                    "chat": {
                        "provider": "ollama",
                        "model_name": "qwen2.5",
                        "custom_base_url": "",
                        "timeout_seconds": 120,
                        "openai_options": {
                            "request": {
                                "force_json_output": False,
                                "temperature": None,
                                "extra_body": {},
                            },
                            "execution": {
                                "use_stream": True,
                                "rpm_limit": 0,
                                "transport_retries": 1,
                                "business_retries": 0,
                            },
                        },
                    }
                }
            )
