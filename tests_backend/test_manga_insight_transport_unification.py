import unittest
from unittest import mock

from src.shared.openai_options import (
    OpenAICompatibleExecutionOptions,
    OpenAICompatibleOptions,
    OpenAICompatibleRequestOptions,
)


class MangaInsightSharedTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_client_reads_nested_openai_options_from_config(self) -> None:
        from src.core.manga_insight.config_models import ChatLLMConfig
        from src.core.manga_insight.embedding_client import ChatClient

        config = ChatLLMConfig.from_dict(
            {
                "provider": "custom",
                "api_key": "test-key",
                "model": "chat-model",
                "base_url": "https://example.com/v1",
                "openai_options": {
                    "request": {"temperature": 0.4},
                    "execution": {"use_stream": False, "rpm_limit": 9, "business_retries": 2},
                },
            }
        )

        with mock.patch(
            "src.core.manga_insight.embedding_client.AsyncOpenAICompatibleTransport.complete",
            new=mock.AsyncMock(return_value="统一回答"),
        ) as complete_mock:
            client = ChatClient(config)
            content = await client.generate("用户问题", system="系统提示")

        self.assertEqual(content, "统一回答")
        request = complete_mock.call_args.args[0]
        self.assertEqual(request.openai_options.request.temperature, 0.4)
        self.assertFalse(request.openai_options.execution.use_stream)
        self.assertEqual(request.openai_options.execution.rpm_limit, 9)
        self.assertEqual(request.openai_options.execution.business_retries, 2)
        self.assertEqual(request.runtime_options.stream_output_label, "漫画分析对话")

    async def test_chat_client_preserves_request_extra_body_from_config(self) -> None:
        from src.core.manga_insight.config_models import ChatLLMConfig
        from src.core.manga_insight.embedding_client import ChatClient

        config = ChatLLMConfig.from_dict(
            {
                "provider": "custom",
                "api_key": "test-key",
                "model": "chat-model",
                "base_url": "https://example.com/v1",
                "openai_options": {
                    "request": {
                        "extra_body": {"thinking": {"type": "disabled"}},
                    },
                    "execution": {"use_stream": False},
                },
            }
        )

        with mock.patch(
            "src.core.manga_insight.embedding_client.AsyncOpenAICompatibleTransport.complete",
            new=mock.AsyncMock(return_value="统一回答"),
        ) as complete_mock:
            client = ChatClient(config)
            await client.generate("用户问题", system="系统提示")

        request = complete_mock.call_args.args[0]
        self.assertEqual(
            request.openai_options.request.extra_body,
            {"thinking": {"type": "disabled"}},
        )

    async def test_chat_client_delegates_to_shared_async_transport(self) -> None:
        from src.core.manga_insight.config_models import ChatLLMConfig
        from src.core.manga_insight.embedding_client import ChatClient

        config = ChatLLMConfig(
            provider="custom",
            api_key="test-key",
            model="chat-model",
            base_url="https://example.com/v1",
            openai_options=OpenAICompatibleOptions(
                execution=OpenAICompatibleExecutionOptions(use_stream=False),
            ),
        )

        with mock.patch(
            "src.core.manga_insight.embedding_client.AsyncOpenAICompatibleTransport.complete",
            new=mock.AsyncMock(return_value="统一回答"),
        ) as complete_mock:
            client = ChatClient(config)
            content = await client.generate("用户问题", system="系统提示", temperature=0.4)

        self.assertEqual(content, "统一回答")
        request = complete_mock.call_args.args[0]
        self.assertEqual(request.provider, "custom")
        self.assertEqual(request.model, "chat-model")
        self.assertEqual(request.base_url, "https://example.com/v1")
        self.assertEqual(request.temperature, 0.4)
        self.assertFalse(request.use_stream)
        self.assertEqual(
            request.messages,
            [
                {"role": "system", "content": "系统提示"},
                {"role": "user", "content": "用户问题"},
            ],
        )

    async def test_embedding_client_delegates_to_shared_async_transport(self) -> None:
        from src.core.manga_insight.config_models import EmbeddingConfig
        from src.core.manga_insight.embedding_client import EmbeddingClient

        config = EmbeddingConfig(
            provider="custom",
            api_key="test-key",
            model="embedding-model",
            base_url="https://example.com/v1",
            rpm_limit=0,
        )

        with mock.patch(
            "src.core.manga_insight.embedding_client.AsyncOpenAICompatibleTransport.embed",
            new=mock.AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]]),
        ) as embed_mock:
            client = EmbeddingClient(config)
            embeddings = await client.embed_batch(["第一页", "第二页"])

        self.assertEqual(embeddings, [[0.1, 0.2], [0.3, 0.4]])
        request = embed_mock.call_args.args[0]
        self.assertEqual(request.provider, "custom")
        self.assertEqual(request.model, "embedding-model")
        self.assertEqual(request.base_url, "https://example.com/v1")
        self.assertEqual(request.inputs, ["第一页", "第二页"])

    async def test_reranker_client_delegates_transport_and_preserves_result_mapping(self) -> None:
        from src.core.manga_insight.config_models import RerankerConfig
        from src.core.manga_insight.reranker_client import RerankerClient

        config = RerankerConfig(
            provider="custom",
            api_key="test-key",
            model="rerank-model",
            base_url="https://example.com/v1",
            top_k=2,
        )
        documents = [
            {"document": "第一页摘要", "page_num": 1},
            {"document": "第二页摘要", "page_num": 2},
        ]

        with mock.patch(
            "src.core.manga_insight.reranker_client.AsyncOpenAICompatibleTransport.rerank",
            new=mock.AsyncMock(
                return_value={
                    "results": [
                        {"index": 1, "relevance_score": 0.93},
                        {"index": 0, "relevance_score": 0.75},
                    ]
                }
            ),
        ) as rerank_mock:
            client = RerankerClient(config)
            reranked = await client.rerank("主角是谁", documents, top_k=2)

        self.assertEqual([item["page_num"] for item in reranked], [2, 1])
        self.assertEqual(reranked[0]["rerank_score"], 0.93)
        request = rerank_mock.call_args.args[0]
        self.assertEqual(request.provider, "custom")
        self.assertEqual(request.model, "rerank-model")
        self.assertEqual(request.query, "主角是谁")
        self.assertEqual(request.documents, ["第一页摘要", "第二页摘要"])
        self.assertEqual(request.top_n, 2)

    async def test_vlm_client_uses_shared_async_transport_for_multimodal_chat(self) -> None:
        from src.core.manga_insight.config_models import PromptsConfig, VLMConfig
        from src.core.manga_insight.vlm_client import VLMClient

        config = VLMConfig(
            provider="custom",
            api_key="test-key",
            model="vlm-model",
            base_url="https://example.com/v1",
            openai_options=OpenAICompatibleOptions(
                request=OpenAICompatibleRequestOptions(
                    force_json_output=True,
                    temperature=0.2,
                ),
                execution=OpenAICompatibleExecutionOptions(use_stream=False),
            ),
            image_max_size=0,
        )

        with mock.patch(
            "src.core.manga_insight.vlm_client.AsyncOpenAICompatibleTransport.complete",
            new=mock.AsyncMock(return_value='{"pages": []}'),
        ) as complete_mock:
            client = VLMClient(config, PromptsConfig())
            content = await client._call_vlm([b"fake-image"], "分析这页漫画")

        self.assertEqual(content, '{"pages": []}')
        request = complete_mock.call_args.args[0]
        self.assertEqual(request.provider, "custom")
        self.assertEqual(request.model, "vlm-model")
        self.assertEqual(request.base_url, "https://example.com/v1")
        self.assertEqual(request.temperature, 0.2)
        self.assertFalse(request.use_stream)
        self.assertEqual(request.response_format, {"type": "json_object"})
        self.assertEqual(request.messages[0]["role"], "user")
        self.assertEqual(request.messages[0]["content"][-1], {"type": "text", "text": "分析这页漫画"})

    async def test_vlm_client_preserves_request_extra_body(self) -> None:
        from src.core.manga_insight.config_models import PromptsConfig, VLMConfig
        from src.core.manga_insight.vlm_client import VLMClient

        config = VLMConfig(
            provider="custom",
            api_key="test-key",
            model="vlm-model",
            base_url="https://example.com/v1",
            openai_options=OpenAICompatibleOptions(
                request=OpenAICompatibleRequestOptions(
                    extra_body={"thinking": {"type": "disabled"}},
                ),
                execution=OpenAICompatibleExecutionOptions(use_stream=False),
            ),
            image_max_size=0,
        )

        with mock.patch(
            "src.core.manga_insight.vlm_client.AsyncOpenAICompatibleTransport.complete",
            new=mock.AsyncMock(return_value='{"pages": []}'),
        ) as complete_mock:
            client = VLMClient(config, PromptsConfig())
            await client._call_vlm([b"fake-image"], "分析这页漫画")

        request = complete_mock.call_args.args[0]
        self.assertEqual(
            request.openai_options.request.extra_body,
            {"thinking": {"type": "disabled"}},
        )

    async def test_vlm_client_reads_nested_openai_options_from_config(self) -> None:
        from src.core.manga_insight.config_models import PromptsConfig, VLMConfig
        from src.core.manga_insight.vlm_client import VLMClient

        config = VLMConfig.from_dict(
            {
                "provider": "custom",
                "api_key": "test-key",
                "model": "vlm-model",
                "base_url": "https://example.com/v1",
                "image_max_size": 0,
                "openai_options": {
                    "request": {"force_json_output": True, "temperature": 0.2},
                    "execution": {"use_stream": False, "rpm_limit": 8, "business_retries": 3},
                },
            }
        )

        with mock.patch(
            "src.core.manga_insight.vlm_client.AsyncOpenAICompatibleTransport.complete",
            new=mock.AsyncMock(return_value='{"pages": []}'),
        ) as complete_mock:
            client = VLMClient(config, PromptsConfig())
            content = await client._call_vlm([b"fake-image"], "分析这页漫画")

        self.assertEqual(content, '{"pages": []}')
        request = complete_mock.call_args.args[0]
        self.assertTrue(request.openai_options.request.force_json_output)
        self.assertEqual(request.openai_options.request.temperature, 0.2)
        self.assertFalse(request.openai_options.execution.use_stream)
        self.assertEqual(request.openai_options.execution.rpm_limit, 8)
        self.assertEqual(request.openai_options.execution.business_retries, 3)
        self.assertEqual(request.runtime_options.stream_output_label, "漫画分析")

    async def test_vlm_client_retries_non_retryable_transport_failures_at_outer_layer(self) -> None:
        from src.core.manga_insight.config_models import PromptsConfig, VLMConfig
        from src.core.manga_insight.vlm_client import VLMClient

        config = VLMConfig(
            provider="custom",
            api_key="test-key",
            model="vlm-model",
            base_url="https://example.com/v1",
            openai_options=OpenAICompatibleOptions(
                execution=OpenAICompatibleExecutionOptions(use_stream=False, business_retries=1),
            ),
        )

        complete_mock = mock.AsyncMock(side_effect=[ValueError("empty choices"), '{"pages": []}'])
        with mock.patch(
            "src.core.manga_insight.vlm_client.AsyncOpenAICompatibleTransport.complete",
            new=complete_mock,
        ):
            client = VLMClient(config, PromptsConfig())
            content = await client._call_vlm([b"fake-image"], "分析这页漫画")

        self.assertEqual(content, '{"pages": []}')
        self.assertEqual(complete_mock.await_count, 2)

    async def test_vlm_test_connection_does_not_send_default_max_tokens(self) -> None:
        from src.core.manga_insight.config_models import VLMConfig
        from src.core.manga_insight.vlm_client import VLMClient

        config = VLMConfig(
            provider="custom",
            api_key="test-key",
            model="vlm-model",
            base_url="https://example.com/v1",
            openai_options=OpenAICompatibleOptions(
                execution=OpenAICompatibleExecutionOptions(use_stream=False),
            ),
        )

        with mock.patch(
            "src.core.manga_insight.vlm_client.AsyncOpenAICompatibleTransport.complete",
            new=mock.AsyncMock(return_value="连接成功"),
        ) as complete_mock:
            client = VLMClient(config)
            success = await client.test_connection()

        self.assertTrue(success)
        request = complete_mock.call_args.args[0]
        self.assertNotIn("max_tokens", request.runtime_options.request_overrides)

    def test_vlm_parse_batch_analysis_accepts_page_analyses_fallback_key(self) -> None:
        from src.core.manga_insight.config_models import PromptsConfig, VLMConfig
        from src.core.manga_insight.vlm_client import VLMClient

        config = VLMConfig(
            provider="custom",
            api_key="test-key",
            model="vlm-model",
            base_url="https://example.com/v1",
            openai_options=OpenAICompatibleOptions(
                execution=OpenAICompatibleExecutionOptions(use_stream=False),
            ),
        )
        client = VLMClient(config, PromptsConfig())

        result = client._parse_batch_analysis(
            '{"page_analyses":[{"page_number":1,"page_summary":"第一页"},{"page_number":2,"page_summary":"第二页"}],"batch_summary":"总览"}',
            1,
            2,
        )

        self.assertEqual(len(result["pages"]), 2)
        self.assertEqual(result["pages"][0]["page_number"], 1)
        self.assertEqual(result["batch_summary"], "总览")

    def test_vlm_parse_batch_analysis_normalizes_page_num_to_page_number(self) -> None:
        from src.core.manga_insight.config_models import PromptsConfig, VLMConfig
        from src.core.manga_insight.vlm_client import VLMClient

        config = VLMConfig(
            provider="custom",
            api_key="test-key",
            model="vlm-model",
            base_url="https://example.com/v1",
            openai_options=OpenAICompatibleOptions(
                execution=OpenAICompatibleExecutionOptions(use_stream=False),
            ),
        )
        client = VLMClient(config, PromptsConfig())

        result = client._parse_batch_analysis(
            '{"pages":[{"page_num":1,"page_summary":"第一页"},{"page_num":2,"page_summary":"第二页"}],"batch_summary":"总览"}',
            1,
            2,
        )

        self.assertEqual(result["pages"][0]["page_number"], 1)
        self.assertEqual(result["pages"][1]["page_number"], 2)
