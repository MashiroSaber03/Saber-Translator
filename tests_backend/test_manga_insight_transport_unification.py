import asyncio
from io import BytesIO
import unittest
from unittest import mock

from PIL import Image

from src.shared.openai_options import (
    OpenAICompatibleExecutionOptions,
    OpenAICompatibleOptions,
    OpenAICompatibleRequestOptions,
)
from src.shared.openai_execution import OpenAICompatibleEmptyContentError


def _png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (2, 2), "white").save(output, format="PNG")
    return output.getvalue()


class MangaInsightSharedTransportTests(unittest.IsolatedAsyncioTestCase):
    def test_incomplete_http_response_is_transport_retryable(self) -> None:
        import httpx

        from src.shared.ai_transport import RETRYABLE_EXCEPTIONS

        self.assertTrue(
            issubclass(httpx.RemoteProtocolError, RETRYABLE_EXCEPTIONS)
        )

    def test_shared_json_parser_ignores_reasoning_tags_before_extracting_json(self) -> None:
        from src.shared.openai_execution import parse_json_block_from_text

        parsed = parse_json_block_from_text(
            '<think>{"draft": 1}</think>\n```json\n{"answer": "ok"}\n```'
        )

        self.assertEqual(parsed, {"answer": "ok"})

    def test_shared_json_parser_rejects_unstructured_prefix_or_suffix(self) -> None:
        from src.shared.openai_execution import (
            OpenAICompatibleBusinessRetryableError,
            parse_json_block_from_text,
        )

        for response in (
            '说明：{"answer": "ok"}',
            '{"answer": "ok"} trailing',
            '```json\n{"answer": "ok"}',
        ):
            with self.subTest(response=response), self.assertRaises(
                OpenAICompatibleBusinessRetryableError
            ):
                parse_json_block_from_text(response)

    async def test_chat_client_reads_nested_openai_options_from_config(self) -> None:
        from src.core.manga_insight.config_models import ChatLLMConfig
        from src.core.manga_insight.embedding_client import ChatClient

        config = ChatLLMConfig.from_dict(
            {
                "provider": "custom",
                "api_key": "test-key",
                "model": "chat-model",
                "base_url": "https://example.com/v1",
                "credential_version_id": None,
                "openai_options": {
                    "request": {
                        "force_json_output": False,
                        "temperature": 0.4,
                        "extra_body": {},
                    },
                    "execution": {
                        "use_stream": False,
                        "rpm_limit": 9,
                        "transport_retries": 1,
                        "business_retries": 2,
                    },
                },
            }
        )

        with mock.patch(
            "src.core.manga_insight.embedding_client.AsyncOpenAICompatibleTransport.complete",
            new=mock.AsyncMock(return_value='{"answer":"统一回答"}'),
        ) as complete_mock:
            client = ChatClient(config)
            content = await client.generate_json("用户问题", system="系统提示")

        self.assertEqual(content, {"answer": "统一回答"})
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
                "credential_version_id": None,
                "openai_options": {
                    "request": {
                        "force_json_output": False,
                        "temperature": None,
                        "extra_body": {"thinking": {"type": "disabled"}},
                    },
                    "execution": {
                        "use_stream": False,
                        "rpm_limit": 0,
                        "transport_retries": 1,
                        "business_retries": 0,
                    },
                },
            }
        )

        with mock.patch(
            "src.core.manga_insight.embedding_client.AsyncOpenAICompatibleTransport.complete",
            new=mock.AsyncMock(return_value='{"answer":"统一回答"}'),
        ) as complete_mock:
            client = ChatClient(config)
            await client.generate_json("用户问题", system="系统提示")

        request = complete_mock.call_args.args[0]
        self.assertEqual(
            request.openai_options.request.extra_body,
            {"thinking": {"type": "disabled"}},
        )

    async def test_chat_client_generate_json_retries_until_markdown_json_parses(self) -> None:
        from src.core.manga_insight.config_models import ChatLLMConfig
        from src.core.manga_insight.embedding_client import ChatClient

        config = ChatLLMConfig.from_dict(
            {
                "provider": "custom",
                "api_key": "test-key",
                "model": "chat-model",
                "base_url": "https://example.com/v1",
                "credential_version_id": None,
                "openai_options": {
                    "request": {
                        "force_json_output": False,
                        "temperature": 0.4,
                        "extra_body": {},
                    },
                    "execution": {
                        "use_stream": False,
                        "rpm_limit": 0,
                        "transport_retries": 1,
                        "business_retries": 1,
                    },
                },
            }
        )

        complete_mock = mock.AsyncMock(
            side_effect=[
                "这不是 JSON",
                '```json\n{"answer": "retry-ok"}\n```',
            ]
        )
        with mock.patch(
            "src.core.manga_insight.embedding_client.AsyncOpenAICompatibleTransport.complete",
            new=complete_mock,
        ):
            client = ChatClient(config)
            parsed = await client.generate_json("用户问题")

        self.assertEqual(parsed, {"answer": "retry-ok"})
        self.assertEqual(complete_mock.await_count, 2)

    async def test_chat_client_bounds_the_complete_logical_call(self) -> None:
        from src.core.manga_insight.config_models import ChatLLMConfig
        from src.core.manga_insight.embedding_client import ChatClient

        async def never_finishes(*_args, **_kwargs):
            await asyncio.Event().wait()

        config = ChatLLMConfig(
            provider="custom",
            api_key="test-key",
            model="chat-model",
            base_url="https://example.com/v1",
        )
        with mock.patch(
            "src.core.manga_insight.embedding_client.AsyncOpenAICompatibleTransport.complete",
            new=mock.AsyncMock(side_effect=never_finishes),
        ):
            client = ChatClient(config)
            client._total_timeout = 0.01
            with self.assertRaisesRegex(
                TimeoutError,
                "对话模型调用超过总时限（0.01 秒）",
            ):
                await client.generate_json("用户问题")

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

    async def test_embedding_client_uses_configured_retries_and_timeout(
        self,
    ) -> None:
        from src.core.manga_insight.config_models import EmbeddingConfig
        from src.core.manga_insight.embedding_client import EmbeddingClient

        config = EmbeddingConfig(
            provider="custom",
            api_key="test-key",
            model="embedding-model",
            base_url="https://example.com/v1",
            rpm_limit=0,
            transport_retries=10,
            business_retries=10,
            timeout_seconds=0,
        )

        with mock.patch(
            "src.core.manga_insight.embedding_client.AsyncOpenAICompatibleTransport.embed",
            new=mock.AsyncMock(return_value=[[0.1, 0.2]]),
        ) as embed_mock:
            client = EmbeddingClient(config)
            await client.embed_batch(["第一页"])

        self.assertEqual(client._transport.max_retries, 10)
        request = embed_mock.call_args.args[0]
        self.assertIsNone(request.timeout)

    async def test_embedding_client_preserves_the_upstream_batch_and_order(self) -> None:
        from src.core.manga_insight.config_models import EmbeddingConfig
        from src.core.manga_insight.embedding_client import EmbeddingClient

        config = EmbeddingConfig(
            provider="custom",
            api_key="test-key",
            model="embedding-model",
            base_url="https://example.com/v1",
            rpm_limit=0,
            transport_retries=0,
            business_retries=0,
            timeout_seconds=0,
        )
        requests = []

        async def embed_batch(request):
            requests.append(request)
            return [[float(text)] for text in request.inputs]

        with mock.patch(
            "src.core.manga_insight.embedding_client.AsyncOpenAICompatibleTransport.embed",
            new=mock.AsyncMock(side_effect=embed_batch),
        ):
            client = EmbeddingClient(config)
            embeddings = await client.embed_batch(
                [str(index) for index in range(35)]
            )

        self.assertEqual(
            [len(request.inputs) for request in requests],
            [35],
        )
        self.assertEqual(
            embeddings,
            [[float(index)] for index in range(35)],
        )

    async def test_embedding_client_retries_empty_business_result(self) -> None:
        from src.core.manga_insight.config_models import EmbeddingConfig
        from src.core.manga_insight.embedding_client import EmbeddingClient

        config = EmbeddingConfig(
            provider="custom",
            api_key="test-key",
            model="embedding-model",
            base_url="https://example.com/v1",
            rpm_limit=0,
            transport_retries=0,
            business_retries=1,
            timeout_seconds=0,
        )

        with mock.patch(
            "src.core.manga_insight.embedding_client.AsyncOpenAICompatibleTransport.embed",
            new=mock.AsyncMock(side_effect=[[], [[0.1, 0.2]]]),
        ) as embed_mock:
            client = EmbeddingClient(config)
            embeddings = await client.embed_batch(["第一页"])

        self.assertEqual(embeddings, [[0.1, 0.2]])
        self.assertEqual(embed_mock.await_count, 2)

    async def test_embedding_client_does_not_business_retry_generic_value_error(self) -> None:
        from src.core.manga_insight.config_models import EmbeddingConfig
        from src.core.manga_insight.embedding_client import EmbeddingClient

        config = EmbeddingConfig(
            provider="custom",
            api_key="test-key",
            model="embedding-model",
            base_url="https://example.com/v1",
            rpm_limit=0,
            transport_retries=0,
            business_retries=10,
            timeout_seconds=0,
        )

        with mock.patch(
            "src.core.manga_insight.embedding_client.AsyncOpenAICompatibleTransport.embed",
            new=mock.AsyncMock(side_effect=ValueError("API 错误 401: unauthorized")),
        ) as embed_mock:
            client = EmbeddingClient(config)
            with self.assertRaisesRegex(ValueError, "401"):
                await client.embed_batch(["第一页"])

        self.assertEqual(embed_mock.await_count, 1)
        self.assertEqual(client._transport.max_retries, 0)

    async def test_vlm_client_uses_shared_async_transport_for_multimodal_chat(self) -> None:
        from src.core.manga_insight.config_models import VLMConfig
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
            new=mock.AsyncMock(return_value='{"pages":[{"page_number":1}]}'),
        ) as complete_mock:
            client = VLMClient(config)
            content = await client.analyze_page(_png_bytes(), 1, "分析这页漫画")

        self.assertEqual(content, {"pages": [{"page_number": 1}]})
        request = complete_mock.call_args.args[0]
        self.assertEqual(request.provider, "custom")
        self.assertEqual(request.model, "vlm-model")
        self.assertEqual(request.base_url, "https://example.com/v1")
        self.assertEqual(request.temperature, 0.2)
        self.assertFalse(request.use_stream)
        self.assertEqual(request.response_format, {"type": "json_object"})
        self.assertEqual(request.runtime_options.timeout, 120.0)
        self.assertEqual(request.messages[0]["role"], "user")
        self.assertEqual(request.messages[0]["content"][-1], {"type": "text", "text": "分析这页漫画"})

    async def test_vlm_client_preserves_request_extra_body(self) -> None:
        from src.core.manga_insight.config_models import VLMConfig
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
            new=mock.AsyncMock(return_value='{"pages":[{"page_number":1}]}'),
        ) as complete_mock:
            client = VLMClient(config)
            await client.analyze_page(_png_bytes(), 1, "分析这页漫画")

        request = complete_mock.call_args.args[0]
        self.assertEqual(
            request.openai_options.request.extra_body,
            {"thinking": {"type": "disabled"}},
        )

    async def test_vlm_client_reads_nested_openai_options_from_config(self) -> None:
        from src.core.manga_insight.config_models import VLMConfig
        from src.core.manga_insight.vlm_client import VLMClient

        config = VLMConfig.from_dict(
            {
                "provider": "custom",
                "api_key": "test-key",
                "model": "vlm-model",
                "base_url": "https://example.com/v1",
                "credential_version_id": None,
                "image_max_size": 0,
                "openai_options": {
                    "request": {
                        "force_json_output": True,
                        "temperature": 0.2,
                        "extra_body": {},
                    },
                    "execution": {
                        "use_stream": False,
                        "rpm_limit": 8,
                        "transport_retries": 1,
                        "business_retries": 3,
                    },
                },
            }
        )

        with mock.patch(
            "src.core.manga_insight.vlm_client.AsyncOpenAICompatibleTransport.complete",
            new=mock.AsyncMock(return_value='{"pages":[{"page_number":1}]}'),
        ) as complete_mock:
            client = VLMClient(config)
            content = await client.analyze_page(_png_bytes(), 1, "分析这页漫画")

        self.assertEqual(content, {"pages": [{"page_number": 1}]})
        request = complete_mock.call_args.args[0]
        self.assertTrue(request.openai_options.request.force_json_output)
        self.assertEqual(request.openai_options.request.temperature, 0.2)
        self.assertFalse(request.openai_options.execution.use_stream)
        self.assertEqual(request.openai_options.execution.rpm_limit, 8)
        self.assertEqual(request.openai_options.execution.business_retries, 3)
        self.assertEqual(request.runtime_options.stream_output_label, "漫画分析")

    async def test_vlm_client_retries_typed_empty_content_at_business_layer(self) -> None:
        from src.core.manga_insight.config_models import VLMConfig
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

        complete_mock = mock.AsyncMock(
            side_effect=[
                OpenAICompatibleEmptyContentError("AI 未返回有效内容"),
                '{"pages":[{"page_number":1}]}',
            ]
        )
        with mock.patch(
            "src.core.manga_insight.vlm_client.AsyncOpenAICompatibleTransport.complete",
            new=complete_mock,
        ):
            client = VLMClient(config)
            content = await client.analyze_page(_png_bytes(), 1, "分析这页漫画")

        self.assertEqual(content, {"pages": [{"page_number": 1}]})
        self.assertEqual(complete_mock.await_count, 2)

    async def test_vlm_client_bounds_the_complete_retrying_call_by_wall_clock(self) -> None:
        from src.core.manga_insight.config_models import VLMConfig
        from src.core.manga_insight.vlm_client import VLMClient

        config = VLMConfig(
            provider="custom",
            api_key="test-key",
            model="vlm-model",
            base_url="https://example.com/v1",
            openai_options=OpenAICompatibleOptions(
                execution=OpenAICompatibleExecutionOptions(
                    use_stream=False,
                    transport_retries=10,
                    business_retries=10,
                ),
            ),
            image_max_size=0,
        )

        async def never_finishes(*_args, **_kwargs):
            await asyncio.sleep(60)
            return '{"pages": []}'

        with mock.patch(
            "src.core.manga_insight.vlm_client.AsyncOpenAICompatibleTransport.complete",
            new=mock.AsyncMock(side_effect=never_finishes),
        ):
            client = VLMClient(config)
            client._total_timeout = 0.01
            with self.assertRaisesRegex(
                TimeoutError,
                "视觉模型调用超过总时限（0.01 秒）",
            ):
                await client.analyze_page(_png_bytes(), 1, "分析这页漫画")

    def test_vlm_single_page_requires_the_requested_page_number(self) -> None:
        from src.core.manga_insight.config_models import VLMConfig
        from src.core.manga_insight.vlm_client import VLMClient

        client = VLMClient(
            VLMConfig(
                provider="custom",
                api_key="test-key",
                model="vlm-model",
                base_url="https://example.com/v1",
            )
        )

        result = client._parse_page_analysis(
            '{"pages":[{"page_number":14,"page_summary":"单页摘要"}]}',
            14,
        )

        self.assertEqual(result["pages"][0]["page_number"], 14)

        with self.assertRaisesRegex(ValueError, "page_number"):
            client._parse_page_analysis(
                '{"pages":[{"page_number":108}]}',
                14,
            )

    def test_vlm_page_analysis_rejects_retired_or_batch_shapes(self) -> None:
        from src.core.manga_insight.config_models import VLMConfig
        from src.core.manga_insight.vlm_client import VLMClient

        client = VLMClient(
            VLMConfig(
                provider="custom",
                api_key="test-key",
                model="vlm-model",
                base_url="https://example.com/v1",
            )
        )
        retired_payloads = (
            '{"page_analyses":[{"page_number":1}]}',
            '{"pages":[{"page_num":1}]}',
            '[{"page_number":1}]',
            '{"pages":[{"page_number":108},{"page_number":109}]}',
        )
        for payload in retired_payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                client._parse_page_analysis(payload, 1)
