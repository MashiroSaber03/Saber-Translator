import unittest
from unittest import mock


class AsyncTransportContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_chat_transport_builds_same_openai_body_shape_as_sync_transport(self) -> None:
        from src.shared.ai_transport import AsyncOpenAICompatibleTransport, UnifiedChatRequest

        class FakeResponse:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": "你好，异步世界"}}]}

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                self.request_calls = []

            async def request(self, method=None, url=None, headers=None, json=None):
                self.request_calls.append(
                    {"method": method, "url": url, "headers": headers, "json": json}
                )
                return FakeResponse()

            async def aclose(self):
                return None

        transport = AsyncOpenAICompatibleTransport()
        request = UnifiedChatRequest(
            provider="custom",
            api_key="test-key",
            model="gpt-test",
            messages=[{"role": "user", "content": "hello"}],
            base_url="https://example.com/v1",
            temperature=0.35,
            response_format={"type": "json_object"},
            request_overrides={"max_tokens": 222},
        )

        fake_client = FakeAsyncClient()
        with mock.patch("src.shared.ai_transport.httpx.AsyncClient", return_value=fake_client):
            content = await transport.complete(request)

        self.assertEqual(content, "你好，异步世界")
        self.assertEqual(len(fake_client.request_calls), 1)
        call = fake_client.request_calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["url"], "https://example.com/v1/chat/completions")
        self.assertEqual(call["json"]["temperature"], 0.35)
        self.assertEqual(call["json"]["response_format"], {"type": "json_object"})
        self.assertEqual(call["json"]["max_tokens"], 222)

    async def test_async_transport_supports_embedding_requests(self) -> None:
        from src.shared.ai_transport import AsyncOpenAICompatibleTransport, UnifiedEmbeddingRequest

        class FakeResponse:
            status_code = 200

            def json(self):
                return {
                    "data": [
                        {"embedding": [0.1, 0.2]},
                        {"embedding": [0.3, 0.4]},
                    ]
                }

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                self.request_calls = []

            async def request(self, method=None, url=None, headers=None, json=None):
                self.request_calls.append(
                    {"method": method, "url": url, "headers": headers, "json": json}
                )
                return FakeResponse()

            async def aclose(self):
                return None

        transport = AsyncOpenAICompatibleTransport()
        request = UnifiedEmbeddingRequest(
            provider="custom",
            api_key="test-key",
            model="text-embedding-test",
            inputs=["第一页", "第二页"],
            base_url="https://example.com/v1",
        )

        fake_client = FakeAsyncClient()
        with mock.patch("src.shared.ai_transport.httpx.AsyncClient", return_value=fake_client):
            embeddings = await transport.embed(request)

        self.assertEqual(embeddings, [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(fake_client.request_calls[0]["url"], "https://example.com/v1/embeddings")
        self.assertEqual(
            fake_client.request_calls[0]["json"],
            {"model": "text-embedding-test", "input": ["第一页", "第二页"]},
        )

    async def test_async_transport_supports_rerank_requests(self) -> None:
        from src.shared.ai_transport import AsyncOpenAICompatibleTransport, UnifiedRerankRequest

        class FakeResponse:
            status_code = 200

            def json(self):
                return {
                    "results": [
                        {"index": 1, "relevance_score": 0.93},
                        {"index": 0, "relevance_score": 0.76},
                    ]
                }

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                self.request_calls = []

            async def request(self, method=None, url=None, headers=None, json=None):
                self.request_calls.append(
                    {"method": method, "url": url, "headers": headers, "json": json}
                )
                return FakeResponse()

            async def aclose(self):
                return None

        transport = AsyncOpenAICompatibleTransport()
        request = UnifiedRerankRequest(
            provider="custom",
            api_key="test-key",
            model="rerank-test",
            query="主角是谁",
            documents=["文档A", "文档B"],
            top_n=2,
            base_url="https://example.com/v1",
            endpoint="/rerank",
        )

        fake_client = FakeAsyncClient()
        with mock.patch("src.shared.ai_transport.httpx.AsyncClient", return_value=fake_client):
            result = await transport.rerank(request)

        self.assertEqual(result["results"][0]["index"], 1)
        self.assertEqual(fake_client.request_calls[0]["url"], "https://example.com/v1/rerank")
        self.assertEqual(
            fake_client.request_calls[0]["json"],
            {
                "model": "rerank-test",
                "query": "主角是谁",
                "documents": ["文档A", "文档B"],
                "top_n": 2,
            },
        )
