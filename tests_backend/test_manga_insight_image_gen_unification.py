import sys
import types
import unittest
from unittest import mock


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


class SharedProviderRegistryImageGenTests(unittest.TestCase):
    def test_shared_registry_exposes_image_gen_capability_and_urls(self) -> None:
        from src.shared.ai_providers import (
            CHAT_CAPABILITY,
            IMAGE_GEN_CAPABILITY,
            provider_supports_capability,
            resolve_provider_base_url_for_capability,
            resolve_provider_endpoint_for_capability,
        )

        self.assertTrue(provider_supports_capability("qwen", IMAGE_GEN_CAPABILITY))
        self.assertFalse(provider_supports_capability("gemini", IMAGE_GEN_CAPABILITY))
        self.assertEqual(
            resolve_provider_base_url_for_capability("qwen", CHAT_CAPABILITY),
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.assertEqual(
            resolve_provider_base_url_for_capability("qwen", IMAGE_GEN_CAPABILITY),
            "https://dashscope.aliyuncs.com/api/v1",
        )
        self.assertEqual(
            resolve_provider_endpoint_for_capability("volcano", IMAGE_GEN_CAPABILITY),
            "/v1/images/generations",
        )


class MangaInsightImageGenClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_image_gen_client_dispatches_qwen_requests(self) -> None:
        from src.core.manga_insight.clients.image_gen_client import ImageGenClient
        from src.core.manga_insight.config_models import ImageGenConfig

        client = ImageGenClient(
            ImageGenConfig(
                provider="qwen",
                api_key="test-key",
                model="wanx-v1",
            )
        )
        try:
            with mock.patch.object(client, "_call_qwen_api", return_value=b"img-bytes") as qwen_mock:
                result = await client.generate("draw something", reference_images=[{"path": "ref.png", "type": "style"}])
        finally:
            await client.close()

        self.assertEqual(result, b"img-bytes")
        qwen_mock.assert_awaited_once()

    async def test_openai_image_gen_client_closes_sdk_client(self) -> None:
        from src.core.manga_insight.clients.image_gen_client import ImageGenClient
        from src.core.manga_insight.config_models import ImageGenConfig

        class FakeCompletions:
            def create(self, **kwargs):
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="data:image/png;base64,aGVsbG8="))]
                )

        class FakeClient:
            def __init__(self):
                self.chat = types.SimpleNamespace(completions=FakeCompletions())
                self.closed = False

            def close(self):
                self.closed = True

        fake_client = FakeClient()
        client = ImageGenClient(
            ImageGenConfig(
                provider="openai",
                api_key="test-key",
                model="dall-e-3",
            )
        )
        try:
            with mock.patch("src.core.manga_insight.clients.image_gen_client.create_openai_client", return_value=fake_client):
                result = await client.generate("draw something")
        finally:
            await client.close()

        self.assertEqual(result, b"hello")
        self.assertTrue(fake_client.closed)

    async def test_openai_image_gen_client_uses_to_thread_for_blocking_sdk_calls(self) -> None:
        from src.core.manga_insight.clients.image_gen_client import ImageGenClient
        from src.core.manga_insight.config_models import ImageGenConfig

        class FakeCompletions:
            def create(self, **kwargs):
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="data:image/png;base64,aGVsbG8="))]
                )

        class FakeClient:
            def __init__(self):
                self.chat = types.SimpleNamespace(completions=FakeCompletions())

            def close(self):
                return None

        calls = []

        async def fake_to_thread(func, *args, **kwargs):
            calls.append(getattr(func, "__name__", repr(func)))
            return func(*args, **kwargs)

        client = ImageGenClient(
            ImageGenConfig(
                provider="openai",
                api_key="test-key",
                model="dall-e-3",
            )
        )
        try:
            with mock.patch("src.core.manga_insight.clients.image_gen_client.create_openai_client", return_value=FakeClient()), \
                 mock.patch("src.core.manga_insight.clients.image_gen_client.asyncio.to_thread", side_effect=fake_to_thread):
                result = await client.generate("draw something")
        finally:
            await client.close()

        self.assertEqual(result, b"hello")
        self.assertIn("create", calls)
        self.assertIn("close", calls)


class ImageGeneratorDelegationTests(unittest.IsolatedAsyncioTestCase):
    async def test_image_generator_delegates_page_generation_to_image_gen_client(self) -> None:
        from src.core.manga_insight.continuation.image_generator import ImageGenerator
        from src.core.manga_insight.continuation.models import ContinuationCharacters, PageContent

        generator = ImageGenerator("test-book")
        try:
            with mock.patch.object(generator, "_build_full_prompt", return_value="page prompt"), \
                 mock.patch.object(generator._client, "generate", return_value=b"generated-image") as generate_mock, \
                 mock.patch.object(generator, "_save_image", return_value="saved-image.png"):
                result = await generator.generate_page_image(
                    page_content=PageContent(
                        page_number=1,
                        characters=[],
                        description="scene",
                        dialogues=[],
                        image_prompt="prompt",
                    ),
                    characters=ContinuationCharacters(book_id="test-book", characters=[]),
                    style_reference_images=["ref.png"],
                    style_ref_count=1,
                )
        finally:
            await generator.close()

        self.assertEqual(result, "saved-image.png")
        generate_mock.assert_awaited_once_with(
            "page prompt",
            reference_images=[{"path": "ref.png", "type": "style"}],
        )


class WebImportProviderResolutionTests(unittest.TestCase):
    def test_web_import_agent_accepts_canonical_custom_provider(self) -> None:
        from src.core.web_import.agent import MangaScraperAgent

        agent = MangaScraperAgent(
            {
                "agent": {
                    "provider": "custom",
                    "apiKey": "test-key",
                    "customBaseUrl": "https://example.com/v1",
                    "modelName": "gpt-test",
                }
            }
        )

        self.assertEqual(agent._get_base_url(), "https://example.com/v1")

    def test_web_import_agent_rejects_provider_without_web_import_capability(self) -> None:
        from src.core.web_import.agent import MangaScraperAgent

        with self.assertRaisesRegex(ValueError, "不支持的 AI Agent 服务商"):
            MangaScraperAgent(
                {
                    "agent": {
                        "provider": "qwen",
                        "apiKey": "test-key",
                        "modelName": "qwen-plus",
                    }
                }
            )


class SharedTransportModelListingTests(unittest.TestCase):
    def test_gemini_model_listing_uses_shared_http_config(self) -> None:
        from src.shared.ai_transport import OpenAICompatibleChatTransport, ProviderModelListRequest

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "models": [
                        {
                            "name": "models/gemini-2.0-flash",
                            "displayName": "Gemini 2.0 Flash",
                            "supportedGenerationMethods": ["generateContent"],
                        }
                    ]
                }

        class FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def get(self, url):
                return FakeResponse()

        transport = OpenAICompatibleChatTransport()
        request = ProviderModelListRequest(
            provider="gemini",
            api_key="test-key",
            timeout=12.0,
        )

        with mock.patch("src.shared.ai_transport.build_httpx_kwargs", return_value={"timeout": 12.0}) as kwargs_mock, \
             mock.patch("src.shared.ai_transport.httpx.Client", return_value=FakeClient()):
            models = transport.list_models(request)

        kwargs_mock.assert_called_once_with("https://generativelanguage.googleapis.com/v1beta/models?key=test-key", 12.0)
        self.assertEqual(models, [{"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash"}])
