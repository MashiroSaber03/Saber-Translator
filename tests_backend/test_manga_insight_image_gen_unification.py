import sys
import types
import unittest
from unittest import mock

import httpx


if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda *_args, **_kwargs: {}
    yaml_stub.safe_dump = lambda *_args, **_kwargs: ""
    sys.modules["yaml"] = yaml_stub

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class _OpenAI:
        def __init__(self, *args, **kwargs):
            pass

    class _AsyncOpenAI:
        def __init__(self, *args, **kwargs):
            pass

    openai_stub.OpenAI = _OpenAI
    openai_stub.AsyncOpenAI = _AsyncOpenAI
    sys.modules["openai"] = openai_stub


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class SharedProviderRegistryImageGenTests(unittest.TestCase):
    def test_image_gen_provider_enum_exposes_newapi(self) -> None:
        from src.core.manga_insight.config_models import APIProvider

        self.assertEqual(APIProvider.GPT2API.value, "gpt2api")
        self.assertEqual(APIProvider.NEWAPI.value, "newapi")

    def test_shared_registry_exposes_gpt2api_and_newapi_as_image_gen_providers(self) -> None:
        from src.shared.ai_providers import (
            IMAGE_GEN_CAPABILITY,
            get_provider_default_model,
            provider_supports_capability,
            resolve_provider_base_url_for_capability,
        )

        self.assertTrue(provider_supports_capability("gpt2api", IMAGE_GEN_CAPABILITY))
        self.assertTrue(provider_supports_capability("newapi", IMAGE_GEN_CAPABILITY))
        self.assertFalse(provider_supports_capability("openai", IMAGE_GEN_CAPABILITY))
        self.assertFalse(provider_supports_capability("qwen", IMAGE_GEN_CAPABILITY))
        self.assertEqual(get_provider_default_model("gpt2api", "image_gen"), "gpt-image-2")
        self.assertEqual(get_provider_default_model("newapi", "image_gen"), "")
        self.assertIsNone(resolve_provider_base_url_for_capability("gpt2api", IMAGE_GEN_CAPABILITY))
        self.assertIsNone(resolve_provider_base_url_for_capability("newapi", IMAGE_GEN_CAPABILITY))


class MangaInsightImageGenClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_image_gen_client_uses_configured_unlimited_timeout(self) -> None:
        from src.core.manga_insight.clients.image_gen_client import ImageGenClient
        from src.core.manga_insight.config_models import ImageGenConfig

        client = ImageGenClient(
            ImageGenConfig(
                provider="gpt2api",
                api_key="test-key",
                model="gpt-image-2",
                base_url="https://gateway.example.com",
                transport_retries=10,
                business_retries=10,
                timeout_seconds=0,
            )
        )
        try:
            self.assertIsNone(client._timeout)
            self.assertEqual(client._transport_retries, 10)
            self.assertEqual(client._business_retries, 10)
        finally:
            await client.close()

    async def test_image_gen_client_uses_generations_route_without_references(self) -> None:
        from src.core.manga_insight.clients.image_gen_client import ImageGenClient
        from src.core.manga_insight.config_models import ImageGenConfig

        client = ImageGenClient(
            ImageGenConfig(
                provider="gpt2api",
                api_key="test-key",
                model="gpt-image-2",
                base_url="https://gateway.example.com",
            )
        )
        try:
            post_mock = mock.AsyncMock(
                return_value=FakeResponse(
                    200,
                    {"data": [{"url": "data:image/png;base64,aGVsbG8="}]},
                )
            )
            client.client.post = post_mock

            result = await client.generate("draw something")
        finally:
            await client.close()

        self.assertEqual(result, b"hello")
        post_mock.assert_awaited_once()
        self.assertEqual(post_mock.await_args.args[0], "https://gateway.example.com/v1/images/generations")
        self.assertEqual(post_mock.await_args.kwargs["json"]["model"], "gpt-image-2")
        self.assertEqual(post_mock.await_args.kwargs["json"]["prompt"], "draw something")
        self.assertNotIn("images", post_mock.await_args.kwargs["json"])

    async def test_image_gen_client_uses_edits_route_with_references(self) -> None:
        from src.core.manga_insight.clients.image_gen_client import ImageGenClient
        from src.core.manga_insight.config_models import ImageGenConfig

        client = ImageGenClient(
            ImageGenConfig(
                provider="gpt2api",
                api_key="test-key",
                model="gpt-image-2",
                base_url="https://gateway.example.com/v1",
            )
        )
        try:
            post_mock = mock.AsyncMock(
                return_value=FakeResponse(
                    200,
                    {"data": [{"url": "data:image/png;base64,aGVsbG8="}]},
                )
            )
            client.client.post = post_mock

            with mock.patch.object(
                client,
                "_prepare_reference_images",
                return_value=[
                    {"filename": "reference.png", "bytes": b"reference", "mime": "image/png"},
                ],
            ):
                result = await client.generate("draw something", reference_images=[{"path": "ref.png"}])
        finally:
            await client.close()

        self.assertEqual(result, b"hello")
        post_mock.assert_awaited_once()
        self.assertEqual(post_mock.await_args.args[0], "https://gateway.example.com/v1/images/edits")
        self.assertEqual(
            post_mock.await_args.kwargs["data"]["prompt"],
            "draw something",
        )
        self.assertEqual(
            post_mock.await_args.kwargs["files"],
            [("image", ("reference.png", b"reference", "image/png"))],
        )

    async def test_image_gen_client_retries_empty_business_result_only(self) -> None:
        from src.core.manga_insight.clients.image_gen_client import ImageGenClient
        from src.core.manga_insight.config_models import ImageGenConfig

        client = ImageGenClient(
            ImageGenConfig(
                provider="gpt2api",
                api_key="test-key",
                model="gpt-image-2",
                base_url="https://gateway.example.com",
                transport_retries=0,
                business_retries=1,
                timeout_seconds=0,
            )
        )
        try:
            post_mock = mock.AsyncMock(
                side_effect=[
                    FakeResponse(200, {"data": []}),
                    FakeResponse(200, {"data": [{"url": "data:image/png;base64,aGVsbG8="}]}),
                ]
            )
            client.client.post = post_mock

            result = await client.generate("draw something")
        finally:
            await client.close()

        self.assertEqual(result, b"hello")
        self.assertEqual(post_mock.await_count, 2)

    async def test_image_gen_client_retries_transport_failures_without_spending_business_retry(self) -> None:
        from src.core.manga_insight.clients.image_gen_client import ImageGenClient
        from src.core.manga_insight.config_models import ImageGenConfig

        client = ImageGenClient(
            ImageGenConfig(
                provider="gpt2api",
                api_key="test-key",
                model="gpt-image-2",
                base_url="https://gateway.example.com",
                transport_retries=1,
                business_retries=0,
                timeout_seconds=0,
            )
        )
        try:
            post_mock = mock.AsyncMock(
                side_effect=[
                    httpx.ReadTimeout("timeout"),
                    FakeResponse(200, {"data": [{"url": "data:image/png;base64,aGVsbG8="}]}),
                ]
            )
            client.client.post = post_mock

            result = await client.generate("draw something")
        finally:
            await client.close()

        self.assertEqual(result, b"hello")
        self.assertEqual(post_mock.await_count, 2)

    async def test_image_gen_client_does_not_business_retry_non_retryable_api_errors(self) -> None:
        from src.core.manga_insight.clients.image_gen_client import ImageGenClient
        from src.core.manga_insight.config_models import ImageGenConfig

        client = ImageGenClient(
            ImageGenConfig(
                provider="gpt2api",
                api_key="test-key",
                model="gpt-image-2",
                base_url="https://gateway.example.com",
                transport_retries=0,
                business_retries=10,
                timeout_seconds=0,
            )
        )
        try:
            post_mock = mock.AsyncMock(return_value=FakeResponse(401, {"error": {"message": "unauthorized"}}))
            client.client.post = post_mock

            with self.assertRaisesRegex(ValueError, "unauthorized"):
                await client.generate("draw something")
        finally:
            await client.close()

        self.assertEqual(post_mock.await_count, 1)
        self.assertEqual(client._transport_retries, 0)

    async def test_image_gen_client_supports_newapi_with_same_openai_compatible_routes(self) -> None:
        from src.core.manga_insight.clients.image_gen_client import ImageGenClient
        from src.core.manga_insight.config_models import ImageGenConfig

        client = ImageGenClient(
            ImageGenConfig(
                provider="newapi",
                api_key="test-key",
                model="flux-dev",
                base_url="https://newapi.example.com",
            )
        )
        try:
            post_mock = mock.AsyncMock(
                return_value=FakeResponse(
                    200,
                    {"data": [{"url": "data:image/png;base64,aGVsbG8="}]},
                )
            )
            client.client.post = post_mock

            result = await client.generate("draw something")
        finally:
            await client.close()

        self.assertEqual(result, b"hello")
        post_mock.assert_awaited_once()
        self.assertEqual(post_mock.await_args.args[0], "https://newapi.example.com/v1/images/generations")
        self.assertEqual(post_mock.await_args.kwargs["json"]["model"], "flux-dev")
        self.assertEqual(post_mock.await_args.kwargs["json"]["prompt"], "draw something")

    async def test_image_gen_client_requires_model_before_request(self) -> None:
        from src.core.manga_insight.clients.image_gen_client import ImageGenClient
        from src.core.manga_insight.config_models import ImageGenConfig

        client = ImageGenClient(
            ImageGenConfig(
                provider="newapi",
                api_key="test-key",
                model="",
                base_url="https://newapi.example.com",
            )
        )
        try:
            with self.assertRaisesRegex(ValueError, "需要设置 model"):
                await client.generate("draw something")
        finally:
            await client.close()


class ImageGeneratorDelegationTests(unittest.IsolatedAsyncioTestCase):
    def test_build_full_prompt_places_style_rules_before_plot_context(self) -> None:
        from src.core.manga_insight.continuation.image_generator import ImageGenerator
        from src.core.manga_insight.continuation.models import PageContent

        generator = ImageGenerator.__new__(ImageGenerator)
        prompt = generator._build_full_prompt(
            PageContent(
                page_number=1,
                continuity_text="上一页里女主终于主动接近男主。",
                story_text="接着上一页，两人来到医院外台阶前继续对话，女主进一步表达心意。",
                dialogue_text="女主：我不后悔哦。\n男主：啊？",
                characters=["男主", "女主"],
                final_prompt="",
            )
        )

        self.assertIn("严格遵守以下风格要求", prompt)
        self.assertIn("严格沿用原作漫画的线条、上色、角色五官比例、页面密度、分镜节奏", prompt)
        self.assertIn("上一页剧情", prompt)
        self.assertIn("本页剧情", prompt)
        self.assertIn("关键对白", prompt)
        self.assertIn("如果页面内容与参考图风格冲突，优先服从参考图风格", prompt)
        self.assertLess(
            prompt.index("严格遵守以下风格要求"),
            prompt.index("上一页剧情"),
        )

    def test_compose_final_prompt_builds_plot_driven_prompt_text(self) -> None:
        from src.core.manga_insight.continuation.image_generator import ImageGenerator
        from src.core.manga_insight.continuation.models import PageContent

        generator = ImageGenerator.__new__(ImageGenerator)
        prompt = generator.compose_final_prompt(
            PageContent(
                page_number=2,
                continuity_text="第一页里主角离开教室。",
                story_text="第二页里主角走到走廊，遇到同学。",
                dialogue_text="Hero：咦？",
                characters=["Hero"],
                character_forms=[{"character": "Hero", "form_id": "battle", "form_name": "Battle Form"}],
            )
        )

        self.assertIn("上一页剧情：第一页里主角离开教室。", prompt)
        self.assertIn("本页剧情：第二页里主角走到走廊，遇到同学。", prompt)
        self.assertIn("关键对白：Hero：咦？", prompt)
        self.assertIn("出场角色：Hero", prompt)
        self.assertIn("角色形态：", prompt)
        self.assertIn("保持原作漫画线条、脸型、上色、页面密度和分镜节奏。", prompt)

    async def test_generate_page_image_rejects_empty_final_prompt(self) -> None:
        from src.core.manga_insight.continuation.image_generator import ImageGenerator
        from src.core.manga_insight.continuation.models import ContinuationCharacters, PageContent

        generator = ImageGenerator("test-book")
        try:
            with self.assertRaises(ValueError):
                await generator.generate_page_image(
                    page_content=PageContent(
                        page_number=1,
                        continuity_text="上一页剧情",
                        story_text="本页剧情",
                        dialogue_text="",
                        characters=[],
                        final_prompt="",
                    ),
                    characters=ContinuationCharacters(book_id="test-book", characters=[]),
                )
        finally:
            await generator.close()

    async def test_generate_page_image_accepts_server_composed_final_prompt(self) -> None:
        from src.core.manga_insight.continuation.image_generator import ImageGenerator
        from src.core.manga_insight.continuation.models import ContinuationCharacters, PageContent

        generator = ImageGenerator("test-book")
        try:
            prompt = generator.compose_final_prompt(
                PageContent(
                    page_number=1,
                    continuity_text="原作末页摘要",
                    story_text="主角来到走廊并遇到同学。",
                    dialogue_text="Hero：咦？",
                    characters=["Hero"],
                )
            )
            with mock.patch.object(generator._client, "generate", return_value=b"generated-image"), \
                 mock.patch.object(generator, "_save_image", return_value="saved-image.png"):
                result = await generator.generate_page_image(
                    page_content=PageContent(
                        page_number=1,
                        continuity_text="原作末页摘要",
                        story_text="主角来到走廊并遇到同学。",
                        dialogue_text="Hero：咦？",
                        characters=["Hero"],
                        final_prompt=prompt,
                    ),
                    characters=ContinuationCharacters(book_id="test-book", characters=[]),
                )
        finally:
            await generator.close()

        self.assertEqual(result, "saved-image.png")

    async def test_image_generator_delegates_page_generation_to_image_gen_client(self) -> None:
        from src.core.manga_insight.continuation.image_generator import ImageGenerator
        from src.core.manga_insight.continuation.models import ContinuationCharacters, PageContent

        generator = ImageGenerator("test-book")
        try:
            with mock.patch.object(generator, "_build_full_prompt", return_value="page prompt"), \
                 mock.patch.object(generator, "_resolve_style_reference_images", return_value=[{"path": "ref.png", "type": "style"}]), \
                 mock.patch.object(generator._client, "generate", return_value=b"generated-image") as generate_mock, \
                 mock.patch.object(generator, "_save_image", return_value="saved-image.png"):
                result = await generator.generate_page_image(
                    page_content=PageContent(
                        page_number=1,
                        continuity_text="上一页剧情",
                        story_text="本页剧情",
                        dialogue_text="关键对白",
                        characters=[],
                        final_prompt="最终提示词",
                    ),
                    characters=ContinuationCharacters(book_id="test-book", characters=[]),
                    style_reference_tokens=["original:10"],
                    style_ref_count=1,
                )
        finally:
            await generator.close()

        self.assertEqual(result, "saved-image.png")
        generate_mock.assert_awaited_once_with(
            "page prompt",
            reference_images=[{"path": "ref.png", "type": "style"}],
        )
