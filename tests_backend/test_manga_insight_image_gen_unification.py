import sys
import types
import unittest
from unittest import mock


if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda *_args, **_kwargs: {}
    yaml_stub.safe_dump = lambda *_args, **_kwargs: ""
    sys.modules["yaml"] = yaml_stub


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class SharedProviderRegistryImageGenTests(unittest.TestCase):
    def test_shared_registry_exposes_gpt2api_as_only_image_gen_provider(self) -> None:
        from src.shared.ai_providers import (
            IMAGE_GEN_CAPABILITY,
            get_provider_default_model,
            provider_supports_capability,
            resolve_provider_base_url_for_capability,
        )

        self.assertTrue(provider_supports_capability("gpt2api", IMAGE_GEN_CAPABILITY))
        self.assertFalse(provider_supports_capability("openai", IMAGE_GEN_CAPABILITY))
        self.assertFalse(provider_supports_capability("qwen", IMAGE_GEN_CAPABILITY))
        self.assertEqual(get_provider_default_model("gpt2api", "image_gen"), "gpt-image-2")
        self.assertIsNone(resolve_provider_base_url_for_capability("gpt2api", IMAGE_GEN_CAPABILITY))


class MangaInsightImageGenClientTests(unittest.IsolatedAsyncioTestCase):
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


class ImageGeneratorDelegationTests(unittest.IsolatedAsyncioTestCase):
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
                        characters=[],
                        description="scene",
                        dialogues=[],
                        image_prompt="prompt",
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
