import unittest
from unittest import mock

import httpx


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class SharedProviderRegistryImageGenTests(unittest.TestCase):
    def test_shared_registry_exposes_openai_compatible_image_gen_providers(self) -> None:
        from src.shared.ai_providers import (
            IMAGE_GEN_CAPABILITY,
            get_provider_manifest,
            provider_supports_capability,
            resolve_provider_base_url_for_capability,
        )

        self.assertTrue(provider_supports_capability("gpt2api", IMAGE_GEN_CAPABILITY))
        self.assertTrue(provider_supports_capability("newapi", IMAGE_GEN_CAPABILITY))
        self.assertTrue(provider_supports_capability("custom", IMAGE_GEN_CAPABILITY))
        self.assertFalse(provider_supports_capability("openai", IMAGE_GEN_CAPABILITY))
        self.assertFalse(provider_supports_capability("qwen", IMAGE_GEN_CAPABILITY))
        self.assertEqual(
            get_provider_manifest("gpt2api").default_models.get("image_gen", ""),
            "gpt-image-2",
        )
        self.assertEqual(
            get_provider_manifest("newapi").default_models.get("image_gen", ""),
            "",
        )
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
            self.assertEqual(client.base_url, "https://gateway.example.com")
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
        self.assertEqual(post_mock.await_args.args[0], "https://gateway.example.com/images/generations")
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
                base_url="https://newapi.example.com/v1",
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
        from src.core.manga_insight.config_models import ImageGenConfig

        with self.assertRaisesRegex(ValueError, "model must not be empty"):
            ImageGenConfig(
                provider="newapi",
                api_key="test-key",
                model="",
                base_url="https://newapi.example.com",
            )
