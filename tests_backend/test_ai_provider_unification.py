import unittest
from unittest import mock

from flask import Flask
from PIL import Image

from src.app.api.system import system_bp
from src.app.api.translation import routes as translation_routes
from src.shared.ai_transport import OpenAICompatibleChatTransport, UnifiedChatRequest


class ProviderRegistryContractTests(unittest.TestCase):
    def test_translation_provider_registry_normalizes_legacy_custom_ids(self) -> None:
        from src.shared.ai_providers import normalize_provider_id

        self.assertEqual(normalize_provider_id("custom_openai"), "custom")
        self.assertEqual(normalize_provider_id("custom_openai_vision"), "custom")
        self.assertEqual(normalize_provider_id("custom"), "custom")

    def test_translation_provider_registry_exposes_capabilities(self) -> None:
        from src.shared.ai_providers import (
            get_provider_manifest,
            provider_supports_capability,
        )

        custom_manifest = get_provider_manifest("custom")
        self.assertEqual(custom_manifest.id, "custom")
        self.assertEqual(custom_manifest.kind, "openai_compatible")
        self.assertTrue(custom_manifest.requires_base_url)
        self.assertTrue(provider_supports_capability("custom", "translation"))
        self.assertTrue(provider_supports_capability("custom", "hq_translation"))
        self.assertTrue(provider_supports_capability("custom", "vision_ocr"))

    def test_ai_vision_provider_list_matches_supported_backend_capabilities(self) -> None:
        from src.shared.ai_providers import provider_supports_capability

        self.assertFalse(provider_supports_capability("deepseek", "vision_ocr"))
        self.assertTrue(provider_supports_capability("custom", "vision_ocr"))

    def test_ai_vision_json_mode_does_not_override_custom_prompt(self) -> None:
        from src.core.ocr import recognize_ocr_results_in_bubbles

        custom_prompt = "对图中的日语进行OCR:"
        with mock.patch(
            "src.core.ocr.call_ai_vision_ocr_service",
            return_value='{"extracted_text":"测试"}',
        ) as vision_mock:
            recognize_ocr_results_in_bubbles(
                Image.new("RGB", (16, 16), color="white"),
                [(0, 0, 16, 16)],
                ocr_engine="ai_vision",
                source_language="japanese",
                ai_vision_provider="custom",
                ai_vision_api_key="vision-key",
                ai_vision_model_name="vision-model",
                ai_vision_ocr_prompt=custom_prompt,
                ai_vision_prompt_mode="paddleocr_vl",
                use_json_format_for_ai_vision=True,
                custom_ai_vision_base_url="https://example.com/v1",
            )

        self.assertEqual(vision_mock.call_args.kwargs["prompt"], custom_prompt)

    def test_ai_vision_empty_prompt_falls_back_to_json_default(self) -> None:
        from src.core.ocr import recognize_ocr_results_in_bubbles

        with mock.patch(
            "src.core.ocr.call_ai_vision_ocr_service",
            return_value='{"extracted_text":"测试"}',
        ) as vision_mock:
            recognize_ocr_results_in_bubbles(
                Image.new("RGB", (16, 16), color="white"),
                [(0, 0, 16, 16)],
                ocr_engine="ai_vision",
                source_language="japanese",
                ai_vision_provider="custom",
                ai_vision_api_key="vision-key",
                ai_vision_model_name="vision-model",
                ai_vision_ocr_prompt="",
                ai_vision_prompt_mode="json",
                use_json_format_for_ai_vision=True,
                custom_ai_vision_base_url="https://example.com/v1",
            )

        prompt = vision_mock.call_args.kwargs["prompt"]
        self.assertIn('"extracted_text"', prompt)

    def test_hq_stream_transport_prints_chunks_when_enabled(self) -> None:
        class FakeResponse:
            status_code = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b""

            def iter_lines(self):
                yield 'data: {"choices":[{"delta":{"content":"你好"}}]}'
                yield 'data: {"choices":[{"delta":{"content":"，世界"}}]}'
                yield 'data: [DONE]'

        class FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def stream(self, *args, **kwargs):
                return FakeResponse()

        transport = OpenAICompatibleChatTransport()
        request = UnifiedChatRequest(
            provider="custom",
            api_key="test-key",
            model="gpt-test",
            messages=[{"role": "user", "content": "hello"}],
            base_url="https://example.com/v1",
            use_stream=True,
            print_stream_output=True,
            stream_output_label="HQ Test",
        )

        with mock.patch("src.shared.ai_transport.httpx.Client", return_value=FakeClient()), \
             mock.patch("builtins.print") as print_mock:
            content = transport.complete(request)

        self.assertEqual(content, "你好，世界")
        printed = "\n".join(" ".join(map(str, call.args)) for call in print_mock.call_args_list)
        self.assertIn("HQ Test", printed)
        self.assertIn("你好", printed)
        self.assertIn("，世界", printed)

    def test_ai_vision_service_dispatches_supported_provider_through_shared_transport(self) -> None:
        from src.interfaces.vision_interface import call_ai_vision_ocr_service

        with mock.patch(
            "src.interfaces.vision_interface.provider_supports_capability",
            side_effect=lambda provider, capability: provider == "qwen" and capability == "vision_ocr",
        ), mock.patch(
            "src.interfaces.vision_interface._transport.complete_vision",
            return_value='{"extracted_text":"测试"}',
        ) as complete_mock:
            content = call_ai_vision_ocr_service(
                Image.new("RGB", (12, 12), color="white"),
                provider="qwen",
                api_key="vision-key",
                model_name="qwen-vl-max",
                prompt="识别图片里的文本",
                prompt_mode="normal",
                use_json_format=True,
            )

        self.assertEqual(content, '{"extracted_text":"测试"}')
        request_arg = complete_mock.call_args.args[0]
        self.assertEqual(request_arg.provider, "qwen")
        self.assertIsNone(request_arg.base_url)
        self.assertTrue(request_arg.use_json_format)


class RouteCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Flask(__name__)
        self.app.register_blueprint(translation_routes.translate_bp)
        self.app.register_blueprint(system_bp)
        self.client = self.app.test_client()

    def test_translate_single_text_accepts_canonical_custom_and_camel_case_aliases(self) -> None:
        with mock.patch(
            "src.app.api.translation.routes.translate_single_text",
            return_value="译文",
        ) as translate_mock:
            response = self.client.post(
                "/api/translate_single_text",
                json={
                    "originalText": "hello",
                    "targetLanguage": "zh",
                    "provider": "custom",
                    "apiKey": "test-key",
                    "model": "gpt-test",
                    "baseUrl": "https://example.com/v1",
                    "promptContent": "translate",
                    "useJsonFormat": True,
                    "rpmLimitTranslation": 0,
                    "maxRetries": 0,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["translated_text"], "译文")
        _, kwargs = translate_mock.call_args
        self.assertEqual(kwargs["api_key"], "test-key")
        self.assertEqual(kwargs["model_name"], "gpt-test")
        self.assertEqual(kwargs["custom_base_url"], "https://example.com/v1")
        self.assertEqual(kwargs["model_provider"], "custom")
        self.assertEqual(kwargs["max_retries"], 0)

    def test_translate_single_text_rejects_provider_without_translation_capability(self) -> None:
        response = self.client.post(
            "/api/translate_single_text",
            json={
                "originalText": "hello",
                "targetLanguage": "zh",
                "provider": "openai",
                "apiKey": "test-key",
                "model": "gpt-4.1",
            },
        )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn("不支持的服务商", payload["error"])

    def test_fetch_models_accepts_canonical_custom_provider(self) -> None:
        with mock.patch(
            "src.app.api.system.tests._chat_transport.list_models",
            return_value=[{"id": "test-model", "name": "test-model"}],
        ) as fetch_mock:
            response = self.client.post(
                "/api/fetch_models",
                json={
                    "provider": "custom",
                    "api_key": "test-key",
                    "base_url": "https://example.com/v1",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["models"][0]["id"], "test-model")
        request_arg = fetch_mock.call_args.args[0]
        self.assertEqual(request_arg.provider, "custom")
        self.assertEqual(request_arg.api_key, "test-key")
        self.assertEqual(request_arg.base_url, "https://example.com/v1")

    def test_hq_translate_batch_uses_manifest_base_url_resolution(self) -> None:
        with mock.patch(
            "src.app.api.translation.routes._hq_chat_transport.complete",
            return_value='{"images":[]}',
        ) as complete_mock:
            response = self.client.post(
                "/api/hq_translate_batch",
                json={
                    "provider": "siliconflow",
                    "api_key": "test-key",
                    "model_name": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "force_json_output": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        request_arg = complete_mock.call_args.args[0]
        self.assertEqual(request_arg.provider, "siliconflow")
        self.assertEqual(request_arg.base_url, "https://api.siliconflow.cn/v1")

    def test_hq_translate_batch_rejects_non_hq_provider_even_if_openai_compatible(self) -> None:
        response = self.client.post(
            "/api/hq_translate_batch",
            json={
                "provider": "openai",
                "api_key": "test-key",
                "model_name": "gpt-4.1",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn("不支持的服务商", payload["error"])
