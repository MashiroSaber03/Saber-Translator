import unittest
from unittest import mock
import threading
import time

from flask import Flask
from PIL import Image

from src.app.api.system import system_bp
from src.app.api.translation import parallel_routes, routes as translation_routes
from src.shared.ai_transport import OpenAICompatibleChatTransport, UnifiedChatRequest
from src.shared.openai_execution import build_openai_compatible_runtime_options
from src.shared.openai_options import (
    OpenAICompatibleExecutionOptions,
    OpenAICompatibleOptions,
    OpenAICompatibleRequestOptions,
)


class OpenAICompatibleOptionsContractTests(unittest.TestCase):
    def test_sync_chat_transport_accepts_nested_openai_options(self) -> None:
        class FakeResponse:
            status_code = 200
            text = ""

            def __init__(self):
                self.request_json = None

            def json(self):
                return {"choices": [{"message": {"content": "测试成功"}}]}

        class FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def request(self, method=None, url=None, headers=None, json=None):
                response = FakeResponse()
                response.request_json = json
                self.last_request = {"method": method, "url": url, "headers": headers, "json": json}
                return response

        transport = OpenAICompatibleChatTransport()
        request = UnifiedChatRequest(
            provider="custom",
            api_key="test-key",
            model="gpt-test",
            messages=[{"role": "user", "content": "hello"}],
            base_url="https://example.com/v1",
            openai_options=OpenAICompatibleOptions(
                request=OpenAICompatibleRequestOptions(
                    force_json_output=True,
                    temperature=0.25,
                ),
                execution=OpenAICompatibleExecutionOptions(
                    use_stream=False,
                    rpm_limit=7,
                    transport_retries=4,
                    business_retries=2,
                ),
            ),
            runtime_options=build_openai_compatible_runtime_options(
                timeout=45.0,
                request_overrides={"max_tokens": 123, "top_p": 0.8},
            ),
        )

        fake_client = FakeClient()
        with mock.patch("src.shared.ai_transport.httpx.Client", return_value=fake_client):
            content = transport.complete(request)

        self.assertEqual(content, "测试成功")
        kwargs = fake_client.last_request["json"]
        self.assertEqual(kwargs["temperature"], 0.25)
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})
        self.assertEqual(kwargs["max_tokens"], 123)
        self.assertEqual(kwargs["top_p"], 0.8)

    def test_sync_chat_transport_merges_extra_body_into_top_level_request_body(self) -> None:
        class FakeResponse:
            status_code = 200
            text = ""

            def json(self):
                return {"choices": [{"message": {"content": "测试成功"}}]}

        class FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def request(self, method=None, url=None, headers=None, json=None):
                self.last_request = {"method": method, "url": url, "headers": headers, "json": json}
                return FakeResponse()

        transport = OpenAICompatibleChatTransport()
        request = UnifiedChatRequest(
            provider="custom",
            api_key="test-key",
            model="gpt-test",
            messages=[{"role": "user", "content": "hello"}],
            base_url="https://example.com/v1",
            openai_options=OpenAICompatibleOptions(
                request=OpenAICompatibleRequestOptions(
                    temperature=0.25,
                    extra_body={
                        "thinking": {"type": "disabled"},
                        "reasoning_effort": "low",
                    },
                ),
            ),
            runtime_options=build_openai_compatible_runtime_options(
                request_overrides={"max_tokens": 123},
            ),
        )

        fake_client = FakeClient()
        with mock.patch("src.shared.ai_transport.httpx.Client", return_value=fake_client):
            content = transport.complete(request)

        self.assertEqual(content, "测试成功")
        kwargs = fake_client.last_request["json"]
        self.assertEqual(kwargs["temperature"], 0.25)
        self.assertEqual(kwargs["thinking"], {"type": "disabled"})
        self.assertEqual(kwargs["reasoning_effort"], "low")
        self.assertEqual(kwargs["max_tokens"], 123)

    def test_sync_chat_transport_rejects_reserved_extra_body_keys(self) -> None:
        transport = OpenAICompatibleChatTransport()
        request = UnifiedChatRequest(
            provider="custom",
            api_key="test-key",
            model="gpt-test",
            messages=[{"role": "user", "content": "hello"}],
            base_url="https://example.com/v1",
            openai_options=OpenAICompatibleOptions(
                request=OpenAICompatibleRequestOptions(
                    extra_body={"model": "override-model"},
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "extra_body"):
            transport.complete(request)

class ProviderRegistryContractTests(unittest.TestCase):
    def test_ai_vision_rpm_limit_is_scoped_per_provider(self) -> None:
        from src.shared.ai_providers import VISION_OCR_CAPABILITY
        from src.shared.openai_rate_limits import (
            apply_sync_rpm_limit,
            build_openai_rpm_bucket_key,
        )

        seen_refs = []

        def record_refs(rpm_limit, service_name, last_reset_ref, request_count_ref):
            seen_refs.append((service_name, last_reset_ref, request_count_ref))

        apply_sync_rpm_limit(
            build_openai_rpm_bucket_key(VISION_OCR_CAPABILITY, "siliconflow"),
            5,
            "AI Vision OCR (siliconflow)",
            record_refs,
        )
        apply_sync_rpm_limit(
            build_openai_rpm_bucket_key(VISION_OCR_CAPABILITY, "gemini"),
            5,
            "AI Vision OCR (gemini)",
            record_refs,
        )

        self.assertEqual(len(seen_refs), 2)
        self.assertNotEqual(seen_refs[0][0], seen_refs[1][0])
        self.assertIsNot(seen_refs[0][1], seen_refs[1][1])
        self.assertIsNot(seen_refs[0][2], seen_refs[1][2])

    def test_ai_vision_rpm_limit_serializes_same_provider_calls(self) -> None:
        from src.shared.ai_providers import VISION_OCR_CAPABILITY
        from src.shared.openai_rate_limits import (
            apply_sync_rpm_limit,
            build_openai_rpm_bucket_key,
        )

        start_event = threading.Event()
        active = 0
        overlap_detected = False
        state_lock = threading.Lock()

        def slow_limit(*args, **kwargs):
            nonlocal active, overlap_detected
            with state_lock:
                active += 1
                if active > 1:
                    overlap_detected = True
            time.sleep(0.05)
            with state_lock:
                active -= 1

        def worker():
            start_event.wait()
            apply_sync_rpm_limit(
                build_openai_rpm_bucket_key(VISION_OCR_CAPABILITY, "siliconflow"),
                5,
                "AI Vision OCR (siliconflow)",
                slow_limit,
            )

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        start_event.set()
        for thread in threads:
            thread.join()

        self.assertFalse(overlap_detected)

    def test_translation_rpm_limit_is_scoped_per_provider(self) -> None:
        from src.core.translation import _apply_translation_rpm_limit

        seen_refs = []

        def record_refs(rpm_limit, service_name, last_reset_ref, request_count_ref):
            seen_refs.append((service_name, last_reset_ref, request_count_ref))

        with mock.patch("src.core.translation._enforce_rpm_limit", side_effect=record_refs):
            _apply_translation_rpm_limit("siliconflow", 5, batch=False)
            _apply_translation_rpm_limit("gemini", 5, batch=False)

        self.assertEqual(len(seen_refs), 2)
        self.assertIsNot(seen_refs[0][1], seen_refs[1][1])
        self.assertIsNot(seen_refs[0][2], seen_refs[1][2])

    def test_hq_translation_rpm_limit_is_scoped_per_provider(self) -> None:
        from src.shared.ai_providers import HQ_TRANSLATION_CAPABILITY
        from src.shared.openai_rate_limits import (
            apply_sync_rpm_limit,
            build_openai_rpm_bucket_key,
        )

        seen_refs = []

        def record_refs(rpm_limit, service_name, last_reset_ref, request_count_ref):
            seen_refs.append((service_name, last_reset_ref, request_count_ref))

        apply_sync_rpm_limit(
            build_openai_rpm_bucket_key(HQ_TRANSLATION_CAPABILITY, "siliconflow"),
            5,
            "HQTranslation (siliconflow)",
            record_refs,
        )
        apply_sync_rpm_limit(
            build_openai_rpm_bucket_key(HQ_TRANSLATION_CAPABILITY, "gemini"),
            5,
            "HQTranslation (gemini)",
            record_refs,
        )

        self.assertEqual(len(seen_refs), 2)
        self.assertIsNot(seen_refs[0][1], seen_refs[1][1])
        self.assertIsNot(seen_refs[0][2], seen_refs[1][2])

    def test_translate_single_text_attempts_once_when_max_retries_is_zero(self) -> None:
        from src.core.translation import translate_single_text
        from src.shared.openai_options import OpenAICompatibleOptions

        with mock.patch(
            "src.core.translation._chat_transport.complete",
            return_value='{"translated_text":"你好"}',
        ) as complete_mock:
            translated = translate_single_text(
                text="どれーせ！！",
                target_language="zh",
                model_provider="siliconflow",
                api_key="test-key",
                model_name="test-model",
                openai_options=OpenAICompatibleOptions.from_dict(
                    {
                        "request": {"force_json_output": True},
                        "execution": {"business_retries": 0, "rpm_limit": 0, "use_stream": False},
                    }
                ),
            )

        self.assertEqual(translated, "你好")
        complete_mock.assert_called_once()

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

    def test_ollama_manifest_is_local_openai_compatible_provider(self) -> None:
        from src.shared.ai_providers import (
            get_provider_manifest,
            provider_supports_capability,
        )

        manifest = get_provider_manifest("ollama")

        self.assertEqual(manifest.id, "ollama")
        self.assertEqual(manifest.kind, "openai_compatible")
        self.assertTrue(manifest.is_local)
        self.assertFalse(manifest.requires_api_key)
        self.assertEqual(manifest.default_base_url, "http://localhost:11434/v1")
        self.assertTrue(manifest.supports_stream)
        self.assertTrue(manifest.supports_json_response)
        self.assertTrue(provider_supports_capability("ollama", "translation"))
        self.assertTrue(provider_supports_capability("ollama", "hq_translation"))
        self.assertTrue(provider_supports_capability("ollama", "vision_ocr"))
        self.assertTrue(provider_supports_capability("ollama", "vlm"))
        self.assertTrue(provider_supports_capability("ollama", "chat"))
        self.assertTrue(provider_supports_capability("ollama", "embedding"))
        self.assertTrue(provider_supports_capability("ollama", "web_import_agent"))

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
                custom_ai_vision_base_url="https://example.com/v1",
                ai_vision_openai_options=OpenAICompatibleOptions(
                    request=OpenAICompatibleRequestOptions(force_json_output=True),
                ),
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
                custom_ai_vision_base_url="https://example.com/v1",
                ai_vision_openai_options=OpenAICompatibleOptions(
                    request=OpenAICompatibleRequestOptions(force_json_output=True),
                ),
            )

        prompt = vision_mock.call_args.kwargs["prompt"]
        self.assertIn('"extracted_text"', prompt)

    def test_ai_vision_retries_empty_results_when_max_retries_configured(self) -> None:
        from src.interfaces.vision_interface import call_ai_vision_ocr_service

        with mock.patch(
            "src.interfaces.vision_interface._transport.complete_vision",
            side_effect=[ValueError("AI 未返回有效内容"), '{"extracted_text":"测试"}'],
        ) as complete_mock, mock.patch("src.shared.openai_execution.time.sleep"):
            content = call_ai_vision_ocr_service(
                Image.new("RGB", (16, 16), color="white"),
                provider="custom",
                api_key="vision-key",
                model_name="vision-model",
                prompt="识别图片里的文本",
                custom_base_url="https://example.com/v1",
                openai_options=OpenAICompatibleOptions(
                    request=OpenAICompatibleRequestOptions(force_json_output=True),
                    execution=OpenAICompatibleExecutionOptions(business_retries=1),
                ),
            )

        self.assertEqual(complete_mock.call_count, 2)
        self.assertEqual(content, "测试")

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
            openai_options=OpenAICompatibleOptions(
                execution=OpenAICompatibleExecutionOptions(use_stream=True),
            ),
            runtime_options=build_openai_compatible_runtime_options(
                print_stream_output=True,
                stream_output_label="HQ Test",
            ),
        )

        with mock.patch("src.shared.ai_transport.httpx.Client", return_value=FakeClient()), \
             mock.patch("builtins.print") as print_mock:
            content = transport.complete(request)

        self.assertEqual(content, "你好，世界")
        printed = "\n".join(" ".join(map(str, call.args)) for call in print_mock.call_args_list)
        self.assertIn("HQ Test", printed)
        self.assertIn("你好", printed)
        self.assertIn("，世界", printed)

    def test_sync_chat_transport_merges_temperature_and_request_overrides_as_top_level_fields(self) -> None:
        class FakeResponse:
            status_code = 200
            text = ""

            def json(self):
                return {"choices": [{"message": {"content": "测试成功"}}]}

        class FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def request(self, method=None, url=None, headers=None, json=None):
                self.last_request = {"method": method, "url": url, "headers": headers, "json": json}
                return FakeResponse()

        transport = OpenAICompatibleChatTransport()
        request = UnifiedChatRequest(
            provider="custom",
            api_key="test-key",
            model="gpt-test",
            messages=[{"role": "user", "content": "hello"}],
            base_url="https://example.com/v1",
            openai_options=OpenAICompatibleOptions(
                request=OpenAICompatibleRequestOptions(temperature=0.25),
            ),
            runtime_options=build_openai_compatible_runtime_options(
                request_overrides={"max_tokens": 123, "top_p": 0.8},
            ),
        )

        fake_client = FakeClient()
        with mock.patch("src.shared.ai_transport.httpx.Client", return_value=fake_client):
            content = transport.complete(request)

        self.assertEqual(content, "测试成功")
        kwargs = fake_client.last_request["json"]
        self.assertEqual(kwargs["temperature"], 0.25)
        self.assertEqual(kwargs["max_tokens"], 123)
        self.assertEqual(kwargs["top_p"], 0.8)
        self.assertNotIn("extra_body", kwargs)

    def test_create_openai_client_uses_placeholder_key_for_local_services_without_api_key(self) -> None:
        from src.shared.openai_helpers import create_openai_client

        fake_http_client = mock.Mock()

        with mock.patch("src.shared.openai_helpers.httpx.Client", return_value=fake_http_client), \
             mock.patch("src.shared.openai_helpers.OpenAI") as openai_ctor:
            create_openai_client(
                api_key="",
                base_url="http://localhost:11434/v1",
                timeout=30,
            )

        self.assertEqual(openai_ctor.call_args.kwargs["api_key"], "ollama")

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
                openai_options=OpenAICompatibleOptions(
                    request=OpenAICompatibleRequestOptions(force_json_output=True),
                ),
            )

        self.assertEqual(content, '测试')
        request_arg = complete_mock.call_args.args[0]
        self.assertEqual(request_arg.provider, "qwen")
        self.assertIsNone(request_arg.base_url)
        self.assertTrue(request_arg.use_json_format)


class RouteCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Flask(__name__)
        self.app.register_blueprint(translation_routes.translate_bp)
        self.app.register_blueprint(parallel_routes.parallel_bp)
        self.app.register_blueprint(system_bp)
        self.client = self.app.test_client()

    def test_translate_single_text_accepts_nested_openai_options(self) -> None:
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
                    "openai_options": {
                        "request": {"force_json_output": True},
                        "execution": {
                            "use_stream": False,
                            "rpm_limit": 0,
                            "transport_retries": 1,
                            "business_retries": 0,
                        },
                    },
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
        self.assertTrue(kwargs["openai_options"].request.force_json_output)
        self.assertEqual(kwargs["openai_options"].execution.business_retries, 0)
        self.assertNotIn("max_retries", kwargs)
        self.assertNotIn("use_json_format", kwargs)
        self.assertNotIn("rpm_limit_translation", kwargs)

    def test_translate_single_text_accepts_extra_body_in_openai_options(self) -> None:
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
                    "openai_options": {
                        "request": {
                            "extra_body": {
                                "thinking": {"type": "disabled"},
                            },
                        },
                        "execution": {
                            "use_stream": False,
                            "rpm_limit": 0,
                            "transport_retries": 1,
                            "business_retries": 0,
                        },
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            translate_mock.call_args.kwargs["openai_options"].request.extra_body,
            {"thinking": {"type": "disabled"}},
        )

    def test_translate_single_text_rejects_non_object_extra_body(self) -> None:
        with mock.patch("src.app.api.translation.routes.translate_single_text") as translate_mock:
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
                    "openai_options": {
                        "request": {"extra_body": []},
                        "execution": {
                            "use_stream": False,
                            "rpm_limit": 0,
                            "transport_retries": 1,
                            "business_retries": 0,
                        },
                    },
                },
            )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn("openai_options.request.extra_body", payload["error"])
        translate_mock.assert_not_called()

    def test_translate_single_text_rejects_reserved_extra_body_keys(self) -> None:
        with mock.patch("src.app.api.translation.routes.translate_single_text") as translate_mock:
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
                    "openai_options": {
                        "request": {"extra_body": {"stream": True}},
                        "execution": {
                            "use_stream": False,
                            "rpm_limit": 0,
                            "transport_retries": 1,
                            "business_retries": 0,
                        },
                    },
                },
            )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn("openai_options.request.extra_body.stream", payload["error"])
        translate_mock.assert_not_called()

    def test_translate_single_text_rejects_legacy_openai_request_fields(self) -> None:
        with mock.patch("src.app.api.translation.routes.translate_single_text") as translate_mock:
            response = self.client.post(
                "/api/translate_single_text",
                json={
                    "original_text": "hello",
                    "target_language": "zh",
                    "model_provider": "custom",
                    "api_key": "test-key",
                    "model_name": "gpt-test",
                    "custom_base_url": "https://example.com/v1",
                    "use_json_format": True,
                    "rpm_limit_translation": 0,
                    "max_retries": 0,
                },
            )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn("openai_options", payload["error"])
        translate_mock.assert_not_called()

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

    def test_translate_single_text_accepts_ollama_without_api_key(self) -> None:
        with mock.patch(
            "src.app.api.translation.routes.translate_single_text",
            return_value="译文",
        ) as translate_mock:
            response = self.client.post(
                "/api/translate_single_text",
                json={
                    "originalText": "hello",
                    "targetLanguage": "zh",
                    "provider": "ollama",
                    "model": "llama3.2",
                    "openai_options": {
                        "request": {"force_json_output": True},
                        "execution": {
                            "use_stream": False,
                            "rpm_limit": 0,
                            "transport_retries": 1,
                            "business_retries": 0,
                        },
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["translated_text"], "译文")
        self.assertEqual(translate_mock.call_args.kwargs["api_key"], None)
        self.assertEqual(translate_mock.call_args.kwargs["model_provider"], "ollama")

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

    def test_fetch_models_allows_ollama_without_api_key(self) -> None:
        with mock.patch(
            "src.app.api.system.tests._chat_transport.list_models",
            return_value=[{"id": "llama3.2", "name": "llama3.2"}],
        ) as fetch_mock:
            response = self.client.post(
                "/api/fetch_models",
                json={
                    "provider": "ollama",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        request_arg = fetch_mock.call_args.args[0]
        self.assertEqual(request_arg.provider, "ollama")
        self.assertEqual(request_arg.api_key, "")
        self.assertIsNone(request_arg.base_url)

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
                    "openai_options": {
                        "request": {"force_json_output": True},
                        "execution": {
                            "use_stream": False,
                            "rpm_limit": 0,
                            "transport_retries": 1,
                            "business_retries": 0,
                        },
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        request_arg = complete_mock.call_args.args[0]
        self.assertEqual(request_arg.provider, "siliconflow")
        self.assertEqual(request_arg.base_url, "https://api.siliconflow.cn/v1")

    def test_hq_translate_batch_accepts_ollama_without_api_key(self) -> None:
        with mock.patch(
            "src.app.api.translation.routes._hq_chat_transport.complete",
            return_value='{"images":[]}',
        ) as complete_mock:
            response = self.client.post(
                "/api/hq_translate_batch",
                json={
                    "provider": "ollama",
                    "model_name": "llama3.2-vision",
                    "messages": [{"role": "user", "content": "hello"}],
                    "openai_options": {
                        "request": {"force_json_output": True},
                        "execution": {
                            "use_stream": False,
                            "rpm_limit": 0,
                            "transport_retries": 1,
                            "business_retries": 0,
                        },
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        request_arg = complete_mock.call_args.args[0]
        self.assertEqual(request_arg.provider, "ollama")
        self.assertEqual(request_arg.api_key, "")
        self.assertEqual(request_arg.base_url, "http://localhost:11434/v1")

    def test_ai_translate_connection_accepts_ollama_without_api_key(self) -> None:
        with mock.patch(
            "src.app.api.system.tests._chat_transport.test_connection",
            return_value=(True, "你好"),
        ) as test_mock:
            response = self.client.post(
                "/api/test_ai_translate_connection",
                json={
                    "provider": "ollama",
                    "model_name": "llama3.2",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        request_arg = test_mock.call_args.args[0]
        self.assertEqual(request_arg.provider, "ollama")
        self.assertEqual(request_arg.api_key, "")

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

    def test_hq_translate_batch_rejects_legacy_openai_request_fields(self) -> None:
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
                    "force_json_output": False,
                    "use_stream": False,
                    "rpm_limit": 1,
                    "max_retries": 1,
                },
            )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn("openai_options", payload["error"])
        complete_mock.assert_not_called()

    def test_ocr_single_bubble_accepts_new_openai_options_payload(self) -> None:
        with mock.patch(
            "src.app.api.translation.routes.recognize_ocr_results_in_bubbles",
            return_value=[mock.Mock(text="测试", to_dict=mock.Mock(return_value={"text": "测试"}))],
        ) as recognize_mock:
            response = self.client.post(
                "/api/ocr_single_bubble",
                json={
                    "bubble_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5N0X8AAAAASUVORK5CYII=",
                    "bubble_coords": [0, 0, 1, 1],
                    "ocr_engine": "ai_vision",
                    "source_language": "japanese",
                    "ai_vision_provider": "gemini",
                    "ai_vision_api_key": "vision-key",
                    "ai_vision_model_name": "gemini-2.0-flash",
                    "openai_options": {
                        "request": {"force_json_output": True},
                        "execution": {"rpm_limit": 13},
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        kwargs = recognize_mock.call_args.kwargs
        self.assertTrue(kwargs["ai_vision_openai_options"].request.force_json_output)
        self.assertEqual(kwargs["ai_vision_openai_options"].execution.rpm_limit, 13)
        self.assertNotIn("use_json_format_for_ai_vision", kwargs)
        self.assertNotIn("rpm_limit_ai_vision", kwargs)

    def test_ocr_single_bubble_rejects_legacy_openai_request_fields(self) -> None:
        with mock.patch("src.app.api.translation.routes.recognize_ocr_results_in_bubbles") as recognize_mock:
            response = self.client.post(
                "/api/ocr_single_bubble",
                json={
                    "bubble_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5N0X8AAAAASUVORK5CYII=",
                    "bubble_coords": [0, 0, 1, 1],
                    "ocr_engine": "ai_vision",
                    "source_language": "japanese",
                    "ai_vision_provider": "gemini",
                    "ai_vision_api_key": "vision-key",
                    "ai_vision_model_name": "gemini-2.0-flash",
                    "use_json_format_for_ai_vision": True,
                    "rpm_limit_ai_vision": 13,
                },
            )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn("openai_options", payload["error"])
        recognize_mock.assert_not_called()

    def test_parallel_translate_rejects_legacy_openai_request_fields(self) -> None:
        with mock.patch("src.app.api.translation.parallel_routes.translate_text_list") as translate_mock:
            response = self.client.post(
                "/api/parallel/translate",
                json={
                    "original_texts": ["hello"],
                    "target_language": "zh",
                    "model_provider": "custom",
                    "api_key": "test-key",
                    "model_name": "gpt-test",
                    "custom_base_url": "https://example.com/v1",
                    "use_json_format": True,
                    "rpm_limit": 0,
                    "max_retries": 0,
                },
            )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn("openai_options", payload["error"])
        translate_mock.assert_not_called()

    def test_parallel_ocr_rejects_legacy_openai_request_fields(self) -> None:
        with mock.patch("src.app.api.translation.parallel_routes.recognize_ocr_results_in_bubbles") as recognize_mock:
            response = self.client.post(
                "/api/parallel/ocr",
                json={
                    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5N0X8AAAAASUVORK5CYII=",
                    "bubble_coords": [[0, 0, 1, 1]],
                    "ocr_engine": "ai_vision",
                    "source_language": "japanese",
                    "ai_vision_provider": "gemini",
                    "ai_vision_api_key": "vision-key",
                    "ai_vision_model_name": "gemini-2.0-flash",
                    "use_json_format_for_ai_vision": True,
                    "rpm_limit_ai_vision": 13,
                },
            )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn("openai_options", payload["error"])
        recognize_mock.assert_not_called()
