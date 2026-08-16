import unittest
from unittest import mock

from PIL import Image

from src.shared.ai_transport import OpenAICompatibleChatTransport, UnifiedChatRequest
from src.shared.openai_execution import (
    OpenAICompatibleEmptyContentError,
    build_openai_compatible_runtime_options,
    resolve_openai_compatible_invocation,
)
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
                    extra_body={"seed": 123, "presence_penalty": 0.8},
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
            ),
        )

        fake_client = FakeClient()
        with mock.patch("src.shared.ai_transport.httpx.Client", return_value=fake_client):
            content = transport.complete(request)

        self.assertEqual(content, "测试成功")
        kwargs = fake_client.last_request["json"]
        self.assertEqual(kwargs["temperature"], 0.25)
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})
        self.assertEqual(kwargs["seed"], 123)
        self.assertEqual(kwargs["presence_penalty"], 0.8)

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
                        "seed": 123,
                    },
                ),
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
        self.assertEqual(kwargs["seed"], 123)

    def test_sync_chat_transport_preserves_provider_specific_limits_from_extra_body(self) -> None:
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
                    extra_body={
                        "max_tokens": 321,
                        "top_p": 0.6,
                    },
                ),
            ),
        )

        fake_client = FakeClient()
        with mock.patch("src.shared.ai_transport.httpx.Client", return_value=fake_client):
            content = transport.complete(request)

        self.assertEqual(content, "测试成功")
        kwargs = fake_client.last_request["json"]
        self.assertEqual(kwargs["max_tokens"], 321)
        self.assertEqual(kwargs["top_p"], 0.6)

    def test_sync_chat_transport_rejects_reserved_extra_body_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "extra_body"):
            OpenAICompatibleRequestOptions(
                extra_body={"model": "override-model"},
            )

    def test_sync_chat_transport_connection_test_does_not_send_default_max_tokens(self) -> None:
        from src.shared.ai_transport import ProviderConnectionTestRequest

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
        fake_client = FakeClient()

        with mock.patch("src.shared.ai_transport.httpx.Client", return_value=fake_client):
            success, message = transport.test_connection(
                ProviderConnectionTestRequest(
                    provider="custom",
                    api_key="test-key",
                    model="gpt-test",
                    base_url="https://example.com/v1",
                )
            )

        self.assertTrue(success)
        self.assertEqual(message, "测试成功")
        self.assertNotIn("max_tokens", fake_client.last_request["json"])

    def test_model_listing_appends_only_models_to_the_configured_base_url(self) -> None:
        from src.shared.ai_transport import ProviderModelListRequest

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"data": [{"id": "model-b"}, {"id": "model-a"}]}

        class FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def get(self, url, **kwargs):
                self.url = url
                return FakeResponse()

        fake_client = FakeClient()
        with mock.patch("src.shared.ai_transport.httpx.Client", return_value=fake_client):
            models = OpenAICompatibleChatTransport().list_models(
                ProviderModelListRequest(
                    provider="custom",
                    api_key="test-key",
                    base_url="https://example.com/openai",
                )
            )

        self.assertEqual(fake_client.url, "https://example.com/openai/models")
        self.assertEqual([item["id"] for item in models], ["model-a", "model-b"])

    def test_model_listing_rejects_unsupported_provider_before_network(self) -> None:
        from src.shared.ai_transport import ProviderModelListRequest

        with mock.patch("src.shared.ai_transport.httpx.Client") as client_factory, \
             self.assertRaisesRegex(ValueError, "不支持模型列表"):
            OpenAICompatibleChatTransport().list_models(
                ProviderModelListRequest(
                    provider="gpt2api",
                    api_key="test-key",
                    base_url="https://example.com/v1",
                )
            )

        client_factory.assert_not_called()

    def test_openai_options_reject_partial_or_coerced_current_schema(self) -> None:
        invalid_values = [
            {},
            {"request": {}, "execution": {}},
            {
                "request": {
                    "force_json_output": "false",
                    "temperature": None,
                    "extra_body": {},
                },
                "execution": {
                    "use_stream": False,
                    "rpm_limit": 0,
                    "transport_retries": 1,
                    "business_retries": 0,
                },
            },
            {
                "request": {
                    "force_json_output": False,
                    "temperature": None,
                    "extra_body": {},
                },
                "execution": {
                    "use_stream": False,
                    "rpm_limit": "0",
                    "transport_retries": 1,
                    "business_retries": 0,
                },
            },
        ]

        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                OpenAICompatibleOptions.from_dict(value)

    def test_openai_extra_body_rejects_non_json_or_nonfinite_values(self) -> None:
        for value in ({"opaque": object()}, {"temperature_hint": float("nan")}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                OpenAICompatibleRequestOptions(extra_body=value)

        options = OpenAICompatibleOptions()
        options.request.extra_body = None  # type: ignore[assignment]
        with self.assertRaisesRegex(ValueError, "extra_body"):
            options.to_dict()

    def test_sync_stream_accepts_sse_data_without_optional_space(self) -> None:
        class FakeResponse:
            status_code = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def iter_lines(self):
                yield 'data:{"choices":[{"delta":{"content":"成功"}}]}'
                yield "data:[DONE]"

        class FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def stream(self, *args, **kwargs):
                return FakeResponse()

        request = UnifiedChatRequest(
            provider="custom",
            api_key="test-key",
            model="gpt-test",
            messages=[{"role": "user", "content": "hello"}],
            base_url="https://example.com/v1",
            openai_options=OpenAICompatibleOptions(
                execution=OpenAICompatibleExecutionOptions(use_stream=True),
            ),
        )

        with mock.patch("src.shared.ai_transport.httpx.Client", return_value=FakeClient()):
            content = OpenAICompatibleChatTransport().complete(request)

        self.assertEqual(content, "成功")

    def test_sync_stream_rejects_malformed_json_instead_of_returning_partial_success(self) -> None:
        class FakeResponse:
            status_code = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def iter_lines(self):
                yield 'data:{"choices":[{"delta":{"content":"部分"}}]}'
                yield "data:{broken"

        class FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def stream(self, *args, **kwargs):
                return FakeResponse()

        request = UnifiedChatRequest(
            provider="custom",
            api_key="test-key",
            model="gpt-test",
            messages=[{"role": "user", "content": "hello"}],
            base_url="https://example.com/v1",
            openai_options=OpenAICompatibleOptions(
                execution=OpenAICompatibleExecutionOptions(use_stream=True),
            ),
        )

        with mock.patch("src.shared.ai_transport.httpx.Client", return_value=FakeClient()), \
             self.assertRaisesRegex(ValueError, "无效 JSON"):
            OpenAICompatibleChatTransport().complete(request)

    def test_sync_stream_keepalive_frames_cannot_extend_the_attempt_deadline(self) -> None:
        import httpx

        class FakeResponse:
            status_code = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def iter_lines(self):
                yield ": keep-alive"

        class FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def stream(self, *args, **kwargs):
                return FakeResponse()

        request = UnifiedChatRequest(
            provider="custom",
            api_key="test-key",
            model="gpt-test",
            messages=[{"role": "user", "content": "hello"}],
            base_url="https://example.com/v1",
            openai_options=OpenAICompatibleOptions(
                execution=OpenAICompatibleExecutionOptions(
                    use_stream=True,
                    transport_retries=0,
                ),
            ),
            runtime_options=build_openai_compatible_runtime_options(timeout=30),
        )

        with mock.patch("src.shared.ai_transport.httpx.Client", return_value=FakeClient()), \
             mock.patch("src.shared.ai_transport.time.monotonic", side_effect=[0, 31]), \
             self.assertRaises(httpx.ReadTimeout):
            OpenAICompatibleChatTransport().complete(request)

    def test_remote_provider_missing_api_key_fails_before_network_request(self) -> None:
        request = UnifiedChatRequest(
            provider="custom",
            api_key="",
            model="gpt-test",
            messages=[{"role": "user", "content": "hello"}],
            base_url="https://example.com/v1",
        )

        with mock.patch("src.shared.ai_transport.httpx.Client") as client_factory, \
             self.assertRaisesRegex(ValueError, "API Key"):
            OpenAICompatibleChatTransport().complete(request)

        client_factory.assert_not_called()

class ProviderRegistryContractTests(unittest.TestCase):
    def test_provider_manifest_rejects_unknown_fields_and_duplicate_ids(self) -> None:
        from src.shared.ai_providers import (
            _build_provider_manifest,
            _load_provider_manifest_data,
        )

        entries = _load_provider_manifest_data()
        with self.assertRaisesRegex(RuntimeError, "多余"):
            _build_provider_manifest({**entries[0], "legacyIds": ["old"]})
        duplicate = [entries[0], dict(entries[0])]
        with mock.patch("src.shared.ai_providers.json.load", return_value=duplicate), \
             self.assertRaisesRegex(RuntimeError, "必须唯一"):
            _load_provider_manifest_data()

    def test_provider_manifest_rejects_malformed_current_collections_cleanly(self) -> None:
        from src.shared.ai_providers import (
            _build_provider_manifest,
            _load_provider_manifest_data,
        )

        entry = _load_provider_manifest_data()[0]
        invalid_entries = [
            ({**entry, "kind": []}, "kind 无效"),
            ({**entry, "capabilities": [{}]}, "capabilities 无效"),
            ({**entry, "capabilityEndpoints": []}, "capabilityEndpoints 必须是对象"),
        ]
        for invalid_entry, message in invalid_entries:
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                _build_provider_manifest(invalid_entry)

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
                        "request": {
                            "force_json_output": True,
                            "temperature": None,
                            "extra_body": {},
                        },
                        "execution": {
                            "business_retries": 0,
                            "rpm_limit": 0,
                            "transport_retries": 1,
                            "use_stream": False,
                        },
                    }
                ),
            )

        self.assertEqual(translated, "你好")
        complete_mock.assert_called_once()

    def test_translation_provider_registry_keeps_current_provider_ids(self) -> None:
        from src.shared.ai_providers import normalize_provider_id

        self.assertEqual(normalize_provider_id(" CUSTOM "), "custom")
        self.assertEqual(normalize_provider_id("custom"), "custom")
        self.assertEqual(normalize_provider_id("unknown-provider"), "unknown-provider")

    def test_unknown_provider_model_requirement_is_not_guessed(self) -> None:
        from src.shared.ai_providers import provider_requires_model

        with self.assertRaisesRegex(ValueError, "未知的 AI 服务商"):
            provider_requires_model("unknown-provider")

    def test_local_service_detection_uses_parsed_hostname_only(self) -> None:
        from src.shared.http_config import is_local_service

        self.assertTrue(is_local_service("http://localhost:11434/v1"))
        self.assertTrue(is_local_service("http://[::1]:11434/v1"))
        self.assertFalse(is_local_service("https://example.com/?next=http://localhost"))

    def test_openai_api_key_resolution_rejects_non_string_values(self) -> None:
        from src.shared.openai_helpers import resolve_openai_api_key

        with self.assertRaisesRegex(TypeError, "api_key"):
            resolve_openai_api_key(0)  # type: ignore[arg-type]

    def test_denied_shared_rate_limit_requires_positive_retry_delay(self) -> None:
        from src.shared.openai_rate_limits import (
            RateLimitDecision,
            SharedRPMLimiter,
            configure_provider_rate_limit_store,
        )

        class DenyingStore:
            def acquire(self, **kwargs):
                return RateLimitDecision(
                    allowed=False,
                    remaining=0,
                    retry_after_seconds=0,
                )

        configure_provider_rate_limit_store(DenyingStore())
        try:
            limiter = SharedRPMLimiter(
                1,
                provider="custom",
                credential_version_id="credential-v1",
            )
            with self.assertRaisesRegex(RuntimeError, "positive retry delay"):
                limiter._acquire()
        finally:
            configure_provider_rate_limit_store(None)

    def test_unsupported_stream_and_json_modes_are_rejected_not_downgraded(self) -> None:
        with self.assertRaisesRegex(ValueError, "不支持流式调用"):
            resolve_openai_compatible_invocation(
                "sakura",
                "translation",
                OpenAICompatibleOptions(
                    execution=OpenAICompatibleExecutionOptions(use_stream=True),
                ),
            )
        with self.assertRaisesRegex(ValueError, "不支持强制 JSON 输出"):
            resolve_openai_compatible_invocation(
                "sakura",
                "translation",
                OpenAICompatibleOptions(
                    request=OpenAICompatibleRequestOptions(force_json_output=True),
                ),
            )

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

    def test_provider_manifest_only_exposes_endpoints_for_supported_capabilities(self) -> None:
        from src.shared.ai_providers import (
            resolve_provider_endpoint_for_capability,
        )

        self.assertIsNone(
            resolve_provider_endpoint_for_capability("baidu_translate", "rerank")
        )
        self.assertEqual(
            resolve_provider_endpoint_for_capability("siliconflow", "rerank"),
            "/rerank",
        )

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

        custom_prompt = "保留用户自定义 OCR 提示词"
        with mock.patch(
            "src.core.ocr.call_ai_vision_ocr_service",
            return_value='{"extracted_text":"测试"}',
        ) as vision_mock:
            recognize_ocr_results_in_bubbles(
                Image.new("RGB", (16, 16), color="white"),
                [(0, 0, 16, 16)],
                ocr_engine="ai_vision",
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

    def test_ai_vision_empty_paddle_prompt_uses_language_aware_default(self) -> None:
        from src.core.ocr import recognize_ocr_results_in_bubbles
        from src.shared.paddleocr_vl import build_paddleocr_vl_prompt

        with mock.patch(
            "src.core.ocr.call_ai_vision_ocr_service",
            return_value="测试",
        ) as vision_mock:
            recognize_ocr_results_in_bubbles(
                Image.new("RGB", (16, 16), color="white"),
                [(0, 0, 16, 16)],
                ocr_engine="ai_vision",
                ai_vision_provider="custom",
                ai_vision_api_key="vision-key",
                ai_vision_model_name="vision-model",
                ai_vision_ocr_prompt="",
                ai_vision_prompt_mode="paddleocr_vl",
                custom_ai_vision_base_url="https://example.com/v1",
            )

        self.assertEqual(
            vision_mock.call_args.kwargs["prompt"],
            build_paddleocr_vl_prompt("japanese"),
        )

    def test_ai_vision_retries_empty_results_when_max_retries_configured(self) -> None:
        from src.interfaces.vision_interface import call_ai_vision_ocr_service

        with mock.patch(
            "src.interfaces.vision_interface._transport.complete_vision",
            side_effect=[
                OpenAICompatibleEmptyContentError("AI 未返回有效内容"),
                '{"extracted_text":"测试"}',
            ],
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

    def test_hq_stream_transport_invokes_stream_callback_with_cumulative_text(self) -> None:
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

        seen_chunks: list[tuple[str, str]] = []

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
                on_stream_chunk=lambda chunk, content: seen_chunks.append((chunk, content)),
            ),
        )

        with mock.patch("src.shared.ai_transport.httpx.Client", return_value=FakeClient()):
            content = transport.complete(request)

        self.assertEqual(content, "你好，世界")
        self.assertEqual(
            seen_chunks,
            [
                ("你好", "你好"),
                ("，世界", "你好，世界"),
            ],
        )

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
        self.assertEqual(openai_ctor.call_args.kwargs["max_retries"], 0)

    def test_ai_vision_service_dispatches_supported_provider_through_shared_transport(self) -> None:
        from src.interfaces.vision_interface import call_ai_vision_ocr_service

        with mock.patch(
            "src.interfaces.vision_interface.provider_supports_capability",
            side_effect=lambda provider, capability: provider == "siliconflow" and capability == "vision_ocr",
        ), mock.patch(
            "src.interfaces.vision_interface._transport.complete_vision",
            return_value='{"extracted_text":"测试"}',
        ) as complete_mock:
            content = call_ai_vision_ocr_service(
                Image.new("RGB", (12, 12), color="white"),
                provider="siliconflow",
                api_key="vision-key",
                model_name="vision-model",
                prompt="识别图片里的文本",
                prompt_mode="normal",
                openai_options=OpenAICompatibleOptions(
                    request=OpenAICompatibleRequestOptions(force_json_output=True),
                ),
            )

        self.assertEqual(content, '测试')
        request_arg = complete_mock.call_args.args[0]
        self.assertEqual(request_arg.provider, "siliconflow")
        self.assertIsNone(request_arg.base_url)
        self.assertTrue(request_arg.use_json_format)

    def test_ai_vision_service_accepts_current_paddleocr_vl_prompt_mode(self) -> None:
        from src.interfaces.vision_interface import call_ai_vision_ocr_service
        from src.shared.paddleocr_vl import build_paddleocr_vl_prompt

        with mock.patch(
            "src.interfaces.vision_interface._transport.complete_vision",
            return_value="测试",
        ), Image.new("RGB", (12, 12), color="white") as image:
            content = call_ai_vision_ocr_service(
                image,
                provider="ollama",
                model_name="vision-model",
                prompt=build_paddleocr_vl_prompt("japanese"),
                prompt_mode="paddleocr_vl",
            )

        self.assertEqual(content, "测试")

    def test_baidu_ocr_maps_current_english_source_language_code(self) -> None:
        from src.interfaces.baidu_ocr_interface import BaiduOCRInterface

        self.assertEqual(BaiduOCRInterface.LANGUAGE_MAPPING["en"], "ENG")
