from __future__ import annotations

from unittest import mock

from PIL import Image
import pytest
import requests

from src.core import translation
from src.interfaces.baidu_ocr_interface import BaiduOCRInterface
from src.interfaces.baidu_translate_interface import BaiduTranslateInterface
from src.interfaces.ocr_48px.interface import Model48pxOCR
from src.interfaces.youdao_translate_interface import YoudaoTranslateInterface
from src.interfaces.paddle_ocr_onnx_interface import PaddleOCRHandlerONNX
from src.interfaces.paddleocr_vl_interface import PaddleOCRVLHandler
from src.shared import ai_adapters, constants
from src.shared.ai_providers import TRANSLATION_CAPABILITY
from src.shared.ai_transport import UnifiedChatRequest
from src.shared.openai_execution import (
    OpenAICompatibleBusinessRetryableError,
    OpenAICompatibleSyncExecutor,
)
from src.shared.openai_options import create_openai_compatible_options


def test_batch_parser_rejects_missing_translation_items() -> None:
    with pytest.raises(
        OpenAICompatibleBusinessRetryableError,
        match="翻译数量不匹配",
    ):
        translation._parse_batch_translation_response(
            "<|1|>第一条",
            texts=["一", "二"],
            use_json_format=False,
        )


def test_paddle_ocr_strict_mode_propagates_per_bubble_failure() -> None:
    handler = PaddleOCRHandlerONNX()
    handler.initialized = True
    handler.ocr = mock.Mock(side_effect=RuntimeError("rapidocr failed"))
    image = Image.new("RGB", (16, 16), "white")
    try:
        with pytest.raises(RuntimeError, match="rapidocr failed"):
            handler.recognize_text_with_details(
                image,
                [(0, 0, 16, 16)],
                [[{
                    "polygon": [[1, 1], [15, 1], [15, 15], [1, 15]],
                    "direction": "h",
                }]],
            )
    finally:
        image.close()


def test_48px_ocr_strict_mode_propagates_model_failure() -> None:
    handler = Model48pxOCR()
    handler.initialized = True
    handler.device = "cpu"
    handler.model = mock.Mock()
    handler.model.infer_beam_batch_tensor.side_effect = RuntimeError(
        "48px inference failed"
    )
    image = Image.new("RGB", (16, 16), "white")
    try:
        with pytest.raises(RuntimeError, match="48px inference failed"):
            handler.recognize_text_with_details(
                image,
                [(0, 0, 16, 16)],
            )
    finally:
        image.close()


def test_paddleocr_vl_strict_mode_propagates_per_bubble_failure() -> None:
    handler = PaddleOCRVLHandler()
    handler.initialized = True
    handler.model = mock.Mock()
    handler._recognize_single = mock.Mock(
        side_effect=RuntimeError("paddleocr-vl failed")
    )
    image = Image.new("RGB", (16, 16), "white")
    try:
        with pytest.raises(RuntimeError, match="paddleocr-vl failed"):
            handler.recognize_text(
                image,
                [(0, 0, 16, 16)],
                "japanese",
            )
    finally:
        image.close()


def test_baidu_ocr_provider_error_is_not_reported_as_empty_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = mock.Mock()
    response.json.return_value = {
        "error_code": 216100,
        "error_msg": "invalid language",
    }
    response.raise_for_status.return_value = None
    monkeypatch.setattr(
        "src.interfaces.baidu_ocr_interface.requests.post",
        mock.Mock(return_value=response),
    )
    interface = BaiduOCRInterface("api-key", "secret-key")
    interface.access_token = "token"

    with pytest.raises(RuntimeError, match="216100"):
        interface.recognize_text(b"image")


def test_ai_vision_provider_failure_is_not_reported_as_empty_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.interfaces.vision_interface import call_ai_vision_ocr_service

    monkeypatch.setattr(
        "src.interfaces.vision_interface._transport.complete_vision",
        mock.Mock(side_effect=RuntimeError("vision provider failed")),
    )
    image = Image.new("RGB", (16, 16), "white")
    try:
        with pytest.raises(RuntimeError, match="vision provider failed"):
            call_ai_vision_ocr_service(
                image,
                provider="custom",
                api_key="key",
                model_name="model",
                prompt="OCR",
                custom_base_url="https://example.com/v1",
                openai_options=create_openai_compatible_options(
                    business_retries=0,
                    transport_retries=0,
                ),
            )
    finally:
        image.close()


def test_batch_parser_accepts_unambiguous_short_number_prefixes() -> None:
    assert translation._parse_batch_translation_response(
        "<1>第一条\n<2>第二条",
        texts=["一", "二"],
        use_json_format=False,
    ) == ["第一条", "第二条"]


@pytest.mark.parametrize(
    "response",
    [
        "<|1|>第一条\n<|1|>重复编号",
        "<|2|>第二条\n<|1|>第一条",
        "<|1|>第一条\n<|3|>越界编号",
    ],
)
def test_batch_parser_rejects_non_sequential_ids(response: str) -> None:
    with pytest.raises(OpenAICompatibleBusinessRetryableError):
        translation._parse_batch_translation_response(
            response,
            texts=["一", "二"],
            use_json_format=False,
        )


def test_single_item_batch_parser_does_not_merge_multiple_ids() -> None:
    with pytest.raises(OpenAICompatibleBusinessRetryableError, match="编号错误"):
        translation._parse_batch_translation_response(
            "<|1|>第一段\n<|2|>第二段",
            texts=["一"],
            use_json_format=False,
        )


@pytest.mark.parametrize(
    "response",
    [
        '[{"id": 1, "text": "译文"}]',
        '{"translations": [{"id": 2, "text": "二"}, {"id": 1, "text": "一"}]}',
        '{"translations": [{"id": 1, "text": 7}]}',
        '{"translations": [{"id": 1, "text": "译文", "legacy": true}]}',
    ],
)
def test_batch_json_parser_rejects_malformed_contract(response: str) -> None:
    texts = ["一", "二"] if '"id": 2' in response else ["一"]
    with pytest.raises(OpenAICompatibleBusinessRetryableError):
        translation._parse_batch_translation_response(
            response,
            texts=texts,
            use_json_format=True,
        )


def test_single_json_parser_rejects_non_string_translation() -> None:
    with pytest.raises(OpenAICompatibleBusinessRetryableError, match="必须是字符串"):
        translation._parse_single_translation_response(
            '{"translated_text": 7}',
            use_json_format=True,
        )


def test_batch_parser_protocol_failure_is_business_retryable() -> None:
    with pytest.raises(
        OpenAICompatibleBusinessRetryableError,
        match=r"编号格式 <\|n\|>",
    ):
        translation._parse_batch_translation_response(
            "第一条\n第二条",
            texts=["一", "二"],
            use_json_format=False,
        )


def test_adapter_translation_failure_propagates_after_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        translation,
        "translate_with_baidu",
        mock.Mock(side_effect=RuntimeError("provider failed")),
    )
    options = create_openai_compatible_options(business_retries=0)

    with pytest.raises(RuntimeError, match="provider failed"):
        translation.translate_single_text(
            "原文",
            "zh",
            constants.BAIDU_TRANSLATE_ENGINE_ID,
            api_key="appid",
            model_name="appkey",
            openai_options=options,
        )


def test_adapter_memory_failure_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    translate = mock.Mock(side_effect=MemoryError("native allocation failed"))
    monkeypatch.setattr(translation, "translate_with_baidu", translate)
    options = create_openai_compatible_options(business_retries=3)

    with pytest.raises(MemoryError, match="allocation failed"):
        translation.translate_single_text(
            "原文",
            "zh",
            constants.BAIDU_TRANSLATE_ENGINE_ID,
            api_key="appid",
            model_name="appkey",
            openai_options=options,
        )
    assert translate.call_count == 1


def test_adapter_non_string_result_is_not_reported_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        translation,
        "translate_with_baidu",
        mock.Mock(return_value={"translated": "译文"}),
    )
    with pytest.raises(OpenAICompatibleBusinessRetryableError, match="必须是字符串"):
        translation.translate_single_text(
            "原文",
            "zh",
            constants.BAIDU_TRANSLATE_ENGINE_ID,
            api_key="appid",
            model_name="appkey",
            openai_options=create_openai_compatible_options(business_retries=0),
        )


def test_shared_business_retry_does_not_hide_nested_memory_failure() -> None:
    class FakeTransport:
        calls = 0

        def complete(self, _request, **_kwargs):
            self.calls += 1
            return "response"

    transport = FakeTransport()
    executor = OpenAICompatibleSyncExecutor(transport)

    def parse(_content: str):
        try:
            raise MemoryError("native allocation failed")
        except MemoryError as cause:
            raise OpenAICompatibleBusinessRetryableError(
                "解析失败"
            ) from cause

    request = UnifiedChatRequest(
        provider="deepseek",
        api_key="key",
        model="model",
        messages=[{"role": "user", "content": "hello"}],
        openai_options=create_openai_compatible_options(business_retries=3),
    )
    with pytest.raises(OpenAICompatibleBusinessRetryableError, match="解析失败"):
        executor.execute(
            request,
            capability=TRANSLATION_CAPABILITY,
            parser=parse,
        )
    assert transport.calls == 1


def test_youdao_provider_error_is_not_reported_as_original_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = mock.Mock()
    response.json.return_value = {"errorCode": "108"}
    monkeypatch.setattr(
        "src.interfaces.youdao_translate_interface.requests.post",
        mock.Mock(return_value=response),
    )
    interface = YoudaoTranslateInterface("app-key", "app-secret")

    with pytest.raises(RuntimeError, match="108"):
        interface.translate("原文")


def test_baidu_translation_uses_one_https_request_with_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = mock.Mock()
    response.json.return_value = {"trans_result": [{"dst": "译文"}]}
    post = mock.Mock(return_value=response)
    monkeypatch.setattr(
        "src.interfaces.baidu_translate_interface.requests.post",
        post,
    )

    assert BaiduTranslateInterface("appid", "appkey").translate("原文") == "译文"
    assert post.call_count == 1
    _, kwargs = post.call_args
    assert post.call_args.args[0].startswith("https://")
    assert kwargs["timeout"] == 30.0
    assert "data" in kwargs
    assert "params" not in kwargs
    response.raise_for_status.assert_called_once_with()


def test_youdao_translation_uses_form_request_with_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = mock.Mock()
    response.json.return_value = {"errorCode": "0", "translation": ["译文"]}
    post = mock.Mock(return_value=response)
    monkeypatch.setattr(
        "src.interfaces.youdao_translate_interface.requests.post",
        post,
    )

    assert YoudaoTranslateInterface("key", "secret").translate("原文") == "译文"
    _, kwargs = post.call_args
    assert post.call_args.args[0].startswith("https://")
    assert kwargs["timeout"] == 30.0
    assert "data" in kwargs
    assert "params" not in kwargs
    response.raise_for_status.assert_called_once_with()


def test_baidu_adapter_credentials_are_isolated_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = mock.Mock()
    first.translate.return_value = "译文1"
    second = mock.Mock()
    second.translate.return_value = "译文2"
    factory = mock.Mock(side_effect=[first, second])
    monkeypatch.setattr(ai_adapters, "BaiduTranslateInterface", factory)

    assert ai_adapters.translate_with_baidu("原文1", "zh", "id1", "key1") == "译文1"
    assert ai_adapters.translate_with_baidu("原文2", "zh", "id2", "key2") == "译文2"
    assert factory.call_args_list == [mock.call("id1", "key1"), mock.call("id2", "key2")]


def test_adapter_rejects_unknown_target_language() -> None:
    with pytest.raises(ValueError, match="不支持目标语言"):
        ai_adapters.translate_with_baidu("原文", "unknown", "id", "key")


def test_sakura_batch_failure_propagates_after_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execute = mock.Mock(side_effect=RuntimeError("sakura failed"))
    monkeypatch.setattr(translation._sync_executor, "execute", execute)
    options = create_openai_compatible_options(business_retries=0)

    with pytest.raises(RuntimeError, match="sakura failed"):
        translation._translate_batch_with_llm(
            ["原文"],
            "sakura",
            "",
            "local-model",
            openai_options=options,
        )
    assert execute.call_count == 1


def test_sakura_memory_failure_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execute = mock.Mock(side_effect=MemoryError("native allocation failed"))
    monkeypatch.setattr(translation._sync_executor, "execute", execute)
    options = create_openai_compatible_options(business_retries=3)

    with pytest.raises(MemoryError, match="allocation failed"):
        translation._translate_batch_with_llm(
            ["原文"],
            "sakura",
            "",
            "local-model",
            openai_options=options,
        )
    assert execute.call_count == 1


def test_single_translation_uses_provider_default_for_empty_custom_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execute = mock.Mock(return_value=mock.Mock(parsed="译文"))
    monkeypatch.setattr(translation._sync_executor, "execute", execute)

    result = translation.translate_single_text(
        "原文",
        "zh",
        "siliconflow",
        api_key="key",
        model_name="model",
        custom_base_url="",
    )

    assert result == "译文"
    assert execute.call_args.args[0].base_url is None


def test_batch_translation_uses_provider_default_for_empty_custom_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execute = mock.Mock(return_value=mock.Mock(parsed=["译文"]))
    monkeypatch.setattr(translation._sync_executor, "execute", execute)

    result = translation._translate_batch_with_llm(
        ["原文"],
        "siliconflow",
        "key",
        "model",
        custom_base_url="",
    )

    assert result == ["译文"]
    assert execute.call_args.args[0].base_url is None


def test_translation_list_rejects_incomplete_batch_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        translation,
        "_translate_batch_with_llm",
        mock.Mock(return_value=[]),
    )

    with pytest.raises(RuntimeError, match="批量翻译结果数量不匹配"):
        translation.translate_text_list(
            ["原文"],
            "zh",
            "deepseek",
            api_key="key",
            model_name="model",
        )


def test_translation_list_sends_one_page_without_arbitrary_character_splitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    translate_batch = mock.Mock(return_value=["长译文", "短译文"])
    monkeypatch.setattr(translation, "_translate_batch_with_llm", translate_batch)

    result = translation.translate_text_list(
        ["原" * 5000, "短原文"],
        "zh",
        "deepseek",
        api_key="key",
        model_name="model",
    )

    assert result == ["长译文", "短译文"]
    assert translate_batch.call_count == 1
    assert translate_batch.call_args.args[0] == ["原" * 5000, "短原文"]


def test_adapter_authentication_failure_is_not_business_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = requests.Response()
    response.status_code = 401
    error = requests.HTTPError("unauthorized", response=response)
    translate = mock.Mock(side_effect=error)
    monkeypatch.setattr(translation, "translate_with_caiyun", translate)

    with pytest.raises(requests.HTTPError, match="unauthorized"):
        translation.translate_single_text(
            "原文",
            "zh",
            "caiyun",
            api_key="key",
            openai_options=create_openai_compatible_options(business_retries=3),
        )

    assert translate.call_count == 1


def test_adapter_transient_server_failure_uses_finite_business_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = requests.Response()
    response.status_code = 503
    error = requests.HTTPError("unavailable", response=response)
    translate = mock.Mock(side_effect=[error, "译文"])
    monkeypatch.setattr(translation, "translate_with_caiyun", translate)
    monkeypatch.setattr(translation.time, "sleep", mock.Mock())

    result = translation.translate_single_text(
        "原文",
        "zh",
        "caiyun",
        api_key="key",
        openai_options=create_openai_compatible_options(business_retries=1),
    )

    assert result == "译文"
    assert translate.call_count == 2
