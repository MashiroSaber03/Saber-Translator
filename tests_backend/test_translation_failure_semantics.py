from __future__ import annotations

from unittest import mock

import pytest

from src.core import translation
from src.shared import constants
from src.shared.openai_execution import OpenAICompatibleBusinessRetryableError
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


def test_batch_parser_accepts_unambiguous_short_number_prefixes() -> None:
    assert translation._parse_batch_translation_response(
        "<1>第一条\n<2>第二条",
        texts=["一", "二"],
        use_json_format=False,
    ) == ["第一条", "第二条"]


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


def test_sakura_batch_failure_propagates_after_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        translation,
        "run_local_chat_completion",
        mock.Mock(side_effect=RuntimeError("sakura failed")),
    )
    options = create_openai_compatible_options(business_retries=0)

    with pytest.raises(RuntimeError, match="sakura failed"):
        translation._translate_batch_with_llm(
            ["原文"],
            "sakura",
            "",
            "local-model",
            openai_options=options,
        )


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
