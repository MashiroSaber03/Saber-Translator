import json
import logging
import re
import time

import requests

from src.shared import constants
from src.shared.ai_adapters import (
    translate_with_baidu,
    translate_with_caiyun,
    translate_with_youdao,
)
from src.shared.ai_providers import (
    TRANSLATION_CAPABILITY,
    get_provider_manifest,
    normalize_provider_id,
    provider_requires_api_key,
    provider_supports_capability,
)
from src.shared.ai_transport import (
    RETRYABLE_STATUS_CODES,
    OpenAICompatibleChatTransport,
    UnifiedChatRequest,
)
from src.shared.memory_errors import is_memory_allocation_error
from src.shared.openai_execution import (
    OpenAICompatibleBusinessRetryableError,
    OpenAICompatibleRuntimeOptions,
    OpenAICompatibleSyncExecutor,
    build_openai_compatible_runtime_options,
)
from src.shared.openai_options import (
    DEFAULT_OPENAI_COMPATIBLE_TRANSPORT_RETRIES,
    OpenAICompatibleOptions,
    create_openai_compatible_options,
)
from src.shared.openai_rate_limits import SharedRPMLimiter
from src.shared.user_logging import (
    log_model_input,
    log_model_request,
    log_model_response,
    log_retry,
)

logger = logging.getLogger("CoreTranslation")
_chat_transport = OpenAICompatibleChatTransport()
_sync_executor = OpenAICompatibleSyncExecutor(_chat_transport)


class TranslationParseException(OpenAICompatibleBusinessRetryableError):
    """批量翻译响应解析失败异常，触发重试"""


def _build_text_chat_messages(prompt_content: str, text: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if prompt_content:
        messages.append({"role": "system", "content": prompt_content})
    messages.append({"role": "user", "content": text})
    return messages


def _build_translation_openai_options(
    *,
    openai_options: OpenAICompatibleOptions | None = None,
    default_force_json_output: bool = False,
    default_rpm_limit: int = constants.DEFAULT_rpm_TRANSLATION,
    default_business_retries: int = constants.DEFAULT_TRANSLATION_MAX_RETRIES,
) -> OpenAICompatibleOptions:
    if openai_options is not None:
        if not isinstance(openai_options, OpenAICompatibleOptions):
            raise TypeError("openai_options 必须是 OpenAICompatibleOptions")
        return OpenAICompatibleOptions.from_dict(openai_options.to_dict())
    return create_openai_compatible_options(
        force_json_output=default_force_json_output,
        use_stream=False,
        rpm_limit=default_rpm_limit,
        transport_retries=DEFAULT_OPENAI_COMPATIBLE_TRANSPORT_RETRIES,
        business_retries=default_business_retries,
    )


def _build_translation_runtime_options(
    *,
    timeout: float,
    label: str,
) -> OpenAICompatibleRuntimeOptions:
    return build_openai_compatible_runtime_options(
        timeout=timeout,
        stream_output_label=label,
    )


def _parse_single_translation_response(content: str, *, use_json_format: bool) -> str:
    if not isinstance(content, str):
        raise OpenAICompatibleBusinessRetryableError("翻译响应必须是字符串")
    translated_text = content.strip()
    if use_json_format:
        try:
            payload = json.loads(translated_text)
        except json.JSONDecodeError as exc:
            raise OpenAICompatibleBusinessRetryableError(f"翻译 JSON 解析失败: {exc}") from exc
        if not isinstance(payload, dict) or set(payload) != {"translated_text"}:
            raise OpenAICompatibleBusinessRetryableError(
                '翻译 JSON 必须仅包含 "translated_text" 字段'
            )
        value = payload["translated_text"]
        if not isinstance(value, str):
            raise OpenAICompatibleBusinessRetryableError(
                '翻译 JSON 的 "translated_text" 必须是字符串'
            )
        translated_text = value.strip()
    if not translated_text:
        raise OpenAICompatibleBusinessRetryableError("AI 返回空翻译结果")
    return translated_text


def _is_retryable_adapter_error(error: Exception) -> bool:
    if isinstance(error, OpenAICompatibleBusinessRetryableError):
        return True
    if isinstance(error, (requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(error, requests.HTTPError):
        response = error.response
        return response is not None and response.status_code in RETRYABLE_STATUS_CODES
    return False


def _parse_batch_translation_response(
    response_text: str,
    *,
    texts: list[str],
    use_json_format: bool,
) -> list[str]:
    if not isinstance(texts, list) or not texts or any(
        not isinstance(text, str) or not text.strip()
        for text in texts
    ):
        raise ValueError("批量翻译解析器需要非空字符串列表")
    translations = (
        _parse_batch_json_response(response_text, len(texts))
        if use_json_format
        else _parse_batch_response(response_text, len(texts))
    )

    if len(translations) != len(texts):
        raise OpenAICompatibleBusinessRetryableError(
            f"翻译数量不匹配: 期望 {len(texts)}, 实际 {len(translations)}"
        )

    empty_count = sum(
        1
        for source, translated in zip(texts, translations)
        if source.strip() and not translated.strip()
    )
    if empty_count > 0:
        raise OpenAICompatibleBusinessRetryableError(f"检测到 {empty_count} 个空翻译")

    return translations


def translate_single_text(
    text,
    target_language,
    model_provider,
    api_key=None,
    model_name=None,
    prompt_content=None,
    custom_base_url=None,
    openai_options: OpenAICompatibleOptions | None = None,
    credential_version_id: str | None = None,
):
    """
    使用指定的大模型翻译单段文本。
    
    注意：此函数用于非 LLM 提供商（如百度翻译）和编辑模式的单气泡重翻译。
    批量翻译请使用 translate_text_list() 函数。

    Args:
        text (str): 需要翻译的原始文本。
        target_language (str): 目标语言代码 (例如 'zh')。
        model_provider (str): 模型提供商。
        api_key (str, optional): API 密钥 (对于非本地部署是必需的)。
        model_name (str, optional): 模型名称。
        prompt_content (str, optional): 自定义提示词。如果为 None，使用默认提示词。
        custom_base_url (str, optional): 用户自定义的 OpenAI 兼容 API 的 Base URL。
    Returns:
        str: 翻译后的文本。
    """
    if not isinstance(text, str):
        raise TypeError("待翻译文本必须是字符串")
    if not isinstance(target_language, str) or not target_language.strip():
        raise ValueError("目标语言必须是非空字符串")
    if not isinstance(model_provider, str) or not model_provider.strip():
        raise ValueError("翻译服务商必须是非空字符串")
    if prompt_content is not None and not isinstance(prompt_content, str):
        raise TypeError("翻译提示词必须是字符串或 null")
    if not text.strip():
        return ""

    effective_options = _build_translation_openai_options(
        openai_options=openai_options,
        default_force_json_output=False,
        default_rpm_limit=constants.DEFAULT_rpm_TRANSLATION,
        default_business_retries=constants.DEFAULT_TRANSLATION_MAX_RETRIES,
    )
    use_json_format = effective_options.request.force_json_output
    rpm_limit_translation = effective_options.execution.rpm_limit
    business_retries = effective_options.execution.business_retries

    if prompt_content is None:
        # 根据是否使用 JSON 格式选择默认提示词
        if use_json_format:
            prompt_content = constants.DEFAULT_TRANSLATE_JSON_PROMPT
        else:
            prompt_content = constants.DEFAULT_PROMPT
    elif use_json_format and '"translated_text"' not in prompt_content:
        logger.debug("翻译启用了 JSON 模式，但用户提示词未声明 translated_text 字段")


    canonical_provider = normalize_provider_id(model_provider)
    manifest = get_provider_manifest(canonical_provider)
    if not provider_supports_capability(canonical_provider, TRANSLATION_CAPABILITY):
        raise ValueError(f"{manifest.display_name}不支持翻译")
    logger.debug(
        "开始翻译文本（服务商=%s, rpm=%s, transport_retries=%s, business_retries=%s）",
        canonical_provider,
        rpm_limit_translation if rpm_limit_translation > 0 else "无",
        effective_options.execution.transport_retries,
        business_retries,
    )

    if manifest.kind in {"openai_compatible", "local"}:
        if provider_requires_api_key(canonical_provider, custom_base_url) and not api_key:
            raise ValueError(f"{manifest.display_name}需要 API Key")
        if manifest.requires_model and not model_name:
            raise ValueError(f"{manifest.display_name}需要模型名称")
        if manifest.requires_base_url and not custom_base_url:
            raise ValueError(f"{manifest.display_name}需要 Base URL")

        messages = _build_text_chat_messages(prompt_content, text)
        if canonical_provider == "sakura":
            messages = _build_text_chat_messages(
                "你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。",
                f"将下面的日文文本翻译成中文：{text}",
            )

        result = _sync_executor.execute(
            UnifiedChatRequest(
                provider=canonical_provider,
                api_key=api_key,
                model=model_name,
                credential_version_id=credential_version_id,
                base_url=custom_base_url or None,
                capability=TRANSLATION_CAPABILITY,
                openai_options=effective_options,
                runtime_options=_build_translation_runtime_options(
                    timeout=30.0,
                    label="普通翻译",
                ),
                messages=messages,
            ),
            capability=TRANSLATION_CAPABILITY,
            parser=lambda content: _parse_single_translation_response(
                content,
                use_json_format=use_json_format,
            ),
            logger_instance=logger,
        )
        translated_text = result.parsed
    else:
        translated_text = None
        last_error = None
        total_attempts = business_retries + 1
        log_model_input(
            "普通翻译",
            (
                f"目标语言：{target_language}",
                f"待翻译文本：\n{text}",
            ),
        )
        for attempt in range(total_attempts):
            log_model_request(
                provider=manifest.display_name,
                model=None,
                stream=False,
                attempt=attempt + 1,
                total_attempts=total_attempts,
            )
            try:
                SharedRPMLimiter(
                    rpm_limit_translation,
                    provider=model_provider,
                    credential_version_id=credential_version_id,
                ).wait_sync()

                if canonical_provider == "caiyun":
                    if not api_key:
                        raise ValueError("彩云小译需要 API Key")
                    translated_text = translate_with_caiyun(
                        text,
                        target_language,
                        api_key,
                    )

                elif canonical_provider == constants.BAIDU_TRANSLATE_ENGINE_ID:
                    if not api_key or (isinstance(api_key, str) and not api_key.strip()):
                        raise ValueError("百度翻译API需要appid")
                    if not model_name or (isinstance(model_name, str) and not model_name.strip()):
                        raise ValueError("百度翻译API需要appkey")
                    translated_text = translate_with_baidu(text, target_language, api_key, model_name)

                elif canonical_provider == constants.YOUDAO_TRANSLATE_ENGINE_ID:
                    if not api_key or (isinstance(api_key, str) and not api_key.strip()):
                        raise ValueError("有道翻译API需要AppKey")
                    if not model_name or (isinstance(model_name, str) and not model_name.strip()):
                        raise ValueError("有道翻译API需要AppSecret")
                    translated_text = translate_with_youdao(text, target_language, api_key, model_name)
                else:
                    raise ValueError(f"不支持的翻译服务提供商: {canonical_provider}")

                if not isinstance(translated_text, str):
                    raise OpenAICompatibleBusinessRetryableError(
                        "翻译服务返回值必须是字符串"
                    )
                translated_text = translated_text.strip()
                if not translated_text:
                    raise OpenAICompatibleBusinessRetryableError(
                        "翻译服务返回空结果"
                    )
                log_model_response("普通翻译", translated_text)
                break
            except Exception as error:
                if is_memory_allocation_error(error):
                    raise
                last_error = error
                logger.error(
                    "翻译失败（尝试 %s/%s，服务商=%s）: %s",
                    attempt + 1,
                    total_attempts,
                    canonical_provider,
                    error,
                    exc_info=True,
                )
                translated_text = None
                if not _is_retryable_adapter_error(error):
                    raise
                if attempt >= business_retries:
                    break
                log_retry(
                    "普通翻译",
                    attempt + 2,
                    total_attempts,
                    error,
                )
                time.sleep(1)

        if translated_text is None:
            if last_error is None:
                raise RuntimeError("翻译失败且未提供错误原因")
            raise last_error

    logger.debug("文本翻译成功")

    return translated_text


def _assemble_batch_prompt(
    texts: list[str],
    custom_prompt: str | None = None,
    use_json_format: bool = False,
) -> tuple[list[dict[str, str]], int]:
    """
    将多个文本组装成批量翻译的 prompt
    
    Args:
        texts: 待翻译的文本列表
        custom_prompt: 自定义提示词 (如果为 None，使用默认批量翻译模板)
        use_json_format: 是否使用 JSON 输出格式
        
    Returns:
        tuple: (messages_list, batch_size) - 消息列表和批次大小
    """
    # 构建消息列表
    messages: list[dict[str, str]] = []

    if use_json_format:
        # --- JSON 模式 ---
        # 1. System prompt
        if custom_prompt:
            system_prompt = custom_prompt
        else:
            system_prompt = constants.BATCH_TRANSLATE_JSON_SYSTEM_TEMPLATE
        messages.append({"role": "system", "content": system_prompt})
        # 2. Few-shot learning: JSON 格式示例
        messages.append({"role": "user", "content": constants.BATCH_TRANSLATE_JSON_SAMPLE_INPUT})
        messages.append({"role": "assistant", "content": constants.BATCH_TRANSLATE_JSON_SAMPLE_OUTPUT})
        logger.debug("已添加 JSON 模式翻译示例")
        # 3. User prompt：构建 JSON 格式的输入
        texts_json = {
            "texts": [
                {"id": index + 1, "text": text}
                for index, text in enumerate(texts)
            ]
        }
        user_prompt = (
            constants.BATCH_TRANSLATE_JSON_USER_TEMPLATE
            + "\n"
            + json.dumps(texts_json, ensure_ascii=False, indent=2)
        )
        messages.append({"role": "user", "content": user_prompt})
    else:
        # --- 纯文本模式 (默认) ---
        # 1. System prompt
        if custom_prompt:
            system_prompt = custom_prompt
        else:
            system_prompt = constants.BATCH_TRANSLATE_SYSTEM_TEMPLATE
        messages.append({"role": "system", "content": system_prompt})
        # 2. Few-shot learning: 添加翻译示例
        messages.append({"role": "user", "content": constants.BATCH_TRANSLATE_SAMPLE_INPUT})
        messages.append({"role": "assistant", "content": constants.BATCH_TRANSLATE_SAMPLE_OUTPUT})
        logger.debug("已添加翻译示例")
        # 3. User prompt：将所有文本编号并合并
        user_prompt = constants.BATCH_TRANSLATE_USER_TEMPLATE
        for index, text in enumerate(texts):
            user_prompt += f"\n<|{index + 1}|>{text}"
        messages.append({"role": "user", "content": user_prompt})

    return messages, len(texts)


def _parse_batch_response(response_text: str, expected_count: int) -> list[str]:
    """按 <|n|> 协议严格解析批量翻译响应。"""
    if not isinstance(response_text, str):
        raise TranslationParseException("批量翻译响应必须是字符串")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 1:
        raise ValueError("批量翻译期望数量必须是正整数")

    cleaned_text = re.sub(
        r"<think>.*?</think>",
        "",
        response_text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    # 某些服务商会把行首 <|n|> 简化为 <n>；该变体仍是无歧义的同一协议。
    cleaned_text = re.sub(
        r"(?m)^(\s*)<(\d+)>",
        lambda match: f"{match.group(1)}<|{match.group(2)}|>",
        cleaned_text,
    )
    markers = list(re.finditer(r"<\|(\d+)\|>", cleaned_text))
    if not markers:
        raise TranslationParseException(
            "无法在响应中找到批量翻译的编号格式 <|n|>"
        )
    if cleaned_text[:markers[0].start()].strip():
        raise TranslationParseException("批量翻译响应在 <|1|> 前包含额外内容")

    actual_ids = [int(marker.group(1)) for marker in markers]
    expected_ids = list(range(1, expected_count + 1))
    if actual_ids != expected_ids:
        raise TranslationParseException(
            f"翻译数量不匹配或编号错误: 期望 {expected_ids}，实际 {actual_ids}"
        )

    translations: list[str] = []
    for index, marker in enumerate(markers):
        content_end = markers[index + 1].start() if index + 1 < len(markers) else len(cleaned_text)
        translated = cleaned_text[marker.end() : content_end].strip()
        if not translated:
            raise TranslationParseException(f"第 {index + 1} 条翻译为空")
        translations.append(translated)
    return translations

def _parse_batch_json_response(response_text: str, expected_count: int) -> list[str]:
    """按当前 translations JSON 协议严格解析批量翻译响应。"""
    if not isinstance(response_text, str):
        raise TranslationParseException("批量翻译 JSON 响应必须是字符串")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 1:
        raise ValueError("批量翻译期望数量必须是正整数")

    cleaned_text = re.sub(
        r"<think>.*?</think>",
        "",
        response_text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned_text)
    if fenced:
        cleaned_text = fenced.group(1).strip()
    try:
        data = json.loads(cleaned_text)
    except json.JSONDecodeError as exc:
        raise TranslationParseException(f"JSON 解析失败: {exc}") from exc

    if not isinstance(data, dict) or set(data) != {"translations"}:
        raise TranslationParseException(
            '批量翻译 JSON 必须仅包含 "translations" 字段'
        )
    items = data["translations"]
    if not isinstance(items, list):
        raise TranslationParseException('"translations" 必须是列表')
    if len(items) != expected_count:
        raise TranslationParseException(
            f"翻译数量不匹配: 期望 {expected_count}, 实际 {len(items)}"
        )

    translations: list[str] = []
    for expected_id, item in enumerate(items, start=1):
        if not isinstance(item, dict) or set(item) != {"id", "text"}:
            raise TranslationParseException(
                f"第 {expected_id} 个翻译条目必须仅包含 id 和 text"
            )
        item_id = item["id"]
        if isinstance(item_id, bool) or not isinstance(item_id, int) or item_id != expected_id:
            raise TranslationParseException(
                f"第 {expected_id} 个翻译条目的 id 必须为 {expected_id}"
            )
        item_text = item["text"]
        if not isinstance(item_text, str) or not item_text.strip():
            raise TranslationParseException(
                f"第 {expected_id} 个翻译条目的 text 必须是非空字符串"
            )
        translations.append(item_text.strip())
    return translations


def _translate_batch_with_llm(
    texts: list[str],
    model_provider: str,
    api_key: str | None,
    model_name: str | None,
    custom_prompt: str | None = None,
    custom_base_url: str | None = None,
    openai_options: OpenAICompatibleOptions | None = None,
    credential_version_id: str | None = None,
) -> list[str]:
    """
    使用 LLM 进行批量翻译
    
    Args:
        texts: 待翻译的文本列表
        model_provider: 模型提供商
        api_key: API 密钥
        model_name: 模型名称
        custom_prompt: 自定义提示词
        custom_base_url: 自定义 API Base URL
    Returns:
        list: 翻译结果列表
    """
    if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
        raise TypeError("批量翻译文本必须是字符串列表")
    if not isinstance(model_provider, str) or not model_provider.strip():
        raise ValueError("翻译服务商必须是非空字符串")
    if custom_prompt is not None and not isinstance(custom_prompt, str):
        raise TypeError("批量翻译提示词必须是字符串或 null")
    if not texts:
        return []

    effective_options = _build_translation_openai_options(
        openai_options=openai_options,
        default_force_json_output=False,
        default_rpm_limit=0,
        default_business_retries=2,
    )
    use_json_format = effective_options.request.force_json_output
    # 组装消息列表 (包含 system prompt、few-shot 示例、user prompt)
    messages, batch_size = _assemble_batch_prompt(texts, custom_prompt, use_json_format)
    
    logger.debug("批量翻译请求：%s 个文本片段（消息数=%s）", batch_size, len(messages))
    
    canonical_provider = normalize_provider_id(model_provider)
    manifest = get_provider_manifest(canonical_provider)
    if not provider_supports_capability(canonical_provider, TRANSLATION_CAPABILITY):
        raise ValueError(f"{manifest.display_name}不支持翻译")
    if manifest.kind not in {"openai_compatible", "local"}:
        raise ValueError(f"不支持批量翻译的服务商: {canonical_provider}")
    if provider_requires_api_key(canonical_provider, custom_base_url) and not api_key:
        raise ValueError(f"{manifest.display_name}需要 API Key")
    if manifest.requires_model and not model_name:
        raise ValueError(f"{manifest.display_name}需要模型名称")
    if manifest.requires_base_url and not custom_base_url:
        raise ValueError(f"{manifest.display_name}需要 Base URL")

    result = _sync_executor.execute(
        UnifiedChatRequest(
            provider=canonical_provider,
            api_key=api_key,
            model=model_name,
            credential_version_id=credential_version_id,
            messages=messages,
            base_url=custom_base_url or None,
            capability=TRANSLATION_CAPABILITY,
            openai_options=effective_options,
            runtime_options=_build_translation_runtime_options(
                timeout=120.0,
                label="普通翻译",
            ),
        ),
        capability=TRANSLATION_CAPABILITY,
        parser=lambda content: _parse_batch_translation_response(
            content,
            texts=texts,
            use_json_format=use_json_format,
        ),
        logger_instance=logger,
    )
    logger.debug("批量翻译成功：%s 个文本片段", len(texts))
    return result.parsed


def translate_text_list(
    texts,
    target_language,
    model_provider,
    api_key=None,
    model_name=None,
    prompt_content=None,
    custom_base_url=None,
    openai_options: OpenAICompatibleOptions | None = None,
    credential_version_id: str | None = None,
):
    """
    翻译文本列表 - 使用批量翻译策略
    
    将一页内所有气泡的文本合并为一个请求发送给 LLM，使用 <|n|> 格式编号，
    一次 API 调用翻译整页内容，大幅提升效率和翻译一致性。
    
    注意：目标语言现在由提示词控制（默认翻译为中文），如需修改请编辑 
    constants.BATCH_TRANSLATE_SYSTEM_TEMPLATE 中的提示词。

    Args:
        texts (list): 包含待翻译文本字符串的列表。
        target_language (str): [已弃用] 目标语言代码，现由提示词控制。
        model_provider (str): 模型提供商。
        api_key (str, optional): API 密钥。
        model_name (str, optional): 模型名称。
        prompt_content (str, optional): 自定义提示词，可覆盖默认提示词。
        custom_base_url (str, optional): 用户自定义的 OpenAI 兼容 API 的 Base URL。
    Returns:
        list: 包含翻译后文本的列表，顺序与输入列表一致。
    """
    if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
        raise TypeError("待翻译内容必须是字符串列表")
    if not isinstance(target_language, str) or not target_language.strip():
        raise ValueError("目标语言必须是非空字符串")
    if not isinstance(model_provider, str) or not model_provider.strip():
        raise ValueError("翻译服务商必须是非空字符串")
    if prompt_content is not None and not isinstance(prompt_content, str):
        raise TypeError("翻译提示词必须是字符串或 null")
    if not texts:
        return []
    
    # 过滤空文本，记录索引
    non_empty_indices = []
    non_empty_texts = []
    final_translations = [""] * len(texts)
    
    for i, text in enumerate(texts):
        if text and text.strip():
            non_empty_indices.append(i)
            non_empty_texts.append(text)
        else:
            final_translations[i] = ""
    
    if not non_empty_texts:
        return final_translations
    
    effective_options = _build_translation_openai_options(
        openai_options=openai_options,
        default_force_json_output=False,
        default_rpm_limit=constants.DEFAULT_rpm_TRANSLATION,
        default_business_retries=constants.DEFAULT_TRANSLATION_MAX_RETRIES,
    )
    rpm_limit_translation = effective_options.execution.rpm_limit

    canonical_provider = normalize_provider_id(model_provider)
    manifest = get_provider_manifest(canonical_provider)
    if not provider_supports_capability(canonical_provider, TRANSLATION_CAPABILITY):
        raise ValueError(f"{manifest.display_name}不支持翻译")
    logger.debug(
        "开始批量翻译 %s 个文本片段（服务商=%s, rpm=%s）",
        len(non_empty_texts),
        canonical_provider,
        rpm_limit_translation if rpm_limit_translation > 0 else "无",
    )

    supports_batch_translation = manifest.kind != "adapter"

    if supports_batch_translation:
        all_translations = _translate_batch_with_llm(
            non_empty_texts,
            canonical_provider,
            api_key,
            model_name,
            custom_prompt=prompt_content,
            custom_base_url=custom_base_url,
            openai_options=effective_options,
            credential_version_id=credential_version_id,
        )
        if len(all_translations) != len(non_empty_indices):
            raise RuntimeError(
                "批量翻译结果数量不匹配: "
                f"expected={len(non_empty_indices)}, actual={len(all_translations)}"
            )
        
        # 将翻译结果写回最终列表
        for index, translated in enumerate(all_translations):
            final_translations[non_empty_indices[index]] = translated

    else:
        # 非 LLM 提供商 (如百度翻译、有道翻译)，使用原有的逐个翻译逻辑
        logger.debug("服务商 %s 使用逐条翻译", canonical_provider)
        for index, text in enumerate(non_empty_texts):
            translated = translate_single_text(
                text,
                target_language,
                canonical_provider,
                api_key=api_key,
                model_name=model_name,
                prompt_content=prompt_content,
                custom_base_url=custom_base_url,
                openai_options=effective_options,
                credential_version_id=credential_version_id,
            )
            final_translations[non_empty_indices[index]] = translated

    completed_count = sum(1 for translated in final_translations if translated)
    logger.debug("批量翻译完成：成功 %s/%s", completed_count, len(texts))
    return final_translations
