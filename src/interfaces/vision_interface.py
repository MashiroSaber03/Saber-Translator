"""
AI视觉OCR服务接口模块：用于调用不同服务商的视觉API进行OCR识别
"""

import json
import logging
import time

from src.shared import constants
from src.shared.ai_providers import (
    VISION_OCR_CAPABILITY,
    get_provider_manifest,
    normalize_provider_id,
    provider_supports_capability,
    resolve_provider_base_url,
)
from src.shared.ai_transport import OpenAICompatibleChatTransport, UnifiedVisionRequest
from src.shared.openai_execution import (
    OpenAICompatibleBusinessRetryableError,
    OpenAICompatibleSyncExecutor,
    build_openai_compatible_runtime_options,
)
from src.shared.openai_options import (
    DEFAULT_OPENAI_COMPATIBLE_TRANSPORT_RETRIES,
    OpenAICompatibleOptions,
    create_openai_compatible_options,
)
from src.shared.image_helpers import image_to_base64
from src.shared.memory_errors import is_memory_allocation_error

# 设置日志
logger = logging.getLogger("VisionInterface")
_transport = OpenAICompatibleChatTransport()
_sync_executor = OpenAICompatibleSyncExecutor(_transport)


def _parse_ai_vision_ocr_response(content: str, *, use_json_format: bool) -> str:
    text = content.strip()
    if use_json_format:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise OpenAICompatibleBusinessRetryableError(f"AI视觉OCR JSON 解析失败: {exc}") from exc
        text = str(payload.get("extracted_text") or "").strip()
    if not text:
        raise OpenAICompatibleBusinessRetryableError("AI视觉OCR 返回空结果")
    return text


def call_ai_vision_ocr_service(image_pil, provider='siliconflow', api_key=None, model_name=None, prompt=None,
                               prompt_mode: str = 'normal',
                               custom_base_url=None,
                               openai_options: OpenAICompatibleOptions | None = None,
                               credential_version_id: str | None = None):
    if not image_pil:
        logger.error("未提供有效图像")
        return ""

    start_time = time.time()
    try:
        image_base64 = image_to_base64(image_pil)
    except Exception as e:
        logger.error(f"图像转Base64失败: {e}")
        if is_memory_allocation_error(e):
            raise
        return ""

    try:
        provider_lower = normalize_provider_id(provider)
        manifest = get_provider_manifest(provider_lower)
        if not provider_supports_capability(provider_lower, VISION_OCR_CAPABILITY):
            logger.error(f"不支持的AI视觉OCR服务提供商: {provider}")
            return ""
        if manifest.requires_api_key and not api_key:
            logger.error(f"未提供 {provider} 的API密钥")
            return ""
        if manifest.requires_model and not model_name:
            logger.error(f"未提供 {provider} 的模型名称")
            return ""

        resolved_base_url = resolve_provider_base_url(provider_lower, custom_base_url)
        if not resolved_base_url:
            logger.error(f"未提供 {provider_lower} 的 Base URL")
            return ""

        effective_options = openai_options or create_openai_compatible_options(
            force_json_output=False,
            use_stream=False,
            rpm_limit=0,
            transport_retries=DEFAULT_OPENAI_COMPATIBLE_TRANSPORT_RETRIES,
            business_retries=0,
        )
        if not prompt:
            if effective_options.request.force_json_output or (prompt_mode or "").strip().lower() == "json":
                prompt = constants.DEFAULT_AI_VISION_OCR_JSON_PROMPT
            else:
                prompt = constants.DEFAULT_AI_VISION_OCR_PROMPT
            logger.info("使用默认AI视觉OCR提示词")

        logger.info(
            "[AI视觉OCR-请求] provider=%s, model=%s, prompt_mode=%s, json_mode=%s, base_url=%s",
            provider_lower,
            model_name,
            prompt_mode,
            effective_options.request.force_json_output,
            resolved_base_url,
        )
        logger.info("[AI视觉OCR-请求] 实际提示词开始\n%s\n[AI视觉OCR-请求] 实际提示词结束", prompt)
        use_json_format = effective_options.request.force_json_output
        result = _sync_executor.execute(
            UnifiedVisionRequest(
                provider=provider_lower,
                api_key=api_key,
                model=model_name,
                credential_version_id=credential_version_id,
                prompt=prompt,
                image_base64=image_base64,
                capability=VISION_OCR_CAPABILITY,
                base_url=custom_base_url if provider_lower == 'custom' else None,
                openai_options=effective_options,
                runtime_options=build_openai_compatible_runtime_options(
                    timeout=120.0,
                    print_stream_output=effective_options.execution.use_stream,
                    stream_output_label="AI视觉OCR",
                ),
            ),
            capability=VISION_OCR_CAPABILITY,
            parser=lambda content: _parse_ai_vision_ocr_response(
                content,
                use_json_format=use_json_format,
            ),
            logger_instance=logger,
        )
        content = str(result.parsed or "").strip()

        elapsed_time = time.time() - start_time
        logger.info(f"{provider_lower} 视觉OCR识别成功，耗时: {elapsed_time:.2f}秒")
        logger.info(f"识别结果 (前100字符): {content[:100]}")
        return content
    except Exception as e:
        logger.error(f"调用AI视觉OCR服务 ({provider}) 时发生顶层异常: {e}", exc_info=True)
        if is_memory_allocation_error(e):
            raise
        return ""
