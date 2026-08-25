"""
Manga Insight Embedding / Chat clients backed by shared async transport.
"""

import asyncio
import logging
from typing import Any

from src.shared.ai_transport import (
    AsyncOpenAICompatibleTransport,
    UnifiedChatRequest,
    UnifiedEmbeddingRequest,
)
from src.shared.openai_execution import (
    OpenAICompatibleAsyncExecutor,
    build_openai_compatible_runtime_options,
    parse_json_block_from_text,
)
from src.shared.openai_options import OpenAICompatibleOptions
from src.shared.ai_providers import (
    CHAT_CAPABILITY,
    EMBEDDING_CAPABILITY,
    get_provider_manifest,
    normalize_provider_id,
    resolve_provider_base_url_for_capability,
)
from src.shared.user_logging import (
    log_model_input,
    log_model_request,
    log_retry,
    user_log,
)

from .config_models import EmbeddingConfig, ChatLLMConfig

logger = logging.getLogger("MangaInsight.Embedding")


class EmbeddingBusinessRetryableError(ValueError):
    """仅用于 Embedding 结果级别的可重试错误。"""


class EmbeddingClient:
    """
    向量模型客户端（复用共享 async transport）。
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.provider = normalize_provider_id(config.provider)
        self._base_url = resolve_provider_base_url_for_capability(
            self.provider,
            EMBEDDING_CAPABILITY,
            config.base_url,
        ) or ""
        self._timeout = (
            None if config.timeout_seconds == 0 else config.timeout_seconds
        )
        self._transport = AsyncOpenAICompatibleTransport(
            max_retries=config.transport_retries,
        )
        self._business_retries = config.business_retries

        logger.debug(
            "EmbeddingClient 初始化: provider=%s, base_url=%s",
            config.provider,
            self._base_url,
        )

    async def embed(self, text: str) -> list[float]:
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list):
            raise TypeError("embedding texts must be a list")
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("embedding texts must contain non-empty strings")

        if not self._base_url:
            raise ValueError(f"服务商 '{self.config.provider}' 需要设置 base_url")
        return await self._embed_request(texts)

    async def _embed_request(self, texts: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        total_attempts = self._business_retries + 1
        log_model_input(
            "向量模型",
            [
                f"{index:02d}. {text}"
                for index, text in enumerate(texts, start=1)
            ],
        )
        provider_label = get_provider_manifest(self.provider).display_name

        for attempt in range(total_attempts):
            log_model_request(
                provider=provider_label,
                model=self.config.model,
                stream=False,
                attempt=attempt + 1,
                total_attempts=total_attempts,
            )
            try:
                embeddings = await self._transport.embed(
                    UnifiedEmbeddingRequest(
                        provider=self.provider,
                        api_key=self.config.api_key,
                        model=self.config.model,
                        inputs=texts,
                        credential_version_id=self.config.credential_version_id,
                        rpm_limit=self.config.rpm_limit,
                        base_url=self.config.base_url or None,
                        timeout=self._timeout,
                    )
                )
                self._validate_embeddings_result(texts, embeddings)
                vector_size = len(embeddings[0]) if embeddings else 0
                user_log(
                    "model",
                    f"向量模型返回 {len(embeddings)} 条结果｜维度 {vector_size}",
                )
                return embeddings
            except EmbeddingBusinessRetryableError as exc:
                last_error = exc
                if attempt >= total_attempts - 1:
                    break
                logger.debug("Embedding 结果校验失败：%s", exc)
                log_retry("向量模型", attempt + 2, total_attempts, exc)
                await asyncio.sleep(1)

        if last_error:
            raise last_error
        raise RuntimeError("embedding request completed without a result")

    @staticmethod
    def _validate_embeddings_result(
        texts: list[str],
        embeddings: list[list[float]],
    ) -> None:
        if len(embeddings) != len(texts):
            raise EmbeddingBusinessRetryableError(
                f"Embedding 返回数量不匹配: 期望 {len(texts)}，实际 {len(embeddings)}"
            )
        if any(not isinstance(item, list) or len(item) == 0 for item in embeddings):
            raise EmbeddingBusinessRetryableError("Embedding 响应包含空向量")


class ChatClient:
    """JSON chat client used by Insight derived-result builders."""

    def __init__(self, config: ChatLLMConfig):
        self.config = config

        provider = normalize_provider_id(config.provider)
        custom_url = config.base_url or None
        self.provider = provider
        self._base_url = resolve_provider_base_url_for_capability(
            provider,
            CHAT_CAPABILITY,
            custom_url,
        ) or ""
        self._timeout = 120.0
        self._total_timeout = 300.0
        self._transport = AsyncOpenAICompatibleTransport()
        self._executor = OpenAICompatibleAsyncExecutor(self._transport)

        logger.debug(
            "ChatClient 初始化: provider=%s, base_url=%s",
            provider,
            self._base_url,
        )

    def _build_messages(
        self,
        prompt: str,
        system: str | None = None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def generate_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
    ) -> Any:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("chat prompt must be a non-empty string")
        if system is not None and not isinstance(system, str):
            raise TypeError("chat system prompt must be a string")
        logger.debug(
            "[ChatClient] provider=%s, base_url=%s, model=%s",
            self.config.provider,
            self._base_url,
            self.config.model,
        )

        if not self._base_url:
            raise ValueError(f"服务商 '{self.config.provider}' 需要设置 base_url")

        options = OpenAICompatibleOptions.from_dict(
            self.config.openai_options.to_dict()
        )
        use_stream = options.execution.use_stream
        logger.debug(
            "[ChatClient] use_stream=%s, config_type=%s",
            use_stream,
            type(self.config).__name__,
        )

        try:
            result = await asyncio.wait_for(
                self._executor.execute(
                    UnifiedChatRequest(
                        provider=self.provider,
                        api_key=self.config.api_key,
                        model=self.config.model,
                        credential_version_id=self.config.credential_version_id,
                        messages=self._build_messages(prompt, system),
                        base_url=self.config.base_url or None,
                        capability="chat",
                        openai_options=options,
                        runtime_options=build_openai_compatible_runtime_options(
                            timeout=self._timeout,
                            stream_output_label="漫画分析对话",
                        ),
                    ),
                    capability="chat",
                    parser=parse_json_block_from_text,
                    logger_instance=logger,
                ),
                timeout=self._total_timeout,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"对话模型调用超过总时限（{self._total_timeout:g} 秒）"
            ) from exc
        return result.parsed
