"""
Manga Insight Embedding / Chat clients backed by shared async transport.
"""

import logging
from typing import Any, Callable, List, Optional, TypeVar

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

from .clients.base_client import RPMLimiter
from .clients.provider_registry import get_base_url
from .config_models import EmbeddingConfig, ChatLLMConfig

logger = logging.getLogger("MangaInsight.Embedding")
DEFAULT_EMBEDDING_MAX_RETRIES = 3
T = TypeVar("T")


def _provider_id(value) -> str:
    if isinstance(value, str):
        return value.lower()
    return str(getattr(value, "value", value)).lower()


class EmbeddingClient:
    """
    向量模型客户端（复用共享 async transport）。
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.provider = _provider_id(config.provider)
        self._base_url = get_base_url(self.provider, config.base_url)
        self._rpm_limiter = RPMLimiter(config.rpm_limit, bucket_id=f"embedding:{self.provider}")
        self._timeout = 60.0
        self._transport = AsyncOpenAICompatibleTransport(max_retries=DEFAULT_EMBEDDING_MAX_RETRIES)

        logger.info(f"EmbeddingClient 初始化: provider={config.provider}, base_url={self._base_url}")

    @property
    def base_url(self) -> str:
        return self._base_url

    async def close(self):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _enforce_rpm_limit(self):
        await self._rpm_limiter.wait()

    async def embed(self, text: str) -> List[float]:
        embeddings = await self.embed_batch([text])
        return embeddings[0] if embeddings else []

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if not self._base_url:
            raise ValueError(f"服务商 '{self.config.provider}' 需要设置 base_url")

        await self._enforce_rpm_limit()
        return await self._transport.embed(
            UnifiedEmbeddingRequest(
                provider=self.provider,
                api_key=self.config.api_key,
                model=self.config.model,
                inputs=texts,
                base_url=self.config.base_url or None,
                timeout=self._timeout,
            )
        )

    async def test_connection(self) -> bool:
        try:
            embedding = await self.embed("测试文本")
            return len(embedding) > 0
        except Exception as e:
            logger.error(f"Embedding 连接测试失败: {e}")
            return False


class ChatClient:
    """
    对话模型客户端（复用共享 async transport）。
    """

    def __init__(self, config: ChatLLMConfig):
        self.config = config

        provider = _provider_id(getattr(config, "provider", "openai"))
        custom_url = config.base_url if hasattr(config, "base_url") and config.base_url else None
        self.provider = provider
        self._base_url = get_base_url(provider, custom_url)
        self._timeout = 120.0
        self._transport = AsyncOpenAICompatibleTransport()
        self._executor = OpenAICompatibleAsyncExecutor(self._transport)

        logger.info(f"ChatClient 初始化: provider={provider}, base_url={self._base_url}")

    @property
    def base_url(self) -> str:
        return self._base_url

    async def close(self):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def _build_messages(self, prompt: str, system: Optional[str] = None) -> List[dict[str, str]]:
        messages: List[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_options(self, temperature: Optional[float] = None) -> OpenAICompatibleOptions:
        options = OpenAICompatibleOptions.from_dict(self.config.openai_options.to_dict())
        if temperature is not None:
            options.request.temperature = temperature
        return options

    async def _execute_request(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        parser: Optional[Callable[[str], T]] = None,
    ) -> T | str:
        logger.debug(f"[ChatClient] provider={self.config.provider}, base_url={self._base_url}, model={self.config.model}")

        if not self._base_url:
            raise ValueError(f"服务商 '{self.config.provider}' 需要设置 base_url")

        options = self._build_options(temperature)
        use_stream = options.execution.use_stream
        logger.debug(f"[ChatClient] use_stream={use_stream}, config_type={type(self.config).__name__}")

        result = await self._executor.execute(
            UnifiedChatRequest(
                provider=self.provider,
                api_key=self.config.api_key,
                model=self.config.model,
                messages=self._build_messages(prompt, system),
                base_url=getattr(self.config, "base_url", None) or None,
                capability="chat",
                openai_options=options,
                runtime_options=build_openai_compatible_runtime_options(
                    timeout=self._timeout,
                    print_stream_output=options.execution.use_stream,
                    stream_output_label="漫画分析对话",
                ),
            ),
            capability="chat",
            parser=parser,
            logger_instance=logger,
        )
        return result.parsed

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        result = await self._execute_request(
            prompt,
            system=system,
            temperature=temperature,
        )
        return str(result)

    async def generate_parsed(
        self,
        prompt: str,
        *,
        parser: Callable[[str], T],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> T:
        result = await self._execute_request(
            prompt,
            system=system,
            temperature=temperature,
            parser=parser,
        )
        return result  # type: ignore[return-value]

    async def generate_json(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Any:
        return await self.generate_parsed(
            prompt,
            parser=parse_json_block_from_text,
            system=system,
            temperature=temperature,
        )

    async def test_connection(self) -> bool:
        try:
            response = await self.generate("测试", temperature=0)
            return len(response) > 0
        except Exception as e:
            logger.error(f"LLM 连接测试失败: {e}")
            return False
