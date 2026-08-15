"""
Manga Insight VLM client using shared async transport.
"""

import asyncio
import base64
import io
import logging
from typing import Any

from PIL import Image

from src.shared.ai_providers import (
    VLM_CAPABILITY,
    normalize_provider_id,
    resolve_provider_base_url_for_capability,
)
from src.shared.ai_transport import AsyncOpenAICompatibleTransport, UnifiedChatRequest
from src.shared.openai_execution import (
    OpenAICompatibleAsyncExecutor,
    OpenAICompatibleBusinessRetryableError,
    build_openai_compatible_runtime_options,
    parse_json_block_from_text,
)
from src.shared.openai_options import OpenAICompatibleOptions

from .config_models import VLMConfig

logger = logging.getLogger("MangaInsight.VLM")


def _prepare_image(image_bytes: bytes, max_size: int) -> tuple[bytes, str]:
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("VLM image must be non-empty bytes")
    if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size < 0:
        raise ValueError("VLM image_max_size must be a non-negative integer")

    with Image.open(io.BytesIO(image_bytes)) as image:
        image.load()
        media_type = Image.MIME.get(image.format or "")
        if not media_type or not media_type.startswith("image/"):
            raise ValueError("VLM image format is unsupported")
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("VLM image dimensions are invalid")
        if max_size == 0 or max(width, height) <= max_size:
            return image_bytes, media_type

        ratio = max_size / max(width, height)
        resized = image.resize(
            (max(1, round(width * ratio)), max(1, round(height * ratio))),
            Image.Resampling.LANCZOS,
        )
        if resized.mode != "RGB":
            resized = resized.convert("RGB")
        output = io.BytesIO()
        resized.save(output, format="JPEG", quality=85)
        result = output.getvalue()
        if not result:
            raise ValueError("VLM image compression produced an empty image")
        logger.debug(
            "压缩 VLM 图片: %sx%s -> %sx%s",
            width,
            height,
            resized.width,
            resized.height,
        )
        return result, "image/jpeg"


class VLMClient:
    """
    多模态大模型客户端（复用共享 async transport）。
    """

    def __init__(self, config: VLMConfig):
        self.config = config
        self.provider = normalize_provider_id(config.provider)
        self._base_url = resolve_provider_base_url_for_capability(
            self.provider,
            VLM_CAPABILITY,
            config.base_url,
        ) or ""
        # Keep one transport attempt shorter than the complete logical call so
        # a stalled stream still leaves time for the configured retry layers.
        self._timeout = 120.0
        self._total_timeout = 300.0
        self._transport = AsyncOpenAICompatibleTransport()
        self._executor = OpenAICompatibleAsyncExecutor(self._transport)

        logger.info(
            "VLMClient 初始化: provider=%s, base_url=%s",
            config.provider,
            self._base_url,
        )

    async def _execute_with_total_timeout(self, *args, **kwargs):
        """Bound the complete logical VLM call, including all retry layers.

        ``httpx`` timeouts are inactivity timeouts. A remote endpoint that
        keeps dripping bytes can therefore keep one worker step alive forever,
        preventing pause/cancel from reaching the next safe point. The VLM
        timeout is intentionally also a wall-clock deadline for the complete
        executor call.
        """
        try:
            return await asyncio.wait_for(
                self._executor.execute(*args, **kwargs),
                timeout=self._total_timeout,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"视觉模型调用超过总时限（{self._total_timeout:g} 秒）"
            ) from exc

    async def analyze_page(
        self,
        image_bytes: bytes,
        page_number: int,
        prompt: str,
    ) -> dict[str, Any]:
        if (
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or page_number < 1
        ):
            raise ValueError("page_number must be a positive integer")

        def parser(response_text: str) -> dict[str, Any]:
            try:
                return self._parse_page_analysis(response_text, page_number)
            except (TypeError, ValueError) as exc:
                raise OpenAICompatibleBusinessRetryableError(
                    f"第{page_number}页 JSON 解析失败"
                ) from exc

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("VLM prompt must be a non-empty string")
        provider = self.provider
        base_url = self._base_url

        prepared, media_type = _prepare_image(
            image_bytes,
            self.config.image_max_size,
        )
        content: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        f"data:{media_type};base64,"
                        f"{base64.b64encode(prepared).decode('ascii')}"
                    )
                },
            }
        ]
        content.append({"type": "text", "text": prompt})

        if not base_url:
            raise ValueError(f"服务商 '{provider}' 需要设置 base_url")

        options = OpenAICompatibleOptions.from_dict(self.config.openai_options.to_dict())
        result = await self._execute_with_total_timeout(
            UnifiedChatRequest(
                provider=provider,
                api_key=self.config.api_key,
                model=self.config.model,
                credential_version_id=self.config.credential_version_id,
                messages=[{"role": "user", "content": content}],
                base_url=self.config.base_url or None,
                capability="vlm",
                openai_options=options,
                runtime_options=build_openai_compatible_runtime_options(
                    timeout=self._timeout,
                    print_stream_output=options.execution.use_stream,
                    stream_output_label="漫画分析",
                ),
            ),
            capability="vlm",
            parser=parser,
            logger_instance=logger,
        )
        parsed = result.parsed
        if not isinstance(parsed, dict):
            raise TypeError("VLM page analysis must be an object")
        return parsed

    def _parse_page_analysis(
        self,
        response_text: str,
        page_number: int,
    ) -> dict[str, Any]:
        result = parse_json_block_from_text(response_text)
        if not isinstance(result, dict) or set(result) != {"pages"}:
            raise ValueError("模型结果必须是只包含 pages 的对象")

        pages = result["pages"]
        if not isinstance(pages, list):
            raise ValueError("pages 必须是数组")
        if len(pages) != 1 or not isinstance(pages[0], dict):
            raise ValueError("pages 必须只包含一个页面对象")
        actual_page_number = pages[0].get("page_number")
        if (
            isinstance(actual_page_number, bool)
            or not isinstance(actual_page_number, int)
            or actual_page_number != page_number
        ):
            raise ValueError(
                f"page_number 必须为 {page_number}，实际为 {actual_page_number}"
            )
        return {"pages": pages}
