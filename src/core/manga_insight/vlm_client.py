"""
Manga Insight VLM client using shared async transport.
"""

import asyncio
import base64
import io
import logging
from typing import List, Dict, Optional

from PIL import Image

from src.shared.ai_transport import AsyncOpenAICompatibleTransport, UnifiedChatRequest
from src.shared.openai_execution import (
    OpenAICompatibleAsyncExecutor,
    OpenAICompatibleBusinessRetryableError,
    build_openai_compatible_runtime_options,
    extract_json_block_from_text,
)
from src.shared.openai_options import OpenAICompatibleOptions
from src.shared.ai_providers import (
    VLM_CAPABILITY,
    normalize_provider_id,
    resolve_provider_base_url_for_capability,
)

from .config_models import (
    VLMConfig,
    PromptsConfig,
)
from .utils.json_parser import parse_llm_json

logger = logging.getLogger("MangaInsight.VLM")


def resize_image_if_needed(image_bytes: bytes, max_size: int) -> bytes:
    if max_size <= 0:
        return image_bytes

    try:
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size

        if max(width, height) <= max_size:
            return image_bytes

        ratio = max_size / max(width, height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)

        logger.debug(f"压缩图片: {width}x{height} -> {new_width}x{new_height}")

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        output = io.BytesIO()
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(output, format='JPEG', quality=85)

        compressed_bytes = output.getvalue()
        original_size = len(image_bytes) / 1024
        compressed_size = len(compressed_bytes) / 1024
        logger.debug(f"图片大小: {original_size:.1f}KB -> {compressed_size:.1f}KB")

        return compressed_bytes
    except Exception as e:
        logger.warning(f"图片压缩失败，使用原图: {e}")
        return image_bytes


class VLMClient:
    """
    多模态大模型客户端（复用共享 async transport）。
    """

    def __init__(self, config: VLMConfig, prompts_config: Optional[PromptsConfig] = None):
        self.config = config
        self.prompts_config = prompts_config or PromptsConfig()
        self.provider = normalize_provider_id(config.provider)
        self._base_url = resolve_provider_base_url_for_capability(
            self.provider,
            VLM_CAPABILITY,
            config.base_url,
        ) or ""
        self._timeout = 300.0
        self._transport = AsyncOpenAICompatibleTransport()
        self._executor = OpenAICompatibleAsyncExecutor(self._transport)

        logger.info(f"VLMClient 初始化: provider={config.provider}, base_url={self._base_url}")

    @property
    def base_url(self) -> str:
        return self._base_url

    async def close(self):
        return None

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
                timeout=self._timeout,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"视觉模型调用超过总时限（{self._timeout:g} 秒）"
            ) from exc


    async def analyze_batch(
        self,
        images: List[bytes],
        start_page: int,
        context: Optional[Dict] = None,
        custom_prompt: Optional[str] = None
    ) -> Dict:
        end_page = start_page + len(images) - 1
        prompt = custom_prompt or self._build_batch_analysis_prompt(start_page, end_page, len(images), context)
        return await self._call_vlm(
            images=images,
            prompt=prompt,
            parser=self._build_batch_analysis_parser(start_page, end_page),
        )


    def _build_batch_analysis_prompt(self, start_page: int, end_page: int, page_count: int, context: Dict = None) -> str:
        base_prompt = self.prompts_config.batch_analysis.strip()
        if not base_prompt:
            raise ValueError("batch analysis prompt is required")
        prompt = base_prompt.replace("{page_count}", str(page_count))
        prompt = prompt.replace("{start_page}", str(start_page))
        prompt = prompt.replace("{end_page}", str(end_page))

        if context and context.get("previous_summary"):
            batch_count = context.get("context_batch_count", 3)
            if batch_count > 1:
                prompt += f"\n\n【前文概要（前{batch_count}批内容）】\n请参考以下前文信息，确保剧情连贯：\n{context['previous_summary']}"
            else:
                prompt += f"\n\n【前文概要】\n{context['previous_summary']}"

        return prompt

    def _build_batch_analysis_parser(self, start_page: int, end_page: int):
        def parser(response_text: str) -> Dict:
            result = self._parse_batch_analysis(response_text, start_page, end_page)
            if result.get("parse_error"):
                raise OpenAICompatibleBusinessRetryableError(
                    f"第{start_page}-{end_page}页 JSON 解析失败"
                )
            return result

        return parser

    async def _call_vlm(
        self,
        images: List[bytes],
        prompt: str,
        parser=None,
    ):
        provider = self.provider
        base_url = self._base_url

        content = []
        for img in images:
            img = resize_image_if_needed(img, self.config.image_max_size)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(img).decode()}"
                }
            })
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
        return result.parsed

    def _extract_json_from_text(self, text: str) -> str:
        return extract_json_block_from_text(text)

    def _parse_batch_analysis(self, response_text: str, start_page: int, end_page: int) -> Dict:
        try:
            text = self._extract_json_from_text(response_text)
            result = parse_llm_json(text)
            expected_page_count = end_page - start_page + 1
            if not isinstance(result, dict) or set(result) != {"pages"}:
                raise ValueError("模型结果必须是只包含 pages 的对象")

            pages = result["pages"]
            if not isinstance(pages, list):
                raise ValueError("pages 必须是数组")
            if len(pages) != expected_page_count:
                raise ValueError(
                    f"页面数必须为 {expected_page_count}，实际为 {len(pages)}"
                )

            expected_numbers = list(range(start_page, end_page + 1))
            page_numbers: list[int] = []
            for index, page in enumerate(pages):
                if not isinstance(page, dict):
                    raise ValueError(f"pages[{index}] 必须是对象")
                page_number = page.get("page_number")
                if isinstance(page_number, bool) or not isinstance(page_number, int):
                    raise ValueError(f"pages[{index}].page_number 必须是整数")
                page_numbers.append(page_number)
            if expected_page_count == 1:
                pages[0] = {**pages[0], "page_number": start_page}
                return {"pages": pages}
            if page_numbers != expected_numbers:
                raise ValueError(
                    f"page_number 必须依次为 {expected_numbers}，实际为 {page_numbers}"
                )

            return {"pages": pages}
        except (OpenAICompatibleBusinessRetryableError, TypeError, ValueError) as exc:
            logger.warning(
                f"批量 JSON 结果无效，第{start_page}-{end_page}页: {exc}"
            )
            return {"pages": [], "parse_error": True}

    async def test_connection(self) -> bool:
        try:
            test_prompt = "请回复'连接成功'"
            if not self._base_url:
                logger.error(f"服务商 '{self.config.provider}' 未配置 base_url")
                return False

            await self._transport.complete(
                UnifiedChatRequest(
                    provider=self.provider,
                    api_key=self.config.api_key,
                    model=self.config.model,
                    credential_version_id=self.config.credential_version_id,
                    messages=[{"role": "user", "content": test_prompt}],
                    base_url=self.config.base_url or None,
                    capability="vlm",
                    openai_options=OpenAICompatibleOptions(),
                    runtime_options=build_openai_compatible_runtime_options(
                        timeout=self._timeout,
                    ),
                )
            )
            return True
        except Exception as e:
            logger.error(f"连接测试失败: {e}")
            return False
