"""
Manga Insight VLM client using shared async transport.
"""

import asyncio
import base64
import io
import logging
import re
from typing import List, Dict, Optional

from PIL import Image

from src.shared.ai_transport import AsyncOpenAICompatibleTransport, UnifiedChatRequest

from .clients.base_client import RPMLimiter
from .clients.provider_registry import get_base_url
from .config_models import (
    VLMConfig,
    PromptsConfig,
    DEFAULT_BATCH_ANALYSIS_PROMPT
)
from .utils.json_parser import parse_llm_json

logger = logging.getLogger("MangaInsight.VLM")
DEFAULT_VLM_MAX_RETRIES = 3


def _provider_id(value) -> str:
    if isinstance(value, str):
        return value.lower()
    return str(getattr(value, "value", value)).lower()


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
        self.provider = _provider_id(config.provider)
        self._base_url = get_base_url(self.provider, config.base_url)
        self._rpm_limiter = RPMLimiter(config.rpm_limit)
        self._timeout = 300.0
        self._max_retries = DEFAULT_VLM_MAX_RETRIES
        self._transport = AsyncOpenAICompatibleTransport(max_retries=DEFAULT_VLM_MAX_RETRIES)

        logger.info(f"VLMClient 初始化: provider={config.provider}, base_url={self._base_url}")

    @property
    def base_url(self) -> str:
        return self._base_url

    async def close(self):
        return None

    async def _enforce_rpm_limit(self):
        await self._rpm_limiter.wait()

    def is_configured(self) -> bool:
        return bool(self.config.api_key and self.config.model)

    async def analyze_batch(
        self,
        images: List[bytes],
        start_page: int,
        context: Optional[Dict] = None,
        custom_prompt: Optional[str] = None
    ) -> Dict:
        end_page = start_page + len(images) - 1
        prompt = custom_prompt or self._build_batch_analysis_prompt(start_page, end_page, len(images), context)

        for attempt in range(DEFAULT_VLM_MAX_RETRIES + 1):
            response_text = await self._call_vlm(
                images=images,
                prompt=prompt
            )

            result = self._parse_batch_analysis(response_text, start_page, end_page)

            if not result.get("parse_error"):
                return result

            if attempt < DEFAULT_VLM_MAX_RETRIES:
                logger.warning(f"第{start_page}-{end_page}页 JSON 解析失败，第 {attempt + 1} 次重试...")
                await asyncio.sleep(2 ** attempt)
            else:
                logger.error(f"第{start_page}-{end_page}页 JSON 解析失败，已达最大重试次数 ({DEFAULT_VLM_MAX_RETRIES})")

        return result

    def _build_batch_analysis_prompt(self, start_page: int, end_page: int, page_count: int, context: Dict = None) -> str:
        base_prompt = self.prompts_config.batch_analysis if self.prompts_config.batch_analysis else DEFAULT_BATCH_ANALYSIS_PROMPT
        prompt = base_prompt.replace("{page_count}", str(page_count))
        prompt = prompt.replace("{start_page}", str(start_page))
        prompt = prompt.replace("{end_page}", str(end_page))

        if context and context.get("previous_summary"):
            batch_count = context.get("context_batch_count", 1)
            if batch_count > 1:
                prompt += f"\n\n【前文概要（前{batch_count}批内容）】\n请参考以下前文信息，确保剧情连贯：\n{context['previous_summary']}"
            else:
                prompt += f"\n\n【前文概要】\n{context['previous_summary']}"

        return prompt

    async def _call_vlm(self, images: List[bytes], prompt: str) -> str:
        await self._enforce_rpm_limit()
        for attempt in range(self._max_retries + 1):
            try:
                return await self._call_openai_compatible(images, prompt)
            except Exception as e:
                error_msg = str(e) if str(e) else type(e).__name__
                if hasattr(e, "response"):
                    try:
                        resp = e.response
                        if hasattr(resp, "text"):
                            error_msg = f"{error_msg} - Response: {resp.text[:500]}"
                        elif hasattr(resp, "content"):
                            error_msg = f"{error_msg} - Content: {resp.content[:500]}"
                    except Exception:
                        pass
                logger.warning(f"VLM 调用失败 (尝试 {attempt + 1}/{self._max_retries + 1}): {error_msg}")
                if attempt < self._max_retries:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise Exception(f"VLM 调用失败: {error_msg}")

    async def _call_openai_compatible(self, images: List[bytes], prompt: str) -> str:
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

        return await self._transport.complete(
            UnifiedChatRequest(
                provider=provider,
                api_key=self.config.api_key,
                model=self.config.model,
                messages=[{"role": "user", "content": content}],
                base_url=self.config.base_url or None,
                timeout=self._timeout,
                use_stream=self.config.use_stream,
                print_stream_output=self.config.use_stream,
                response_format={"type": "json_object"} if self.config.force_json else None,
                temperature=self.config.temperature,
            )
        )

    def _clean_thinking_tags(self, text: str) -> str:
        patterns = [
            r'<think>.*?</think>',
            r'<thinking>.*?</thinking>',
            r'<reasoning>.*?</reasoning>',
            r'<thought>.*?</thought>',
            r'<reflection>.*?</reflection>',
            r'<内心独白>.*?</内心独白>',
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        return text.strip()

    def _extract_json_from_text(self, text: str) -> str:
        text = self._clean_thinking_tags(text)
        text = text.strip()

        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        if not text.startswith('{') and not text.startswith('['):
            json_start = -1
            for i, char in enumerate(text):
                if char in '{[':
                    json_start = i
                    break
            if json_start >= 0:
                text = text[json_start:]

        text = self._find_complete_json(text)
        return text

    def _find_complete_json(self, text: str) -> str:
        if not text:
            return text

        open_char = text[0] if text else ''
        if open_char == '{':
            close_char = '}'
        elif open_char == '[':
            close_char = ']'
        else:
            return text

        depth = 0
        in_string = False
        escape = False

        for i, char in enumerate(text):
            if escape:
                escape = False
                continue

            if char == '\\':
                escape = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
                if depth == 0:
                    return text[:i + 1]

        return text

    def _parse_batch_analysis(self, response_text: str, start_page: int, end_page: int) -> Dict:
        text = self._extract_json_from_text(response_text)

        result = parse_llm_json(text)

        if not result:
            logger.warning(f"批量 JSON 解析失败，第{start_page}-{end_page}页")
            page_count = end_page - start_page + 1
            return {
                "page_range": {"start": start_page, "end": end_page},
                "pages": [{
                    "page_number": start_page + i,
                    "raw_response": response_text[:2000] if i == 0 else "",
                    "parse_error": True
                } for i in range(page_count)],
                "batch_summary": "",
                "key_events": [],
                "continuity_notes": "",
                "parse_error": True
            }

        try:
            if isinstance(result, list):
                result = {
                    "page_range": {"start": start_page, "end": end_page},
                    "pages": result,
                    "batch_summary": "",
                    "key_events": [],
                    "continuity_notes": ""
                }

            if "page_range" not in result:
                result["page_range"] = {"start": start_page, "end": end_page}

            if "pages" not in result or not result["pages"]:
                logger.warning(f"批量分析结果缺少或空的 pages 字段，返回的键: {list(result.keys())}")
                for key in ["page_analyses", "analysis", "results", "data", "page_list"]:
                    if key in result and isinstance(result[key], list) and len(result[key]) > 0:
                        result["pages"] = result[key]
                        logger.info(f"从 '{key}' 字段提取到 {len(result['pages'])} 个页面")
                        break
                else:
                    if not result.get("pages"):
                        batch_summary = result.get("batch_summary", "")
                        if batch_summary:
                            logger.info(f"使用 batch_summary 为第{start_page}-{end_page}页生成基本页面数据")
                            result["pages"] = []
                            for page_num in range(start_page, end_page + 1):
                                result["pages"].append({
                                    "page_number": page_num,
                                    "page_summary": batch_summary if page_num == start_page else f"（见第{start_page}页批次摘要）",
                                    "from_batch_summary": True
                                })
                        else:
                            result["pages"] = []
                            logger.warning(f"无法提取页面数据，原始响应前500字符: {response_text[:500]}")

            normalized_pages = []
            for page in result.get("pages", []):
                if not isinstance(page, dict):
                    normalized_pages.append(page)
                    continue
                normalized = dict(page)
                if "page_number" not in normalized and isinstance(normalized.get("page_num"), int):
                    normalized["page_number"] = normalized["page_num"]
                normalized_pages.append(normalized)
            result["pages"] = normalized_pages

            expected_page_count = end_page - start_page + 1
            pages = result.get("pages", [])
            if len(pages) != expected_page_count:
                logger.warning(
                    f"页面数不匹配: 期望 {expected_page_count}, 实际 {len(pages)} "
                    f"(第{start_page}-{end_page}页)"
                )
                result["parse_error"] = True

                if pages:
                    page_numbers = [
                        p.get("page_number", p.get("page_num", 0))
                        for p in pages
                        if isinstance(p, dict)
                    ]
                    if set(page_numbers) == set(range(start_page, end_page + 1)):
                        result["pages"] = sorted(
                            pages,
                            key=lambda x: x.get("page_number", x.get("page_num", 0)) if isinstance(x, dict) else 0,
                        )
                    else:
                        result["pages"] = []
                        logger.warning(f"无法提取页面数据，原始响应前500字符: {response_text[:500]}")

            return result
        except Exception as e:
            logger.warning(f"批量分析结果处理异常: {e}")
            return result if result else {
                "page_range": {"start": start_page, "end": end_page},
                "pages": [],
                "batch_summary": "",
                "key_events": [],
                "continuity_notes": "",
                "parse_error": True
            }

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
                    messages=[{"role": "user", "content": test_prompt}],
                    base_url=self.config.base_url or None,
                    timeout=self._timeout,
                    request_overrides={"max_tokens": 10},
                )
            )
            return True
        except Exception as e:
            logger.error(f"连接测试失败: {e}")
            return False
