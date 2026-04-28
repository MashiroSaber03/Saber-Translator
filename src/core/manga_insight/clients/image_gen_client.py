"""
Manga Insight 生图客户端。

统一封装图片生成服务的 provider-specific 调用逻辑，让上层业务只关心：
- prompt
- 参考图
- 返回的图片 bytes
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import re
from typing import Dict, List, Optional

import httpx
from PIL import Image, ImageDraw, ImageFont

from src.shared.http_config import build_httpx_kwargs
from src.shared.ai_providers import IMAGE_GEN_CAPABILITY, resolve_provider_endpoint_for_capability
from src.shared.openai_helpers import create_openai_client

from ..config_models import ImageGenConfig
from .base_client import BaseAPIClient
from .provider_registry import get_image_gen_base_url

logger = logging.getLogger("MangaInsight.ImageGenClient")


class ImageGenClient(BaseAPIClient):
    """统一的生图客户端。"""

    def __init__(self, config: ImageGenConfig):
        self.config = config
        resolved_base_url = config.base_url or get_image_gen_base_url(config.provider)
        super().__init__(
            provider=config.provider,
            api_key=config.api_key,
            base_url=config.base_url,
            resolved_base_url=resolved_base_url,
            timeout=300.0,
            max_retries=config.max_retries,
        )
        logger.info("ImageGenClient 初始化: provider=%s, base_url=%s", config.provider, self._base_url)

    async def generate(
        self,
        prompt: str,
        reference_images: Optional[List[Dict]] = None,
    ) -> bytes:
        provider = (self.config.provider or "").lower()
        if provider == "openai":
            return await self._call_openai_api(prompt, reference_images)
        if provider == "siliconflow":
            return await self._call_siliconflow_api(prompt, reference_images)
        if provider == "qwen":
            return await self._call_qwen_api(prompt, reference_images)
        if provider == "volcano":
            return await self._call_volcano_api(prompt, reference_images)
        return await self._call_openai_compatible_api(prompt, reference_images)

    async def _call_openai_api(
        self,
        prompt: str,
        reference_images: Optional[List[Dict]] = None,
    ) -> bytes:
        base_url = self.config.base_url or get_image_gen_base_url("openai")
        return await self._call_openai_chat_completion(
            base_url=base_url,
            prompt=prompt,
            reference_images=reference_images,
            error_label="OpenAI",
        )

    async def _call_openai_compatible_api(
        self,
        prompt: str,
        reference_images: Optional[List[Dict]] = None,
    ) -> bytes:
        if not self.config.base_url:
            raise ValueError("自定义服务商需要设置 base_url")
        return await self._call_openai_chat_completion(
            base_url=self.config.base_url,
            prompt=prompt,
            reference_images=reference_images,
            error_label="OpenAI-compatible",
        )

    async def _call_openai_chat_completion(
        self,
        *,
        base_url: str,
        prompt: str,
        reference_images: Optional[List[Dict]],
        error_label: str,
    ) -> bytes:
        client = create_openai_client(
            api_key=self.config.api_key,
            base_url=base_url,
            timeout=self._timeout,
        )
        try:
            messages = self._build_multimodal_messages(prompt, reference_images)

            for retry in range(self.config.max_retries):
                try:
                    response = await asyncio.to_thread(
                        client.chat.completions.create,
                        model=self.config.model,
                        messages=messages,
                        max_tokens=4096,
                        n=1,
                    )
                    result = response.choices[0].message.content if response.choices else ""
                    return await self._extract_image_bytes(result)
                except Exception as exc:
                    logger.warning(
                        "%s API 调用失败 (尝试 %s/%s): %s",
                        error_label,
                        retry + 1,
                        self.config.max_retries,
                        exc,
                    )
                    if retry < self.config.max_retries - 1:
                        await asyncio.sleep(2 ** retry)
                    else:
                        raise
        finally:
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                await asyncio.to_thread(close_fn)

    def _build_multimodal_messages(
        self,
        prompt: str,
        reference_images: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        content: List[Dict] = []
        for ref_img in reference_images or []:
            encoded = self._encode_reference_image(ref_img)
            if not encoded:
                continue
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded}",
                },
            })
        content.append({
            "type": "text",
            "text": prompt,
        })
        return [{
            "role": "user",
            "content": content,
        }]

    def _encode_reference_image(self, ref_img: Dict) -> str:
        img_path = ref_img.get("path", "")
        if not img_path or not os.path.exists(img_path):
            return ""
        char_name = ref_img.get("name") if ref_img.get("type") == "character" else None
        encoded = self._encode_image_to_base64(img_path, character_name=char_name)
        if encoded:
            img_type = ref_img.get("type", "unknown")
            if char_name:
                logger.info("已添加角色参考图: %s (%s)", char_name, img_path)
            else:
                logger.info("已添加%s参考图: %s", img_type, img_path)
        return encoded

    async def _extract_image_bytes(self, result: str) -> bytes:
        if not result:
            raise ValueError("API 响应为空")

        md_match = re.search(r'!\[.*?\]\((data:image/[^;]+;base64,([^)]+))\)', result)
        if md_match:
            logger.info("从 Markdown 格式中提取图片数据")
            return base64.b64decode(md_match.group(2))
        if result.startswith("data:image"):
            return base64.b64decode(result.split(",", 1)[-1])
        if result.startswith("/9j/") or result.startswith("iVBOR"):
            return base64.b64decode(result)
        if result.startswith("http"):
            async with httpx.AsyncClient(**build_httpx_kwargs(result, 60)) as http_client:
                img_response = await http_client.get(result)
                img_response.raise_for_status()
                return img_response.content
        logger.warning("未知的响应格式: %s...", result[:300])
        raise ValueError("无法解析图片响应")

    def _encode_image_to_base64(self, image_path: str, character_name: Optional[str] = None) -> str:
        try:
            if character_name:
                labeled_image = self._add_character_label(image_path, character_name)
                if labeled_image:
                    return labeled_image
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as exc:
            logger.error("编码图片失败: %s", exc)
            return ""

    def _add_character_label(self, image_path: str, character_name: str) -> str:
        try:
            img = Image.open(image_path)
            label_height = max(30, min(80, int(img.height * 0.08)))
            new_height = img.height + label_height
            new_img = Image.new("RGB", (img.width, new_height), "white")
            new_img.paste(img, (0, 0))

            draw = ImageDraw.Draw(new_img)
            font = None
            font_size = int(label_height * 0.6)
            font_paths = [
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/simhei.ttf",
                "C:/Windows/Fonts/simsun.ttc",
                "/System/Library/Fonts/PingFang.ttc",
                "/Library/Fonts/Arial Unicode.ttf",
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            ]
            for font_path in font_paths:
                if not os.path.exists(font_path):
                    continue
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except Exception:
                    continue
            if font is None:
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except Exception:
                    font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), character_name, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (img.width - text_width) // 2
            y = img.height + (label_height - text_height) // 2
            draw.text((x, y), character_name, fill="black", font=font)

            buffer = io.BytesIO()
            img_format = "PNG" if image_path.lower().endswith(".png") else "JPEG"
            new_img.save(buffer, format=img_format, quality=95)
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode("utf-8")
        except Exception as exc:
            logger.warning("添加角色标签失败: %s，将使用原图", exc)
            return ""

    async def _call_siliconflow_api(
        self,
        prompt: str,
        reference_images: Optional[List[Dict]] = None,
    ) -> bytes:
        headers = self._get_headers()
        body = {
            "model": self.config.model,
            "prompt": prompt,
            "batch_size": 1,
            "num_inference_steps": 30,
        }
        if reference_images:
            encoded_images = []
            for ref in reference_images[:3]:
                try:
                    with open(ref["path"], "rb") as image_file:
                        img_data = base64.b64encode(image_file.read()).decode()
                        encoded_images.append({
                            "image": f"data:image/png;base64,{img_data}",
                            "weight": 0.3 if ref["type"] == "style" else 0.5,
                        })
                except Exception as exc:
                    logger.warning("无法读取参考图 %s: %s", ref.get("path"), exc)
            if encoded_images:
                body["image_prompts"] = encoded_images

        for retry in range(self.config.max_retries):
            try:
                endpoint = resolve_provider_endpoint_for_capability("siliconflow", IMAGE_GEN_CAPABILITY) or "/images/generations"
                response = await self.client.post(
                    f"{self.base_url.rstrip('/')}{endpoint}",
                    headers=headers,
                    json=body,
                )
                response.raise_for_status()
                result = response.json()
                return await self._extract_image_bytes_from_payload(result)
            except Exception as exc:
                logger.warning("SiliconFlow API 调用失败 (尝试 %s/%s): %s", retry + 1, self.config.max_retries, exc)
                if retry < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** retry)
                else:
                    raise

    async def _call_qwen_api(
        self,
        prompt: str,
        reference_images: Optional[List[Dict]] = None,
    ) -> bytes:
        headers = self._get_headers()
        body = {
            "model": self.config.model,
            "input": {
                "prompt": prompt,
            },
            "parameters": {
                "n": 1,
            },
        }
        for retry in range(self.config.max_retries):
            try:
                endpoint = resolve_provider_endpoint_for_capability("qwen", IMAGE_GEN_CAPABILITY) or "/services/aigc/text2image/image-synthesis"
                response = await self.client.post(
                    f"{self.base_url.rstrip('/')}{endpoint}",
                    headers=headers,
                    json=body,
                )
                response.raise_for_status()
                result = response.json()
                if result.get("output", {}).get("task_status") == "PENDING":
                    task_id = result["output"]["task_id"]
                    return await self._poll_qwen_task(task_id, headers)
                if "output" in result and "results" in result["output"]:
                    image_url = result["output"]["results"][0]["url"]
                    async with httpx.AsyncClient(**build_httpx_kwargs(image_url, 60)) as download_client:
                        image_response = await download_client.get(image_url)
                        image_response.raise_for_status()
                        return image_response.content
                raise ValueError("无法解析 API 响应")
            except Exception as exc:
                logger.warning("通义万相 API 调用失败 (尝试 %s/%s): %s", retry + 1, self.config.max_retries, exc)
                if retry < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** retry)
                else:
                    raise

    async def _poll_qwen_task(self, task_id: str, headers: Dict[str, str], max_polls: int = 60) -> bytes:
        for _ in range(max_polls):
            await asyncio.sleep(2)
            response = await self.client.get(
                f"{self.base_url.rstrip('/')}/tasks/{task_id}",
                headers=headers,
            )
            response.raise_for_status()
            result = response.json()
            status = result.get("output", {}).get("task_status")
            if status == "SUCCEEDED":
                image_url = result["output"]["results"][0]["url"]
                async with httpx.AsyncClient(**build_httpx_kwargs(image_url, 60)) as download_client:
                    image_response = await download_client.get(image_url)
                    image_response.raise_for_status()
                    return image_response.content
            if status == "FAILED":
                raise ValueError(f"任务失败: {result.get('output', {}).get('message')}")
        raise TimeoutError("任务超时")

    async def _call_volcano_api(
        self,
        prompt: str,
        reference_images: Optional[List[Dict]] = None,
    ) -> bytes:
        headers = self._get_headers()
        body = {
            "model": self.config.model,
            "prompt": prompt,
            "n": 1,
        }
        for retry in range(self.config.max_retries):
            try:
                endpoint = resolve_provider_endpoint_for_capability("volcano", IMAGE_GEN_CAPABILITY) or "/v1/images/generations"
                response = await self.client.post(
                    f"{self.base_url.rstrip('/')}{endpoint}",
                    headers=headers,
                    json=body,
                )
                response.raise_for_status()
                result = response.json()
                if "data" in result:
                    image_url = result["data"][0].get("url")
                    if image_url:
                        async with httpx.AsyncClient(**build_httpx_kwargs(image_url, 60)) as download_client:
                            image_response = await download_client.get(image_url)
                            image_response.raise_for_status()
                            return image_response.content
                raise ValueError("无法解析 API 响应")
            except Exception as exc:
                logger.warning("火山引擎 API 调用失败 (尝试 %s/%s): %s", retry + 1, self.config.max_retries, exc)
                if retry < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** retry)
                else:
                    raise

    async def _extract_image_bytes_from_payload(self, payload: Dict) -> bytes:
        for top_level_key in ("images", "data"):
            items = payload.get(top_level_key) or []
            if not items:
                continue
            image_data = items[0]
            if image_data.get("url"):
                image_url = image_data["url"]
                async with httpx.AsyncClient(**build_httpx_kwargs(image_url, 60)) as download_client:
                    image_response = await download_client.get(image_url)
                    image_response.raise_for_status()
                    return image_response.content
            if image_data.get("b64_json"):
                return base64.b64decode(image_data["b64_json"])
        raise ValueError("无法解析 API 响应")
