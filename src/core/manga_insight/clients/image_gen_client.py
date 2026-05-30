"""
Manga Insight 生图客户端。

当前版本仅保留一个明确支持的生图网关：gpt2api。
调用策略：
- 无参考图：POST /v1/images/generations
- 有任意参考图：POST /v1/images/edits

上层业务继续只关心：
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
from typing import Dict, List, Optional, Protocol
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from PIL import Image, ImageDraw, ImageFont

from src.shared.ai_transport import RETRYABLE_EXCEPTIONS, RETRYABLE_STATUS_CODES
from src.shared.http_config import build_httpx_kwargs
from src.shared.ai_providers import IMAGE_GEN_CAPABILITY, normalize_provider_id, provider_supports_capability

from ..config_models import ImageGenConfig
from .base_client import BaseAPIClient
from .provider_registry import get_image_gen_base_url

logger = logging.getLogger("MangaInsight.ImageGenClient")

_DOWNLOAD_TIMEOUT_SECONDS = 60
DEFAULT_IMAGE_GEN_TRANSPORT_RETRIES = 10
DEFAULT_IMAGE_GEN_BUSINESS_RETRIES = 10


class ImageGenBusinessRetryableError(ValueError):
    """仅用于生图结果级别的可重试错误。"""


class ImageGenAdapter(Protocol):
    async def generate(
        self,
        client: "ImageGenClient",
        prompt: str,
        prepared_refs: List[Dict[str, object]],
    ) -> bytes:
        ...


class GPT2ApiImageGenAdapter:
    async def generate(
        self,
        client: "ImageGenClient",
        prompt: str,
        prepared_refs: List[Dict[str, object]],
    ) -> bytes:
        request_url = client._build_api_url("images/edits" if prepared_refs else "images/generations")
        return await client._request_image_bytes(request_url, prompt, prepared_refs)


IMAGE_GEN_ADAPTERS: Dict[str, ImageGenAdapter] = {
    "gpt2api": GPT2ApiImageGenAdapter(),
}


class ImageGenClient(BaseAPIClient):
    """专门适配 gpt2api 的生图客户端。"""

    def __init__(self, config: ImageGenConfig):
        self.config = config
        resolved_base_url = config.base_url or get_image_gen_base_url(config.provider)
        timeout_value = float(config.timeout_seconds or 0)
        self._timeout = None if timeout_value <= 0 else timeout_value
        transport_retries = config.transport_retries if config.transport_retries is not None else DEFAULT_IMAGE_GEN_TRANSPORT_RETRIES
        business_retries = config.business_retries if config.business_retries is not None else DEFAULT_IMAGE_GEN_BUSINESS_RETRIES
        self._transport_retries = max(0, int(transport_retries))
        self._business_retries = max(0, int(business_retries))
        super().__init__(
            provider=config.provider,
            api_key=config.api_key,
            base_url=config.base_url,
            resolved_base_url=resolved_base_url,
            timeout=self._timeout,
        )
        logger.info("ImageGenClient 初始化: provider=%s, base_url=%s", config.provider, self._base_url)

    async def generate(
        self,
        prompt: str,
        reference_images: Optional[List[Dict]] = None,
    ) -> bytes:
        provider = normalize_provider_id(self.config.provider)
        if not provider_supports_capability(provider, IMAGE_GEN_CAPABILITY):
            raise ValueError(f"服务商 '{self.config.provider}' 不支持 image_gen 能力")
        adapter = IMAGE_GEN_ADAPTERS.get(provider)
        if adapter is None:
            raise ValueError(f"服务商 '{self.config.provider}' 尚未注册生图适配器")
        if not self.base_url:
            raise ValueError(f"{self.config.provider} 生图服务商需要设置 base_url")

        prepared_refs = self._prepare_reference_images(reference_images)
        return await adapter.generate(self, prompt, prepared_refs)

    async def _request_image_bytes(
        self,
        request_url: str,
        prompt: str,
        prepared_refs: List[Dict[str, object]],
    ) -> bytes:
        last_error: Optional[Exception] = None
        total_attempts = self._business_retries + 1

        for attempt in range(total_attempts):
            try:
                payload = await self._request_generation_payload(request_url, prompt, prepared_refs)
                if not self._payload_has_result(payload):
                    raise ImageGenBusinessRetryableError("gpt2api 返回中没有图片结果")
                try:
                    return await self._extract_image_bytes_from_payload(payload)
                except ValueError as exc:
                    raise ImageGenBusinessRetryableError(str(exc)) from exc
            except ImageGenBusinessRetryableError as exc:
                last_error = exc
                if attempt >= total_attempts - 1:
                    break
                logger.warning(
                    "gpt2api 生图业务重试 %s/%s: %s",
                    attempt + 1,
                    self._business_retries,
                    exc,
                )
                await asyncio.sleep(1)

        if last_error:
            raise last_error
        raise RuntimeError("生图响应为空")

    async def _request_generation_payload(
        self,
        request_url: str,
        prompt: str,
        prepared_refs: List[Dict[str, object]],
    ) -> Dict:
        last_exception: Optional[Exception] = None

        for attempt in range(self._transport_retries + 1):
            try:
                if prepared_refs:
                    response = await self.client.post(
                        request_url,
                        headers=self._build_multipart_headers(),
                        data=self._build_edit_form_data(prompt),
                        files=self._build_edit_files(prepared_refs),
                    )
                else:
                    response = await self.client.post(
                        request_url,
                        headers=self._get_headers(),
                        json=self._build_generation_body(prompt),
                    )

                if response.status_code in RETRYABLE_STATUS_CODES and attempt < self._transport_retries:
                    logger.warning(
                        "gpt2api 生图传输重试 %s/%s: HTTP %s",
                        attempt + 1,
                        self._transport_retries,
                        response.status_code,
                    )
                    await asyncio.sleep(2 ** attempt)
                    continue

                payload = self._decode_response_payload(response)
                self._raise_api_error_if_needed(response, payload)
                return payload
            except RETRYABLE_EXCEPTIONS as exc:
                last_exception = exc
                if attempt < self._transport_retries:
                    logger.warning(
                        "gpt2api 生图传输重试 %s/%s: %s",
                        attempt + 1,
                        self._transport_retries,
                        type(exc).__name__,
                    )
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("生图传输重试耗尽")

    def _build_generation_body(self, prompt: str) -> Dict[str, object]:
        return {
            "model": self.config.model,
            "prompt": prompt,
            "n": 1,
            "response_format": "b64_json",
        }

    def _build_edit_form_data(self, prompt: str) -> Dict[str, str]:
        return {
            "model": self.config.model,
            "prompt": prompt,
            "n": "1",
            "response_format": "b64_json",
        }

    def _build_edit_files(self, references: List[Dict[str, object]]) -> List[tuple[str, tuple[str, bytes, str]]]:
        files: List[tuple[str, tuple[str, bytes, str]]] = []
        for ref in references:
            files.append((
                "image",
                (
                    str(ref["filename"]),
                    ref["bytes"],  # type: ignore[index]
                    str(ref["mime"]),
                ),
            ))
        return files

    def _build_multipart_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
        }

    def _prepare_reference_images(self, reference_images: Optional[List[Dict]]) -> List[Dict[str, object]]:
        prepared: List[Dict[str, object]] = []
        for ref_img in reference_images or []:
            encoded = self._encode_reference_image(ref_img)
            if encoded is not None:
                prepared.append(encoded)
        return prepared

    def _encode_reference_image(self, ref_img: Dict) -> Optional[Dict[str, object]]:
        img_path = ref_img.get("path", "")
        if not img_path or not os.path.exists(img_path):
            return None

        char_name = ref_img.get("name") if ref_img.get("type") == "character" else None
        try:
            image_bytes, mime = self._read_image_bytes(img_path, character_name=char_name)
            if char_name:
                logger.info("已添加角色参考图: %s (%s)", char_name, img_path)
            else:
                logger.info("已添加%s参考图: %s", ref_img.get("type", "unknown"), img_path)
            suffix = os.path.splitext(img_path)[1].lower()
            ext = suffix if suffix else ".png"
            filename = os.path.basename(img_path) or f"reference{ext}"
            return {
                "filename": filename,
                "bytes": image_bytes,
                "mime": mime,
            }
        except Exception as exc:
            logger.error("编码参考图失败: %s", exc)
            return None

    def _read_image_bytes(self, image_path: str, character_name: Optional[str] = None) -> tuple[bytes, str]:
        if character_name:
            labeled = self._add_character_label(image_path, character_name)
            if labeled is not None:
                return labeled

        with open(image_path, "rb") as image_file:
            data = image_file.read()
        mime = self._guess_mime_type(image_path)
        return data, mime

    def _guess_mime_type(self, image_path: str) -> str:
        suffix = os.path.splitext(image_path)[1].lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(suffix, "image/png")

    def _add_character_label(self, image_path: str, character_name: str) -> Optional[tuple[bytes, str]]:
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
            mime = self._guess_mime_type(image_path)
            image_format = "PNG" if mime == "image/png" else "JPEG"
            new_img.save(buffer, format=image_format, quality=95)
            return buffer.getvalue(), mime
        except Exception as exc:
            logger.warning("添加角色标签失败: %s，将使用原图", exc)
            return None

    def _build_api_url(self, route: str) -> str:
        parsed = urlparse(self.base_url.rstrip("/"))
        path = parsed.path.rstrip("/")
        route_path = route.lstrip("/")
        if path.endswith("/v1"):
            new_path = f"{path}/{route_path}"
        elif path:
            new_path = f"{path}/v1/{route_path}"
        else:
            new_path = f"/v1/{route_path}"
        return urlunparse(parsed._replace(path=new_path, params="", query="", fragment=""))

    def _payload_has_result(self, payload: Dict) -> bool:
        return bool(self._extract_result_items(payload))

    def _extract_result_items(self, payload: Dict) -> List[Dict]:
        if isinstance(payload.get("data"), list):
            return [item for item in payload["data"] if isinstance(item, dict)]
        result = payload.get("result")
        if isinstance(result, dict) and isinstance(result.get("data"), list):
            return [item for item in result["data"] if isinstance(item, dict)]
        return []

    def _extract_error_message(self, payload: Dict) -> str:
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message", "")).strip()
        if isinstance(error, str):
            return error.strip()
        return ""

    def _raise_api_error_if_needed(self, response: httpx.Response, payload: Dict) -> None:
        error_message = self._extract_error_message(payload)
        if response.status_code >= 400:
            if error_message:
                raise ValueError(error_message)
            raise ValueError(f"gpt2api 请求失败: HTTP {response.status_code}")
        if error_message and not self._payload_has_result(payload):
            raise ValueError(error_message)

    def _decode_response_payload(self, response: httpx.Response) -> Dict:
        try:
            return response.json()
        except ValueError as exc:
            raw_excerpt = (response.text or "")[:300]
            raise ValueError(f"gpt2api 返回了非 JSON 响应: {raw_excerpt}") from exc

    async def _extract_image_bytes_from_payload(self, payload: Dict) -> bytes:
        items = self._extract_result_items(payload)
        if not items:
            raise ValueError("gpt2api 返回中没有图片数据")

        image_item = items[0]
        if image_item.get("b64_json"):
            return base64.b64decode(image_item["b64_json"])

        image_url = str(image_item.get("url", "")).strip()
        if not image_url:
            raise ValueError("gpt2api 返回的图片项缺少 url")
        return await self._download_image_asset(image_url)

    async def _download_image_asset(self, asset_value: str) -> bytes:
        asset_value = asset_value.strip()
        if not asset_value:
            raise ValueError("图片资源为空")

        if asset_value.startswith("data:image"):
            return base64.b64decode(asset_value.split(",", 1)[-1])
        if asset_value.startswith("/9j/") or asset_value.startswith("iVBOR"):
            return base64.b64decode(asset_value)

        if asset_value.startswith(("http://", "https://", "/")) or not asset_value.startswith("data:"):
            download_url = asset_value if asset_value.startswith(("http://", "https://")) else urljoin(self.base_url.rstrip("/") + "/", asset_value.lstrip("/"))
            async with httpx.AsyncClient(**build_httpx_kwargs(download_url, _DOWNLOAD_TIMEOUT_SECONDS)) as http_client:
                response = await http_client.get(download_url)
                response.raise_for_status()
                return response.content

        raise ValueError(f"无法解析 gpt2api 图片资源: {asset_value[:120]}")
