"""
Manga Insight 生图客户端。

当前版本支持 OpenAI 兼容图片接口网关。
调用策略（相对于用户配置的 API Base URL）：
- 无参考图：POST images/generations
- 有任意参考图：POST images/edits

上层业务继续只关心：
- prompt
- 参考图
- 返回的图片 bytes
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx

from src.shared.ai_providers import (
    IMAGE_GEN_CAPABILITY,
    get_provider_manifest,
    normalize_provider_id,
    provider_requires_model,
    provider_supports_capability,
    resolve_provider_base_url_for_capability,
)
from src.shared.ai_transport import RETRYABLE_EXCEPTIONS, RETRYABLE_STATUS_CODES
from src.shared.http_config import build_httpx_kwargs, is_local_service

from ..config_models import ImageGenConfig

logger = logging.getLogger("MangaInsight.ImageGenClient")

class ImageGenBusinessRetryableError(ValueError):
    """仅用于生图结果级别的可重试错误。"""


class ImageGenClient:
    """OpenAI 兼容图片接口客户端。"""

    def __init__(self, config: ImageGenConfig):
        self.config = config
        self.provider = normalize_provider_id(config.provider)
        self.api_key = config.api_key
        self._base_url = config.base_url or (
            resolve_provider_base_url_for_capability(
                self.provider,
                IMAGE_GEN_CAPABILITY,
            )
            or ""
        )
        self._timeout = (
            None if config.timeout_seconds == 0 else config.timeout_seconds
        )
        self._transport_retries = config.transport_retries
        self._business_retries = config.business_retries
        if is_local_service(self._base_url):
            logger.info("检测到本地生图服务 (%s)，禁用代理", self._base_url)
        self.client = httpx.AsyncClient(
            **build_httpx_kwargs(self._base_url, self._timeout)
        )
        logger.info(
            "ImageGenClient 初始化: provider=%s, base_url=%s",
            config.provider,
            self._base_url,
        )

    @property
    def base_url(self) -> str:
        return self._base_url

    async def close(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> "ImageGenClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate(
        self,
        prompt: str,
        reference_images: list[dict[str, object]] | None = None,
    ) -> bytes:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("image generation prompt must be a non-empty string")
        provider = self.provider
        if not provider_supports_capability(provider, IMAGE_GEN_CAPABILITY):
            raise ValueError(f"服务商 '{self.config.provider}' 不支持 image_gen 能力")
        if not self.base_url:
            raise ValueError(f"{self.config.provider} 生图服务商需要设置 base_url")
        if get_provider_manifest(provider).requires_api_key and not self.api_key.strip():
            raise ValueError(f"{self.config.provider} 生图服务商需要设置 api_key")
        if provider_requires_model(provider) and not self.config.model.strip():
            raise ValueError(f"{self.config.provider} 生图服务商需要设置 model")

        prepared_refs = self._prepare_reference_images(reference_images)
        request_url = self._build_api_url(
            "images/edits" if prepared_refs else "images/generations"
        )
        return await self._request_image_bytes(
            request_url,
            prompt,
            prepared_refs,
        )

    async def _request_image_bytes(
        self,
        request_url: str,
        prompt: str,
        prepared_refs: list[dict[str, object]],
    ) -> bytes:
        last_error: Exception | None = None
        total_attempts = self._business_retries + 1

        for attempt in range(total_attempts):
            try:
                payload = await self._request_generation_payload(request_url, prompt, prepared_refs)
                if not self._payload_has_result(payload):
                    raise ImageGenBusinessRetryableError(f"{self.config.provider} 返回中没有图片结果")
                try:
                    return await self._extract_image_bytes_from_payload(payload)
                except ValueError as exc:
                    raise ImageGenBusinessRetryableError(str(exc)) from exc
            except ImageGenBusinessRetryableError as exc:
                last_error = exc
                if attempt >= total_attempts - 1:
                    break
                logger.warning(
                    "%s 生图业务重试 %s/%s: %s",
                    self.config.provider,
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
        prepared_refs: list[dict[str, object]],
    ) -> dict[str, Any]:

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

                if (
                    response.status_code in RETRYABLE_STATUS_CODES
                    and attempt < self._transport_retries
                ):
                    logger.warning(
                        "%s 生图传输重试 %s/%s: HTTP %s",
                        self.config.provider,
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
                if attempt < self._transport_retries:
                    logger.warning(
                        "%s 生图传输重试 %s/%s: %s",
                        self.config.provider,
                        attempt + 1,
                        self._transport_retries,
                        type(exc).__name__,
                    )
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise

        raise RuntimeError("生图传输重试耗尽")

    def _build_generation_body(self, prompt: str) -> dict[str, object]:
        return {
            "model": self.config.model,
            "prompt": prompt,
            "n": 1,
            "response_format": "b64_json",
        }

    def _build_edit_form_data(self, prompt: str) -> dict[str, str]:
        return {
            "model": self.config.model,
            "prompt": prompt,
            "n": "1",
            "response_format": "b64_json",
        }

    def _build_edit_files(
        self,
        references: list[dict[str, object]],
    ) -> list[tuple[str, tuple[str, bytes, str]]]:
        files: list[tuple[str, tuple[str, bytes, str]]] = []
        for ref in references:
            filename = ref.get("filename")
            image_bytes = ref.get("bytes")
            media_type = ref.get("mime")
            if not isinstance(filename, str) or not filename:
                raise ValueError("reference filename is invalid")
            if not isinstance(image_bytes, bytes) or not image_bytes:
                raise ValueError("reference image bytes are invalid")
            if not isinstance(media_type, str) or not media_type.startswith("image/"):
                raise ValueError("reference image MIME type is invalid")
            files.append((
                "image",
                (
                    filename,
                    image_bytes,
                    media_type,
                ),
            ))
        return files

    def _build_multipart_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
        }

    def _prepare_reference_images(
        self,
        reference_images: list[dict[str, object]] | None,
    ) -> list[dict[str, object]]:
        if reference_images is None:
            return []
        if not isinstance(reference_images, list):
            raise TypeError("reference_images must be a list")
        return [self._encode_reference_image(ref) for ref in reference_images]

    def _encode_reference_image(
        self,
        ref_img: dict[str, object],
    ) -> dict[str, object]:
        if not isinstance(ref_img, dict) or set(ref_img) != {"path", "type"}:
            raise ValueError("reference image must contain exactly path and type")
        image_path = ref_img["path"]
        reference_type = ref_img["type"]
        if not isinstance(image_path, str) or not image_path:
            raise ValueError("reference image path is invalid")
        if reference_type != "style":
            raise ValueError("reference image type must be style")
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"reference image does not exist: {image_path}")
        image_bytes, media_type = self._read_image_bytes(image_path)
        filename = os.path.basename(image_path)
        if not filename:
            raise ValueError("reference image filename is invalid")
        logger.info("已添加风格参考图: %s", image_path)
        return {
            "filename": filename,
            "bytes": image_bytes,
            "mime": media_type,
        }

    def _read_image_bytes(self, image_path: str) -> tuple[bytes, str]:
        with open(image_path, "rb") as image_file:
            data = image_file.read()
        if not data:
            raise ValueError("reference image is empty")
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return data, "image/png"
        if data.startswith(b"\xff\xd8\xff"):
            return data, "image/jpeg"
        if data.startswith((b"GIF87a", b"GIF89a")):
            return data, "image/gif"
        if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return data, "image/webp"
        raise ValueError("reference image format is unsupported")

    def _build_api_url(self, route: str) -> str:
        parsed = urlparse(self.base_url.rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("image generation base_url must be an absolute HTTP URL")
        if parsed.params or parsed.query or parsed.fragment:
            raise ValueError(
                "image generation base_url must not contain params, query, "
                "or fragment"
            )
        path = parsed.path.rstrip("/")
        route_path = route.lstrip("/")
        new_path = f"{path}/{route_path}" if path else f"/{route_path}"
        return urlunparse(parsed._replace(path=new_path, params="", query="", fragment=""))

    def _payload_has_result(self, payload: dict[str, Any]) -> bool:
        return bool(self._extract_result_items(payload))

    def _extract_result_items(
        self,
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        data = payload.get("data")
        if data is None:
            return []
        if not isinstance(data, list):
            raise ValueError("image generation response data must be an array")
        if not data:
            return []
        if len(data) != 1 or not isinstance(data[0], dict):
            raise ValueError("image generation response must contain one image object")
        return data

    def _extract_error_message(self, payload: dict[str, Any]) -> str:
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            return message.strip() if isinstance(message, str) else ""
        if isinstance(error, str):
            return error.strip()
        return ""

    def _raise_api_error_if_needed(
        self,
        response: httpx.Response,
        payload: dict[str, Any],
    ) -> None:
        error_message = self._extract_error_message(payload)
        if response.status_code >= 400:
            if error_message:
                raise ValueError(error_message)
            raise ValueError(f"{self.config.provider} 请求失败: HTTP {response.status_code}")
        if error_message and not self._payload_has_result(payload):
            raise ValueError(error_message)

    def _decode_response_payload(self, response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise ValueError(f"{self.config.provider} 返回了非 JSON 响应") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{self.config.provider} 响应必须是 JSON 对象")
        return payload

    async def _extract_image_bytes_from_payload(
        self,
        payload: dict[str, Any],
    ) -> bytes:
        items = self._extract_result_items(payload)
        if not items:
            raise ValueError(f"{self.config.provider} 返回中没有图片数据")

        image_item = items[0]
        keys = {key for key in ("b64_json", "url") if key in image_item}
        if len(keys) != 1:
            raise ValueError("image result must contain exactly one of b64_json or url")
        if "b64_json" in keys:
            encoded = image_item["b64_json"]
            if not isinstance(encoded, str) or not encoded:
                raise ValueError("image b64_json must be a non-empty string")
            try:
                result = base64.b64decode(encoded, validate=True)
            except ValueError as exc:
                raise ValueError("image b64_json is invalid") from exc
            if not result:
                raise ValueError("image b64_json decoded to empty bytes")
            return result

        image_url = image_item["url"]
        if not isinstance(image_url, str) or not image_url.strip():
            raise ValueError("image url must be a non-empty string")
        return await self._download_image_asset(image_url)

    async def _download_image_asset(self, asset_value: str) -> bytes:
        asset_value = asset_value.strip()
        if not asset_value:
            raise ValueError("图片资源为空")

        if asset_value.startswith("data:image/"):
            header, separator, encoded = asset_value.partition(",")
            if not separator or not header.endswith(";base64") or not encoded:
                raise ValueError("image data URL is invalid")
            try:
                result = base64.b64decode(encoded, validate=True)
            except ValueError as exc:
                raise ValueError("image data URL contains invalid base64") from exc
            if not result:
                raise ValueError("image data URL decoded to empty bytes")
            return result

        parsed = urlparse(asset_value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("image URL must be an absolute HTTP URL")
        async with httpx.AsyncClient(
            **build_httpx_kwargs(asset_value, self._timeout)
        ) as http_client:
            response = await http_client.get(asset_value)
            response.raise_for_status()
            if not response.content:
                raise ValueError("downloaded image is empty")
            return response.content
