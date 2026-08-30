"""One-shot DOM image target selection using the shared AI transport."""

from __future__ import annotations

from collections.abc import Mapping
import json
import logging
import math
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy import Engine, select

from src.backend_v2.auth.ownership import effective_owner_id
from src.backend_v2.settings.validation import (
    validate_provider_setting_payload,
    validate_setting_payload,
)
from src.backend_v2.storage.platform_repositories import SettingsRepository
from src.backend_v2.storage.schema import app_settings, provider_settings
from src.shared.ai_providers import (
    PLUGIN_AGENT_CAPABILITY,
    get_provider_manifest,
    normalize_provider_id,
    provider_requires_api_key,
    provider_supports_capability,
)
from src.shared.ai_transport import OpenAICompatibleChatTransport, UnifiedChatRequest
from src.shared.openai_execution import (
    OpenAICompatibleBusinessRetryableError,
    OpenAICompatibleSyncExecutor,
    build_openai_compatible_runtime_options,
)
from src.shared.openai_options import OpenAICompatibleOptions


LOGGER = logging.getLogger("BrowserDomAgent")
MAX_DOM_NODES = 600
NODE_FIELDS = frozenset(
    {"id", "tag", "classes", "parent", "attributes", "rect", "naturalSize"}
)
NODE_ATTRIBUTE_FIELDS = frozenset(
    {
        "alt",
        "role",
        "aria-label",
        "data-src",
        "data-original",
        "data-page",
        "loading",
    }
)
SOURCE_ATTRIBUTE_PATTERN = re.compile(
    r"^(?:blob:image|image-source-present|data:image/[a-z0-9.+-]{1,40}|"
    r"image-url(?:\.[a-z0-9]{2,8})?)$",
    re.IGNORECASE,
)


class BrowserDomAgentUnavailable(RuntimeError):
    pass


def _object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return dict(value)


def _openai_options(value: object) -> OpenAICompatibleOptions:
    raw = _object(value, "browserDomAgent.openaiOptions")
    if set(raw) != {"request", "execution"}:
        raise ValueError("Browser DOM Agent OpenAI options are invalid")
    request = _object(raw["request"], "browserDomAgent.openaiOptions.request")
    execution = _object(
        raw["execution"],
        "browserDomAgent.openaiOptions.execution",
    )
    return OpenAICompatibleOptions.from_dict(
        {
            "request": {
                "force_json_output": request.get("forceJsonOutput", False),
                "temperature": request.get("temperature"),
                "extra_body": request.get("extraBody", {}),
            },
            "execution": {
                "use_stream": execution.get("useStream", False),
                "rpm_limit": execution.get("rpmLimit", 0),
                "transport_retries": execution.get("transportRetries", 1),
                "business_retries": execution.get("businessRetries", 1),
            },
        }
    )


class BrowserDomAgentProviderResolver:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.settings = SettingsRepository(engine)

    def runtime_config(self) -> dict[str, Any]:
        owner = effective_owner_id()
        with self.engine.connect() as connection:
            app_row = connection.execute(
                select(
                    app_settings.c.payload_json,
                    app_settings.c.schema_version,
                ).where(
                    app_settings.c.domain == "translation",
                    app_settings.c.owner_user_id == owner,
                )
            ).mappings().one_or_none()
            if app_row is None:
                raise ValueError("translation settings are missing")
            translation = validate_setting_payload(
                "translation",
                json.loads(app_row["payload_json"]),
                schema_version=int(app_row["schema_version"]),
            )
            selected_settings = dict(translation["browserDomAgent"])
            selected = str(selected_settings["provider"])
            provider_row = connection.execute(
                select(provider_settings).where(
                    provider_settings.c.owner_user_id == owner,
                    provider_settings.c.domain == "browser_dom_agent",
                    provider_settings.c.provider == selected,
                )
            ).mappings().one_or_none()
        if provider_row is None:
            raise BrowserDomAgentUnavailable(
                "请先在设置的“网页漫画”中保存 Browser DOM Agent 配置"
            )
        payload = validate_provider_setting_payload(
            "browser_dom_agent",
            selected,
            json.loads(provider_row["payload_json"]),
            schema_version=int(provider_row["schema_version"]),
        )
        selected_settings.update(payload)
        api_key = ""
        credential_version_id = provider_row["credential_version_id"]
        if credential_version_id is not None:
            secret = self.settings.resolve_secret(str(credential_version_id))
            if set(secret) != {"api_key"} or not isinstance(secret["api_key"], str):
                raise ValueError("Browser DOM Agent credential is invalid")
            api_key = secret["api_key"]
        return {
            "provider": selected,
            "credential_version_id": (
                str(credential_version_id)
                if credential_version_id is not None
                else None
            ),
            "api_key": api_key,
            "model_name": str(selected_settings["modelName"]),
            "custom_base_url": str(selected_settings["customBaseUrl"]),
            "openai_options": _openai_options(selected_settings["openaiOptions"]),
        }


class BrowserDomAgentService:
    def __init__(
        self,
        engine: Engine,
        *,
        resolver: BrowserDomAgentProviderResolver | None = None,
        transport: OpenAICompatibleChatTransport | None = None,
    ) -> None:
        self.resolver = resolver or BrowserDomAgentProviderResolver(engine)
        self.executor = OpenAICompatibleSyncExecutor(
            transport or OpenAICompatibleChatTransport()
        )

    def detect(self, payload: dict[str, Any]) -> dict[str, object]:
        normalized = self._normalize_payload(payload)
        config = self.resolver.runtime_config()
        provider = normalize_provider_id(config["provider"])
        custom_base_url = config["custom_base_url"] or None
        manifest = get_provider_manifest(provider)
        if not provider_supports_capability(provider, PLUGIN_AGENT_CAPABILITY):
            raise BrowserDomAgentUnavailable(
                f"{manifest.display_name} 不支持 Browser DOM Agent"
            )
        if provider_requires_api_key(provider, custom_base_url) and not config["api_key"]:
            raise BrowserDomAgentUnavailable(f"{manifest.display_name} 需要 API Key")
        if manifest.requires_model and not config["model_name"]:
            raise BrowserDomAgentUnavailable(f"{manifest.display_name} 需要模型名称")
        if manifest.requires_base_url and not custom_base_url:
            raise BrowserDomAgentUnavailable(f"{manifest.display_name} 需要 Base URL")
        messages = [
            {
                "role": "system",
                "content": (
                    "你负责从已经脱敏的网页图片节点摘要中选择漫画正文图片。"
                    "排除头像、图标、广告、导航、封面推荐和重复缩略图。"
                    "优先选择尺寸大、连续出现、父级结构相似的漫画页。"
                    "只能返回 JSON 对象，字段必须严格为 nodeIds、selector、"
                    "confidence、reason。nodeIds 只能使用输入中的 id；selector"
                    "可以为空字符串；confidence 为 0 到 1。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(normalized, ensure_ascii=False),
            },
        ]
        result = self.executor.execute(
            UnifiedChatRequest(
                provider=provider,
                api_key=config["api_key"],
                model=config["model_name"],
                credential_version_id=config["credential_version_id"],
                base_url=custom_base_url,
                capability=PLUGIN_AGENT_CAPABILITY,
                openai_options=config["openai_options"],
                runtime_options=build_openai_compatible_runtime_options(
                    timeout=90.0,
                    stream_output_label="Browser DOM Agent",
                ),
                messages=messages,
            ),
            capability=PLUGIN_AGENT_CAPABILITY,
            parser=lambda content: self._parse_result(
                content,
                allowed_ids={node["id"] for node in normalized["nodes"]},
            ),
            logger_instance=LOGGER,
        )
        return result.parsed

    @staticmethod
    def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
        if set(payload) != {"pageUrl", "pageTitle", "nodes"}:
            raise ValueError("Browser DOM Agent request fields are invalid")
        page_url = payload["pageUrl"]
        page_title = payload["pageTitle"]
        nodes = payload["nodes"]
        if not isinstance(page_url, str) or not page_url.startswith(
            ("http://", "https://")
        ):
            raise ValueError("pageUrl must be an HTTP(S) URL")
        parsed_page_url = urlsplit(page_url)
        if not parsed_page_url.hostname:
            raise ValueError("pageUrl must be an HTTP(S) URL")
        try:
            port = f":{parsed_page_url.port}" if parsed_page_url.port else ""
        except ValueError as error:
            raise ValueError("pageUrl port is invalid") from error
        redacted_page_url = urlunsplit(
            (
                parsed_page_url.scheme,
                f"{parsed_page_url.hostname}{port}",
                parsed_page_url.path,
                "",
                "",
            )
        )
        if not isinstance(page_title, str) or len(page_title) > 500:
            raise ValueError("pageTitle must be a string of at most 500 characters")
        if not isinstance(nodes, list) or not nodes or len(nodes) > MAX_DOM_NODES:
            raise ValueError(f"nodes must contain 1-{MAX_DOM_NODES} items")
        normalized_nodes: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, raw in enumerate(nodes):
            node = _object(raw, f"nodes[{index}]")
            if set(node) != NODE_FIELDS:
                raise ValueError(f"nodes[{index}] fields are invalid")
            node_id = node["id"]
            if (
                not isinstance(node_id, str)
                or not node_id
                or len(node_id) > 100
                or node_id in seen
            ):
                raise ValueError(f"nodes[{index}].id is invalid")
            seen.add(node_id)
            tag = node["tag"]
            classes = node["classes"]
            parent = node["parent"]
            attributes = node["attributes"]
            rect = node["rect"]
            natural_size = node["naturalSize"]
            if not isinstance(tag, str) or len(tag) > 32:
                raise ValueError(f"nodes[{index}].tag is invalid")
            if (
                not isinstance(classes, list)
                or len(classes) > 20
                or any(
                    not isinstance(value, str) or len(value) > 100
                    for value in classes
                )
            ):
                raise ValueError(f"nodes[{index}].classes is invalid")
            if not isinstance(parent, str) or len(parent) > 500:
                raise ValueError(f"nodes[{index}].parent is invalid")
            if not isinstance(attributes, Mapping) or any(
                key not in NODE_ATTRIBUTE_FIELDS
                or not isinstance(key, str)
                or not isinstance(value, str)
                or len(value) > 500
                for key, value in attributes.items()
            ):
                raise ValueError(f"nodes[{index}].attributes is invalid")
            if any(
                key in {"data-src", "data-original"}
                and SOURCE_ATTRIBUTE_PATTERN.fullmatch(value) is None
                for key, value in attributes.items()
            ):
                raise ValueError(
                    f"nodes[{index}].source attributes are not sanitized"
                )
            for value, field, required_fields in (
                (rect, "rect", {"width", "height", "top", "left"}),
                (natural_size, "naturalSize", {"width", "height"}),
            ):
                if (
                    not isinstance(value, Mapping)
                    or set(value) != required_fields
                    or any(
                        isinstance(number, bool)
                        or not isinstance(number, (int, float))
                        or not math.isfinite(number)
                        or abs(number) > 10_000_000
                        for number in value.values()
                    )
                ):
                    raise ValueError(f"nodes[{index}].{field} is invalid")
            normalized_nodes.append(
                {
                    "id": node_id,
                    "tag": tag.lower(),
                    "classes": classes[:20],
                    "parent": parent,
                    "attributes": dict(attributes),
                    "rect": dict(rect),
                    "naturalSize": dict(natural_size),
                }
            )
        return {
            "pageUrl": redacted_page_url,
            "pageTitle": page_title,
            "nodes": normalized_nodes,
        }

    @staticmethod
    def _parse_result(content: str, *, allowed_ids: set[str]) -> dict[str, object]:
        try:
            parsed = json.loads(content)
        except (TypeError, json.JSONDecodeError) as error:
            raise OpenAICompatibleBusinessRetryableError(
                f"Browser DOM Agent JSON 解析失败: {error}"
            ) from error
        if not isinstance(parsed, Mapping) or set(parsed) != {
            "nodeIds",
            "selector",
            "confidence",
            "reason",
        }:
            raise OpenAICompatibleBusinessRetryableError(
                "Browser DOM Agent 返回字段无效"
            )
        node_ids = parsed["nodeIds"]
        selector = parsed["selector"]
        confidence = parsed["confidence"]
        reason = parsed["reason"]
        if (
            not isinstance(node_ids, list)
            or any(not isinstance(value, str) or value not in allowed_ids for value in node_ids)
            or len(node_ids) != len(set(node_ids))
        ):
            raise OpenAICompatibleBusinessRetryableError(
                "Browser DOM Agent nodeIds 无效"
            )
        if not isinstance(selector, str) or len(selector) > 1_000:
            raise OpenAICompatibleBusinessRetryableError(
                "Browser DOM Agent selector 无效"
            )
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= float(confidence) <= 1
        ):
            raise OpenAICompatibleBusinessRetryableError(
                "Browser DOM Agent confidence 无效"
            )
        if not isinstance(reason, str) or len(reason) > 500:
            raise OpenAICompatibleBusinessRetryableError(
                "Browser DOM Agent reason 无效"
            )
        return {
            "nodeIds": node_ids,
            "selector": selector,
            "confidence": float(confidence),
            "reason": reason,
        }
