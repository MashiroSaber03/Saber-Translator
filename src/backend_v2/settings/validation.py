"""Closed backend validation for persisted settings and provider facts."""

from __future__ import annotations

from copy import deepcopy
import math
import re
from typing import Any, Mapping

from src.backend_v2.serialization import canonical_json
from src.backend_v2.storage.defaults import default_translation_settings
from src.shared.ai_providers import (
    CHAT_CAPABILITY,
    EMBEDDING_CAPABILITY,
    HQ_TRANSLATION_CAPABILITY,
    IMAGE_GEN_CAPABILITY,
    PLUGIN_AGENT_CAPABILITY,
    RERANK_CAPABILITY,
    TRANSLATION_CAPABILITY,
    VISION_OCR_CAPABILITY,
    VLM_CAPABILITY,
    WEB_IMPORT_AGENT_CAPABILITY,
    get_provider_manifest,
    provider_supports_capability,
)


APP_SETTING_DOMAINS = frozenset(
    {
        "translation",
        "text_style_defaults",
        "workflow_preferences",
        "web_import",
        "insight",
        "ocr",
        "detection",
        "hq",
        "proofreading",
        "misc",
        "inpainting",
        "rendering",
    }
)
APP_SETTING_SCHEMA_VERSIONS = {
    domain: (3 if domain == "translation" else 1)
    for domain in APP_SETTING_DOMAINS
}
PROVIDER_SETTING_SCHEMA_VERSION = 1
BOOK_SETTING_SCHEMA_VERSION = 1
PROVIDER_CAPABILITIES = {
    "translation": TRANSLATION_CAPABILITY,
    "hq": HQ_TRANSLATION_CAPABILITY,
    "plugin_agent": PLUGIN_AGENT_CAPABILITY,
    "ai_vision_ocr": VISION_OCR_CAPABILITY,
    "web_import_agent": WEB_IMPORT_AGENT_CAPABILITY,
    "insight_vlm": VLM_CAPABILITY,
    "insight_chat": CHAT_CAPABILITY,
    "insight_embedding": EMBEDDING_CAPABILITY,
    "insight_reranker": RERANK_CAPABILITY,
    "insight_image_gen": IMAGE_GEN_CAPABILITY,
}
_PROOFREADING_DOMAIN = re.compile(r"^proofreading_\d+$")
_SECRET_KEYS = frozenset(
    {
        "apikey",
        "apikeys",
        "secretkey",
        "secret",
        "token",
        "password",
        "cookie",
        "headers",
        "credentialversionid",
    }
)
_WORKFLOW_MODES = frozenset(
    {
        "translate-current",
        "translate-batch",
        "hq-batch",
        "proofread-batch",
        "remove-current",
        "remove-batch",
        "retry-failed",
        "delete-current",
        "clear-all",
    }
)
_PROVIDER_PAYLOAD_FIELDS = frozenset(
    {
        "modelName",
        "customBaseUrl",
        "openaiOptions",
        "prompt",
        "translationMode",
        "promptMode",
        "batchSize",
        "minImageSize",
        "imageMaxSize",
        "rpmLimit",
        "topK",
        "transportRetries",
        "businessRetries",
        "timeoutSeconds",
        "version",
        "sourceLanguage",
    }
)


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return deepcopy(dict(value))


def _bounded_json(value: object, label: str, maximum: int = 512 * 1024) -> None:
    encoded = canonical_json(value).encode("utf-8")
    if len(encoded) > maximum:
        raise ValueError(f"{label} exceeds {maximum} bytes")


def _reject_secret_fields(value: object, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
            if normalized in _SECRET_KEYS:
                raise ValueError(f"{path} must not contain secret field {key}")
            _reject_secret_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_secret_fields(child, f"{path}[{index}]")


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str] | frozenset[str],
    path: str,
) -> None:
    actual = set(value)
    if actual != set(expected):
        unknown = sorted(actual - set(expected))
        missing = sorted(set(expected) - actual)
        detail = []
        if unknown:
            detail.append("unknown=" + ",".join(unknown))
        if missing:
            detail.append("missing=" + ",".join(missing))
        raise ValueError(f"{path} has invalid fields ({'; '.join(detail)})")


def _finite_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be a finite number")
    return result


def _integer(
    value: object,
    path: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{path} must be an integer from {minimum} to {maximum}")
    return value


def _validate_openai_options(
    value: object,
    path: str,
    *,
    wire_format: bool = False,
) -> None:
    options = _object(value, path)
    _exact_keys(options, {"request", "execution"}, path)
    request = _object(options["request"], f"{path}.request")
    force_json_key = "force_json_output" if wire_format else "forceJsonOutput"
    extra_body_key = "extra_body" if wire_format else "extraBody"
    allowed_request = {force_json_key, "temperature", extra_body_key}
    unknown = set(request) - allowed_request
    if unknown or force_json_key not in request:
        raise ValueError(f"{path}.request has invalid fields")
    if not isinstance(request[force_json_key], bool):
        raise ValueError(f"{path}.request.{force_json_key} must be boolean")
    if "temperature" in request:
        temperature = _finite_number(
            request["temperature"],
            f"{path}.request.temperature",
        )
        if not 0 <= temperature <= 2:
            raise ValueError(f"{path}.request.temperature must be from 0 to 2")
    if extra_body_key in request and not isinstance(
        request[extra_body_key],
        Mapping,
    ):
        raise ValueError(f"{path}.request.{extra_body_key} must be an object")
    execution = _object(options["execution"], f"{path}.execution")
    use_stream_key = "use_stream" if wire_format else "useStream"
    rpm_limit_key = "rpm_limit" if wire_format else "rpmLimit"
    transport_retries_key = (
        "transport_retries" if wire_format else "transportRetries"
    )
    business_retries_key = (
        "business_retries" if wire_format else "businessRetries"
    )
    _exact_keys(
        execution,
        {
            use_stream_key,
            rpm_limit_key,
            transport_retries_key,
            business_retries_key,
        },
        f"{path}.execution",
    )
    if not isinstance(execution[use_stream_key], bool):
        raise ValueError(f"{path}.execution.{use_stream_key} must be boolean")
    _integer(
        execution[rpm_limit_key],
        f"{path}.execution.{rpm_limit_key}",
        minimum=0,
        maximum=100_000,
    )
    _integer(
        execution[transport_retries_key],
        f"{path}.execution.{transport_retries_key}",
        minimum=0,
        maximum=100,
    )
    _integer(
        execution[business_retries_key],
        f"{path}.execution.{business_retries_key}",
        minimum=0,
        maximum=100,
    )


def _validate_shape(value: object, template: object, path: str) -> None:
    if path.endswith("openaiOptions"):
        _validate_openai_options(value, path)
        return
    if isinstance(template, Mapping):
        current = _object(value, path)
        _exact_keys(current, set(template), path)
        for key, child_template in template.items():
            _validate_shape(current[key], child_template, f"{path}.{key}")
        return
    if isinstance(template, list):
        if not isinstance(value, list):
            raise ValueError(f"{path} must be an array")
        return
    if isinstance(template, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{path} must be boolean")
        return
    if isinstance(template, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{path} must be an integer")
        return
    if isinstance(template, float):
        _finite_number(value, path)
        return
    if isinstance(template, str) and not isinstance(value, str):
        raise ValueError(f"{path} must be a string")


def _require_provider(value: object, capability: str, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a provider ID")
    manifest = get_provider_manifest(value)
    if not provider_supports_capability(manifest.id, capability):
        raise ValueError(f"{path} provider does not support {capability}")
    return manifest.id


def _validate_proofreading_rounds(payload: Mapping[str, object]) -> None:
    proofreading = _object(payload["proofreading"], "translation.proofreading")
    rounds = proofreading["rounds"]
    if not isinstance(rounds, list) or len(rounds) > 20:
        raise ValueError("translation.proofreading.rounds must contain at most 20 items")
    expected = {
        "name",
        "provider",
        "modelName",
        "customBaseUrl",
        "openaiOptions",
        "batchSize",
        "prompt",
    }
    for index, value in enumerate(rounds):
        path = f"translation.proofreading.rounds[{index}]"
        round_config = _object(value, path)
        _exact_keys(round_config, expected, path)
        _require_provider(
            round_config["provider"],
            HQ_TRANSLATION_CAPABILITY,
            f"{path}.provider",
        )
        for key in ("name", "modelName", "customBaseUrl", "prompt"):
            if not isinstance(round_config[key], str):
                raise ValueError(f"{path}.{key} must be a string")
        _validate_openai_options(round_config["openaiOptions"], f"{path}.openaiOptions")
        _integer(round_config["batchSize"], f"{path}.batchSize", minimum=1, maximum=10)


def _validate_translation(payload: dict[str, Any], schema_version: int) -> None:
    if payload.get("settingsSchemaVersion") != schema_version:
        raise ValueError("translation settings schema version must be 3")
    _validate_shape(payload, default_translation_settings(), "translation")
    _validate_proofreading_rounds(payload)
    if payload["ocrEngine"] not in {
        "manga_ocr",
        "paddle_ocr",
        "paddleocr_vl",
        "baidu_ocr",
        "ai_vision",
        "48px_ocr",
    }:
        raise ValueError("translation.ocrEngine is invalid")
    if payload["textDetector"] not in {"ctd", "yolo", "default"}:
        raise ValueError("translation.textDetector is invalid")
    if payload["translation"]["translationMode"] not in {"batch", "single"}:
        raise ValueError("translation.translationMode is invalid")
    if payload["aiVisionOcr"]["promptMode"] not in {
        "normal",
        "json",
        "paddleocr_vl",
    }:
        raise ValueError("translation.aiVisionOcr.promptMode is invalid")
    for key in ("auxYoloConfThreshold", "auxYoloOverlapThreshold"):
        value = _finite_number(payload[key], f"translation.{key}")
        if not 0 <= value <= 1:
            raise ValueError(f"translation.{key} must be from 0 to 1")
    refine_threshold = _finite_number(
        payload["saberYoloRefineOverlapThreshold"],
        "translation.saberYoloRefineOverlapThreshold",
    )
    if not 0 <= refine_threshold <= 100:
        raise ValueError(
            "translation.saberYoloRefineOverlapThreshold must be from 0 to 100"
        )
    _require_provider(
        payload["translation"]["provider"],
        TRANSLATION_CAPABILITY,
        "translation.translation.provider",
    )
    _require_provider(
        payload["hqTranslation"]["provider"],
        HQ_TRANSLATION_CAPABILITY,
        "translation.hqTranslation.provider",
    )
    _require_provider(
        payload["pluginAgent"]["provider"],
        PLUGIN_AGENT_CAPABILITY,
        "translation.pluginAgent.provider",
    )
    _require_provider(
        payload["aiVisionOcr"]["provider"],
        VISION_OCR_CAPABILITY,
        "translation.aiVisionOcr.provider",
    )
    _integer(
        payload["parallel"]["deepLearningLockSize"],
        "translation.parallel.deepLearningLockSize",
        minimum=1,
        maximum=4,
    )
    _integer(
        payload["hqTranslation"]["batchSize"],
        "translation.hqTranslation.batchSize",
        minimum=1,
        maximum=10,
    )
    _integer(
        payload["proofreading"]["maxRetries"],
        "translation.proofreading.maxRetries",
        minimum=0,
        maximum=10,
    )


def _validate_workflow_preferences(payload: dict[str, Any]) -> None:
    _exact_keys(
        payload,
        {"rememberWorkflowModeEnabled", "lastWorkflowMode"},
        "workflow_preferences",
    )
    if not isinstance(payload["rememberWorkflowModeEnabled"], bool):
        raise ValueError("rememberWorkflowModeEnabled must be boolean")
    if payload["lastWorkflowMode"] not in _WORKFLOW_MODES:
        raise ValueError("lastWorkflowMode is invalid")


def _validate_web_import(payload: dict[str, Any]) -> None:
    expected = {
        "firecrawl",
        "agent",
        "extraction",
        "download",
        "imagePreprocess",
        "advanced",
        "ui",
    }
    _exact_keys(payload, expected, "web_import")
    nested_fields = {
        "firecrawl": set(),
        "agent": {
            "provider",
            "customBaseUrl",
            "modelName",
            "useStream",
            "forceJsonOutput",
            "maxRetries",
            "timeout",
        },
        "extraction": {"prompt", "maxIterations"},
        "download": {"concurrency", "timeout", "retries", "delay", "useReferer"},
        "advanced": {"bypassProxy"},
        "ui": {"showAgentLogs", "autoImport"},
    }
    for key, fields in nested_fields.items():
        section = _object(payload[key], f"web_import.{key}")
        _exact_keys(section, fields, f"web_import.{key}")
    preprocess = _object(payload["imagePreprocess"], "web_import.imagePreprocess")
    _exact_keys(
        preprocess,
        {"enabled", "autoRotate", "compression", "formatConvert"},
        "web_import.imagePreprocess",
    )
    compression = _object(
        preprocess["compression"],
        "web_import.imagePreprocess.compression",
    )
    _exact_keys(
        compression,
        {"enabled", "quality", "maxWidth", "maxHeight"},
        "web_import.imagePreprocess.compression",
    )
    conversion = _object(
        preprocess["formatConvert"],
        "web_import.imagePreprocess.formatConvert",
    )
    _exact_keys(
        conversion,
        {"enabled", "targetFormat"},
        "web_import.imagePreprocess.formatConvert",
    )
    agent = payload["agent"]
    assert isinstance(agent, Mapping)
    _require_provider(
        agent["provider"],
        WEB_IMPORT_AGENT_CAPABILITY,
        "web_import.agent.provider",
    )
    if conversion["targetFormat"] not in {"jpeg", "png", "webp", "original"}:
        raise ValueError("web_import targetFormat is invalid")
    for path, value, minimum, maximum in (
        ("agent.maxRetries", agent["maxRetries"], 0, 100),
        ("agent.timeout", agent["timeout"], 1, 3600),
        ("extraction.maxIterations", payload["extraction"]["maxIterations"], 1, 100),
        ("download.concurrency", payload["download"]["concurrency"], 1, 32),
        ("download.timeout", payload["download"]["timeout"], 1, 3600),
        ("download.retries", payload["download"]["retries"], 0, 100),
        ("download.delay", payload["download"]["delay"], 0, 60_000),
        ("compression.quality", compression["quality"], 1, 100),
        ("compression.maxWidth", compression["maxWidth"], 0, 100_000),
        ("compression.maxHeight", compression["maxHeight"], 0, 100_000),
    ):
        _integer(value, f"web_import.{path}", minimum=minimum, maximum=maximum)


def _validate_insight(payload: dict[str, Any]) -> None:
    _exact_keys(
        payload,
        {"analysis", "vlm", "chat", "embedding", "reranker", "imageGen"},
        "insight",
    )
    analysis = _object(payload["analysis"], "insight.analysis")
    _exact_keys(analysis, {"batch"}, "insight.analysis")
    batch = _object(analysis["batch"], "insight.analysis.batch")
    _exact_keys(
        batch,
        {
            "pagesPerBatch",
            "contextBatchCount",
            "architecturePreset",
            "customLayers",
        },
        "insight.analysis.batch",
    )
    _integer(
        batch["pagesPerBatch"],
        "insight.analysis.batch.pagesPerBatch",
        minimum=1,
        maximum=20,
    )
    _integer(
        batch["contextBatchCount"],
        "insight.analysis.batch.contextBatchCount",
        minimum=0,
        maximum=10,
    )
    if batch["architecturePreset"] not in {
        "simple",
        "standard",
        "chapter_based",
        "full",
        "custom",
    }:
        raise ValueError("insight architecturePreset is invalid")
    layers = batch["customLayers"]
    if not isinstance(layers, list) or len(layers) > 8:
        raise ValueError("insight customLayers must contain at most 8 items")
    if batch["architecturePreset"] == "custom" and len(layers) < 2:
        raise ValueError("custom Insight architecture must contain 2-8 layers")
    for index, layer_value in enumerate(layers):
        layer = _object(
            layer_value,
            f"insight.analysis.batch.customLayers[{index}]",
        )
        _exact_keys(
            layer,
            {"name", "unitsPerGroup", "alignToChapter"},
            f"insight.analysis.batch.customLayers[{index}]",
        )
        if not isinstance(layer["name"], str) or not layer["name"].strip():
            raise ValueError("insight custom layer name is required")
        _integer(
            layer["unitsPerGroup"],
            "insight custom layer unitsPerGroup",
            minimum=0,
            maximum=100,
        )
        if not isinstance(layer["alignToChapter"], bool):
            raise ValueError("insight custom layer alignToChapter must be boolean")
    capability_by_key = {
        "vlm": VLM_CAPABILITY,
        "chat": CHAT_CAPABILITY,
        "embedding": EMBEDDING_CAPABILITY,
        "reranker": RERANK_CAPABILITY,
        "imageGen": IMAGE_GEN_CAPABILITY,
    }
    for key, capability in capability_by_key.items():
        section = _object(payload[key], f"insight.{key}")
        expected = {"provider", "useSameAsVlm"} if key == "chat" else {"provider"}
        _exact_keys(section, expected, f"insight.{key}")
        provider = section["provider"]
        if provider:
            _require_provider(provider, capability, f"insight.{key}.provider")
        elif not isinstance(provider, str):
            raise ValueError(f"insight.{key}.provider must be a string")
        if key == "chat" and not isinstance(section["useSameAsVlm"], bool):
            raise ValueError("insight.chat.useSameAsVlm must be boolean")


def validate_setting_payload(
    domain: str,
    payload: object,
    *,
    schema_version: int,
) -> dict[str, Any]:
    if domain not in APP_SETTING_DOMAINS:
        raise ValueError(f"unsupported setting domain: {domain}")
    expected_schema_version = APP_SETTING_SCHEMA_VERSIONS[domain]
    if schema_version != expected_schema_version:
        raise ValueError(
            f"{domain} settings schema version must be "
            f"{expected_schema_version}"
        )
    result = _object(payload, f"{domain} setting")
    _bounded_json(result, f"{domain} setting")
    _reject_secret_fields(result, domain)
    if domain == "translation":
        _validate_translation(result, schema_version)
    elif domain == "workflow_preferences":
        _validate_workflow_preferences(result)
    elif domain == "web_import":
        _validate_web_import(result)
    elif domain == "insight":
        _validate_insight(result)
    elif domain == "proofreading":
        if set(result) - {"enabled"} or (
            "enabled" in result and not isinstance(result["enabled"], bool)
        ):
            raise ValueError("proofreading setting is invalid")
    return result


def validate_provider_setting_payload(
    domain: str,
    provider: str,
    payload: object,
    *,
    schema_version: int,
) -> dict[str, Any]:
    if schema_version != PROVIDER_SETTING_SCHEMA_VERSION:
        raise ValueError(
            "provider setting schema version must be "
            f"{PROVIDER_SETTING_SCHEMA_VERSION}"
        )
    result = _object(payload, "provider setting payload")
    _bounded_json(result, "provider setting payload", 256 * 1024)
    _reject_secret_fields(result, f"provider_settings.{domain}.{provider}")
    if domain in {"web_import_firecrawl", "web_import_http"}:
        expected_provider = "firecrawl" if domain == "web_import_firecrawl" else "headers"
        if provider != expected_provider or result:
            raise ValueError(f"{domain} provider setting is invalid")
        return result
    capability = PROVIDER_CAPABILITIES.get(domain)
    if capability is None and _PROOFREADING_DOMAIN.fullmatch(domain):
        capability = HQ_TRANSLATION_CAPABILITY
    if capability is None:
        if domain == "ocr" and provider == "baidu":
            allowed = {"version", "sourceLanguage"}
            if set(result) - allowed:
                raise ValueError("ocr provider setting has unknown fields")
            return result
        raise ValueError(f"unsupported provider setting domain: {domain}")
    _require_provider(provider, capability, f"provider_settings.{domain}.provider")
    unknown = set(result) - _PROVIDER_PAYLOAD_FIELDS
    if unknown:
        raise ValueError(
            "provider setting has unknown fields: " + ", ".join(sorted(unknown))
        )
    if "openaiOptions" in result:
        _validate_openai_options(
            result["openaiOptions"],
            f"provider_settings.{domain}.openaiOptions",
            wire_format=domain in {"insight_vlm", "insight_chat"},
        )
    for key, value in result.items():
        if key == "openaiOptions":
            continue
        if key in {
            "batchSize",
            "minImageSize",
            "imageMaxSize",
            "rpmLimit",
            "topK",
            "transportRetries",
            "businessRetries",
            "timeoutSeconds",
        }:
            _finite_number(value, f"provider_settings.{domain}.{key}")
        elif not isinstance(value, str):
            raise ValueError(f"provider_settings.{domain}.{key} must be a string")
    return result


def validate_credential_secret(
    domain: str,
    provider: str,
    secret: object,
) -> dict[str, Any]:
    result = _object(secret, "credential secret")
    if domain == "ocr" and provider == "baidu":
        allowed = {"baidu_api_key", "baidu_secret_key"}
    elif domain == "ai_vision_ocr":
        allowed = {"ai_vision_api_key"}
    elif domain == "web_import_http" and provider == "headers":
        allowed = {"cookie", "headers"}
    elif (
        domain in PROVIDER_CAPABILITIES
        or _PROOFREADING_DOMAIN.fullmatch(domain)
        or domain == "web_import_firecrawl"
    ):
        allowed = {"api_key"}
    else:
        raise ValueError(f"unsupported credential domain/provider: {domain}/{provider}")
    if domain == "web_import_http":
        if not result or set(result) - allowed:
            raise ValueError(
                "web import HTTP credential must contain cookie and/or headers"
            )
    elif set(result) != allowed:
        raise ValueError(
            "credential secret fields must be exactly: "
            + ", ".join(sorted(allowed))
        )
    if domain == "web_import_http":
        if (
            "cookie" in result
            and not isinstance(result["cookie"], str)
        ) or (
            "headers" in result
            and not isinstance(result["headers"], Mapping)
        ):
            raise ValueError("web import HTTP credential is invalid")
    elif any(not isinstance(value, str) or not value for value in result.values()):
        raise ValueError("credential secret values must be non-empty strings")
    _bounded_json(result, "credential secret", 128 * 1024)
    return result


def validate_book_setting_payload(
    domain: str,
    payload: object,
    *,
    schema_version: int,
) -> dict[str, Any]:
    if schema_version != BOOK_SETTING_SCHEMA_VERSION:
        raise ValueError(
            "book setting schema version must be "
            f"{BOOK_SETTING_SCHEMA_VERSION}"
        )
    if domain != "insight":
        raise ValueError(f"unsupported book setting domain: {domain}")
    result = _object(payload, "book setting payload")
    _bounded_json(result, "book setting payload")
    _reject_secret_fields(result, f"book_settings.{domain}")
    _validate_insight(result)
    return result
