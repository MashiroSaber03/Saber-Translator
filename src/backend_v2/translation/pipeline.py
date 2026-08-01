"""Durable per-page translation steps executed exclusively by the Worker."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import base64
import hashlib
from io import BytesIO
import json
import logging
from pathlib import Path
import re
from typing import Any, Protocol
import uuid

from PIL import Image
from sqlalchemy import Engine, delete, insert, select, update
from sqlalchemy.engine import Connection

from src.backend_v2.serialization import canonical_json as _json
from src.backend_v2.timestamps import utcnow
from src.backend_v2.content.page_style import rgb_to_hex, validate_page_style
from src.backend_v2.content.translation_constraints import (
    validate_translation_constraints,
    with_glossary_delta,
)
from src.backend_v2.jobs.repository import AttemptFence, JobConflict, JobQueueRepository
from src.backend_v2.rendering.service import (
    publish_png_asset,
    publish_thumbnail_asset,
)
from src.backend_v2.storage.assets import AssetRecord, AssetStorageService
from src.backend_v2.storage.platform_repositories import SettingsRepository
from src.backend_v2.storage.schema import (
    assets,
    bubbles,
    job_items,
    jobs,
    job_steps,
    job_step_asset_outputs,
    page_assets,
    pages,
    translation_constraints,
)


LOGGER = logging.getLogger("saber.worker.translation")


@dataclass(frozen=True, slots=True)
class PageSnapshot:
    page_id: str
    source_revision: int
    document_revision: int
    render_status: str
    style_defaults: dict[str, Any]
    bubble_ids: tuple[str, ...]
    bubbles: tuple[dict[str, Any], ...]


class TranslationAlgorithms(Protocol):
    def detect(self, image: Image.Image, config: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def ocr(
        self,
        image: Image.Image,
        bubble_payloads: list[dict[str, Any]],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def colors(
        self,
        image: Image.Image,
        bubble_payloads: list[dict[str, Any]],
    ) -> list[Mapping[str, Any]]: ...

    def extract_terms(
        self,
        texts: list[str],
        config: Mapping[str, Any],
        *,
        prompt: str,
    ) -> Mapping[str, Any]: ...

    def translate(
        self,
        texts: list[str],
        config: Mapping[str, Any],
        *,
        mode: str,
    ) -> Mapping[str, Any]: ...

    def translate_batch(
        self,
        pages: list[Mapping[str, Any]],
        images: list[Image.Image],
        config: Mapping[str, Any],
        *,
        mode: str,
    ) -> Mapping[str, Any]: ...

    def repair(
        self,
        image: Image.Image,
        bubble_payloads: list[dict[str, Any]],
        config: Mapping[str, Any],
        *,
        precise_mask: Image.Image | None = None,
    ) -> Image.Image: ...

    def render(
        self,
        clean_image: Image.Image,
        bubble_payloads: list[dict[str, Any]],
        config: Mapping[str, Any],
    ) -> Image.Image: ...


def _openai_options(value: object):
    from src.shared.openai_options import OpenAICompatibleOptions

    source = dict(value) if isinstance(value, Mapping) else {}
    request = dict(source.get("request", {})) if isinstance(
        source.get("request"), Mapping
    ) else {}
    execution = dict(source.get("execution", {})) if isinstance(
        source.get("execution"), Mapping
    ) else {}
    normalized = {
        "request": {
            "force_json_output": request.get("force_json_output", False),
            "temperature": request.get("temperature"),
            "extra_body": request.get("extra_body", {}),
        },
        "execution": {
            "use_stream": execution.get("use_stream", False),
            "rpm_limit": execution.get("rpm_limit", 0),
            "transport_retries": execution.get("transport_retries", 1),
            "business_retries": execution.get("business_retries", 0),
        },
    }
    return OpenAICompatibleOptions.from_dict(normalized)


def _validate_stable_batch_result(
    payload: object,
    *,
    expected_pages: list[Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("pages"), list):
        raise ValueError("HQ response must be an object containing a pages array")
    expected_by_page = {
        str(page["pageId"]): {
            str(bubble["bubbleId"]): bubble
            for bubble in page.get("bubbles", [])
        }
        for page in expected_pages
    }
    if len(expected_by_page) != len(expected_pages):
        raise ValueError("HQ request contains duplicate pageId values")

    parsed: dict[str, dict[str, str]] = {}
    for page in payload["pages"]:
        if not isinstance(page, Mapping):
            raise ValueError("HQ response page entries must be objects")
        page_id = str(page.get("pageId", ""))
        if not page_id or page_id in parsed:
            raise ValueError("HQ response contains a missing or duplicate pageId")
        if page_id not in expected_by_page:
            raise ValueError(f"HQ response contains unknown pageId: {page_id}")
        raw_bubbles = page.get("bubbles")
        if not isinstance(raw_bubbles, list):
            raise ValueError(f"HQ response page {page_id} has no bubbles array")
        bubble_results: dict[str, str] = {}
        for bubble in raw_bubbles:
            if not isinstance(bubble, Mapping):
                raise ValueError("HQ response bubble entries must be objects")
            bubble_id = str(bubble.get("bubbleId", ""))
            if not bubble_id or bubble_id in bubble_results:
                raise ValueError(
                    f"HQ response page {page_id} has a missing or duplicate bubbleId"
                )
            expected_bubble = expected_by_page[page_id].get(bubble_id)
            if expected_bubble is None:
                raise ValueError(
                    f"HQ response page {page_id} contains unknown bubbleId: {bubble_id}"
                )
            translated = bubble.get("translatedText")
            if not isinstance(translated, str):
                raise ValueError(
                    f"HQ response bubble {bubble_id} translatedText must be a string"
                )
            if (
                str(expected_bubble.get("originalText", "")).strip()
                or str(expected_bubble.get("translatedText", "")).strip()
            ) and not translated.strip():
                raise ValueError(
                    f"HQ response bubble {bubble_id} returned an empty translation"
                )
            bubble_results[bubble_id] = translated
        if set(bubble_results) != set(expected_by_page[page_id]):
            raise ValueError(
                f"HQ response page {page_id} bubble IDs do not match the request"
            )
        parsed[page_id] = bubble_results
    if set(parsed) != set(expected_by_page):
        raise ValueError("HQ response page IDs do not match the request")
    return parsed


def _batch_input_fingerprint(
    *,
    page_id: str,
    document_revision: int,
    bubbles: list[Mapping[str, Any]],
    mode: str,
    round_index: int | None,
    constraint_context: Mapping[str, Any],
) -> str:
    payload = _json(
        {
            "pageId": page_id,
            "documentRevision": document_revision,
            "bubbles": bubbles,
            "mode": mode,
            "roundIndex": round_index,
            "translationConstraints": constraint_context,
        },
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _matching_fragments(
    text: str,
    *,
    pattern: str,
    match_mode: str,
) -> list[str]:
    if match_mode == "regex":
        return [
            match.group(0)
            for match in re.finditer(pattern, text)
            if match.group(0)
        ]
    return [pattern] if pattern and pattern in text else []


def _protect_non_translate_text(
    text: str,
    entries: list[Mapping[str, Any]],
    *,
    token_prefix: str,
    token_by_fragment: dict[str, str] | None = None,
) -> tuple[str, dict[str, str]]:
    protected = text
    fragment_tokens = token_by_fragment if token_by_fragment is not None else {}
    restore: dict[str, str] = {}
    token_counter = len(fragment_tokens)
    for entry in entries:
        pattern = str(entry.get("pattern", ""))
        match_mode = str(entry.get("matchMode", "text"))
        fragments = _matching_fragments(
            protected,
            pattern=pattern,
            match_mode=match_mode,
        )
        for fragment in fragments:
            token = fragment_tokens.get(fragment)
            if token is None:
                digest = hashlib.sha256(fragment.encode("utf-8")).hexdigest()[:10]
                token = f"⟦SABER_NT_{token_prefix}_{token_counter}_{digest}⟧"
                token_counter += 1
                fragment_tokens[fragment] = token
            protected = protected.replace(fragment, token)
            restore[token] = fragment
    return protected, restore


def _restore_non_translate_text(
    text: str,
    restore: Mapping[str, str],
) -> str:
    restored = text
    for token, fragment in restore.items():
        if token not in restored:
            # Vision models can read the protected fragment directly from the
            # page image and return it verbatim instead of echoing our token.
            # That already satisfies the non-translate constraint, so do not
            # reject an otherwise valid translation.
            if fragment in restored:
                continue
            raise JobConflict(
                f"translation response lost protected non-translate token {token}"
            )
        restored = restored.replace(token, fragment)
    return restored


def _translation_constraint_warnings(
    originals: list[str],
    translated: list[str],
    constraints: Mapping[str, Any],
) -> list[dict[str, Any]]:
    glossary = constraints.get("glossary")
    if not isinstance(glossary, Mapping) or not bool(glossary.get("enabled")):
        return []
    entries = glossary.get("entries")
    if not isinstance(entries, list):
        return []
    warnings: list[dict[str, Any]] = []
    for bubble_index, (source_text, translated_text) in enumerate(
        zip(originals, translated)
    ):
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            source = str(entry.get("source", ""))
            target = str(entry.get("target", ""))
            if (
                _matching_fragments(
                    source_text,
                    pattern=source,
                    match_mode=str(entry.get("matchMode", "text")),
                )
                and target not in translated_text
            ):
                warnings.append(
                    {
                        "bubbleIndex": bubble_index,
                        "source": source,
                        "expectedTarget": target,
                        "actualTranslation": translated_text,
                    }
                )
    return warnings


_DETECTION_TEXT_FIELDS = (
    "originalText",
    "translatedText",
    "textboxText",
)


def _box_iou(left: object, right: object) -> float:
    if (
        not isinstance(left, (list, tuple))
        or not isinstance(right, (list, tuple))
        or len(left) < 4
        or len(right) < 4
    ):
        return 0.0
    try:
        left_x1, left_y1, left_x2, left_y2 = (float(value) for value in left[:4])
        right_x1, right_y1, right_x2, right_y2 = (
            float(value) for value in right[:4]
        )
    except (TypeError, ValueError):
        return 0.0
    left_x1, left_x2 = sorted((left_x1, left_x2))
    left_y1, left_y2 = sorted((left_y1, left_y2))
    right_x1, right_x2 = sorted((right_x1, right_x2))
    right_y1, right_y2 = sorted((right_y1, right_y2))
    intersection_width = max(
        0.0,
        min(left_x2, right_x2) - max(left_x1, right_x1),
    )
    intersection_height = max(
        0.0,
        min(left_y2, right_y2) - max(left_y1, right_y1),
    )
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(
        0.0,
        right_y2 - right_y1,
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _preserve_detected_text(
    detected: list[dict[str, Any]],
    existing: tuple[dict[str, Any], ...],
    *,
    minimum_iou: float = 0.5,
) -> list[dict[str, Any]]:
    """Keep published text while re-detection replaces bubble geometry."""

    unmatched = set(range(len(existing)))
    reconciled: list[dict[str, Any]] = []
    for payload in detected:
        best_index: int | None = None
        best_iou = minimum_iou
        for index in unmatched:
            overlap = _box_iou(
                payload.get("coords"),
                existing[index].get("coords"),
            )
            if overlap >= best_iou:
                best_index = index
                best_iou = overlap
        current = dict(payload)
        if best_index is not None:
            previous = existing[best_index]
            unmatched.remove(best_index)
            for field in _DETECTION_TEXT_FIELDS:
                if field in previous:
                    current[field] = previous[field]
        reconciled.append(current)
    return reconciled


class CoreTranslationAlgorithms:
    """Worker-side adapters around the current core algorithms."""

    def detect(self, image: Image.Image, config: Mapping[str, Any]) -> Mapping[str, Any]:
        from src.core.detection import (
            get_bubble_detection_result_with_auto_directions,
        )

        allowed = {
            "detector_type",
            "expand_ratio",
            "expand_top",
            "expand_bottom",
            "expand_left",
            "expand_right",
            "enable_aux_yolo_detection",
            "aux_yolo_conf_threshold",
            "aux_yolo_overlap_threshold",
            "enable_saber_yolo_refine",
            "saber_yolo_refine_overlap_threshold",
            "min_text_block_area_percent",
        }
        kwargs = {key: value for key, value in config.items() if key in allowed}
        return get_bubble_detection_result_with_auto_directions(image, **kwargs)

    def ocr(
        self,
        image: Image.Image,
        bubble_payloads: list[dict[str, Any]],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        from src.core.ocr import recognize_ocr_results_in_bubbles
        from src.core.ocr_types import (
            extract_texts_from_ocr_results,
            ocr_results_to_dicts,
        )

        coords = [payload.get("coords", [0, 0, 0, 0]) for payload in bubble_payloads]
        textlines = [payload.get("textlines", []) for payload in bubble_payloads]
        allowed = {
            "source_language",
            "ocr_engine",
            "baidu_api_key",
            "baidu_secret_key",
            "baidu_version",
            "baidu_ocr_language",
            "ai_vision_provider",
            "ai_vision_api_key",
            "ai_vision_model_name",
            "ai_vision_ocr_prompt",
            "ai_vision_prompt_mode",
            "custom_ai_vision_base_url",
            "ai_vision_min_image_size",
            "ai_vision_openai_options",
            "credential_version_id",
            "enable_hybrid_ocr",
            "secondary_ocr_engine",
            "hybrid_ocr_threshold",
        }
        kwargs = {key: value for key, value in config.items() if key in allowed}
        kwargs["textlines_per_bubble"] = textlines
        kwargs["strict_errors"] = True
        results = recognize_ocr_results_in_bubbles(image, coords, **kwargs)
        return {
            "texts": extract_texts_from_ocr_results(results),
            "results": ocr_results_to_dicts(results),
        }

    def colors(
        self,
        image: Image.Image,
        bubble_payloads: list[dict[str, Any]],
    ) -> list[Mapping[str, Any]]:
        from src.core.color_extractor import extract_bubble_colors

        coords = [payload.get("coords", [0, 0, 0, 0]) for payload in bubble_payloads]
        textlines = [payload.get("textlines", []) for payload in bubble_payloads]
        # ``extract_bubble_colors`` already returns serialized dictionaries.
        # Detach each mapping from the extractor-owned result.
        return [
            dict(result)
            for result in extract_bubble_colors(image, coords, textlines)
        ]

    def extract_terms(
        self,
        texts: list[str],
        config: Mapping[str, Any],
        *,
        prompt: str,
    ) -> Mapping[str, Any]:
        from src.shared.ai_providers import TRANSLATION_CAPABILITY
        from src.shared.ai_transport import UnifiedChatRequest
        from src.shared.openai_execution import (
            OpenAICompatibleBusinessRetryableError,
            OpenAICompatibleSyncExecutor,
            build_openai_compatible_runtime_options,
            parse_json_block_from_text,
        )

        rendered_prompt = (
            prompt.replace(
                "{ocr_text}",
                json.dumps(texts, ensure_ascii=False, separators=(",", ":")),
            )
            if "{ocr_text}" in prompt
            else (
                f"{prompt.rstrip()}\n\nOCR 文本：\n"
                f"{json.dumps(texts, ensure_ascii=False, separators=(',', ':'))}"
            )
        )
        options = _openai_options(config.get("openai_options"))
        options.request.force_json_output = False

        def parse_terms(raw: str) -> list[Mapping[str, Any]]:
            parsed = parse_json_block_from_text(raw)
            if isinstance(parsed, Mapping):
                parsed = parsed.get("terms")
            if not isinstance(parsed, list) or any(
                not isinstance(entry, Mapping) for entry in parsed
            ):
                raise OpenAICompatibleBusinessRetryableError(
                    "自动术语提取必须返回 JSON 数组或包含 terms 数组的对象"
                )
            return [dict(entry) for entry in parsed]

        request = UnifiedChatRequest(
            provider=str(config["provider"]),
            api_key=str(config.get("api_key", "")),
            model=str(config.get("model_name", "")),
            credential_version_id=(
                str(config["credential_version_id"])
                if config.get("credential_version_id")
                else None
            ),
            messages=[{"role": "user", "content": rendered_prompt}],
            base_url=(
                str(config["custom_base_url"])
                if config.get("custom_base_url")
                else None
            ),
            openai_options=options,
            runtime_options=build_openai_compatible_runtime_options(timeout=120),
            capability=TRANSLATION_CAPABILITY,
        )
        result = OpenAICompatibleSyncExecutor().execute(
            request,
            capability=TRANSLATION_CAPABILITY,
            parser=parse_terms,
        )
        return {
            "rawContent": result.raw_content,
            "candidates": result.parsed,
        }

    def translate(
        self,
        texts: list[str],
        config: Mapping[str, Any],
        *,
        mode: str,
    ) -> Mapping[str, Any]:
        from src.core.translation import translate_single_text, translate_text_list

        provider = config["provider"]
        target_language = str(config.get("target_language", "zh"))
        translation_mode = str(config["translation_mode"])
        if translation_mode not in {"batch", "single"}:
            raise ValueError("unsupported translation mode")
        openai_options = _openai_options(config.get("openai_options"))
        enable_debug_logs = bool(config["enable_debug_logs"])

        def run(prompt: object, options: object, *, label: str) -> list[str]:
            if enable_debug_logs:
                LOGGER.info(
                    "[详细日志][%s] 提示词：%s\n输入文本：%s",
                    label,
                    prompt,
                    json.dumps(texts, ensure_ascii=False),
                )
            arguments = {
                "target_language": target_language,
                "model_provider": str(provider),
                "api_key": config.get("api_key"),
                "model_name": config.get("model_name"),
                "prompt_content": prompt,
                "custom_base_url": config.get("custom_base_url"),
                "openai_options": options,
                "credential_version_id": config.get("credential_version_id"),
            }
            if translation_mode == "single":
                result = [
                    translate_single_text(text, **arguments)
                    for text in texts
                ]
            else:
                result = translate_text_list(texts, **arguments)
            if enable_debug_logs:
                LOGGER.info(
                    "[详细日志][%s] 模型结果：%s",
                    label,
                    json.dumps(result, ensure_ascii=False),
                )
            return result

        translated = run(
            config.get("prompt_content"),
            openai_options,
            label="标准翻译",
        )
        textbox: list[str] = []
        textbox_prompt = str(config["textbox_prompt_content"])
        if bool(config["use_textbox_prompt"]) and textbox_prompt:
            textbox_options = type(openai_options).from_dict(
                openai_options.to_dict()
            )
            textbox_options.request.force_json_output = False
            textbox = run(
                textbox_prompt,
                textbox_options,
                label="文本框二次翻译",
            )
        return {"translated": translated, "textbox": textbox, "mode": mode}

    def translate_batch(
        self,
        pages: list[Mapping[str, Any]],
        images: list[Image.Image],
        config: Mapping[str, Any],
        *,
        mode: str,
    ) -> Mapping[str, Any]:
        from src.shared.ai_providers import HQ_TRANSLATION_CAPABILITY
        from src.shared.ai_transport import UnifiedChatRequest
        from src.shared.openai_execution import (
            OpenAICompatibleBusinessRetryableError,
            OpenAICompatibleSyncExecutor,
            build_openai_compatible_runtime_options,
            parse_json_block_from_text,
        )

        if len(pages) != len(images) or not pages:
            raise ValueError("HQ batch pages and images must be non-empty and aligned")
        request_pages = [
            {
                "pageId": str(page["pageId"]),
                "bubbles": [
                    {
                        "bubbleId": str(bubble["bubbleId"]),
                        "originalText": str(bubble.get("originalText", "")),
                        "translatedText": str(bubble.get("translatedText", "")),
                        "textDirection": str(
                            bubble.get("textDirection", "vertical")
                        ),
                    }
                    for bubble in page.get("bubbles", [])
                ],
            }
            for page in pages
        ]
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "请严格按稳定 ID 返回 JSON。输出格式只能是 "
                    '{"pages":[{"pageId":"...","bubbles":'
                    '[{"bubbleId":"...","translatedText":"..."}]}]}。'
                    "不得遗漏、增加或重复 pageId/bubbleId。\n\n"
                    + json.dumps(
                        {
                            "schemaVersion": 1,
                            "mode": mode,
                            "targetLanguage": str(
                                config.get("target_language", "zh")
                            ),
                            "pages": request_pages,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                ),
            }
        ]
        for page, image in zip(request_pages, images):
            payload = BytesIO()
            image.save(payload, format="PNG")
            content.extend(
                (
                    {
                        "type": "text",
                        "text": f"pageId={page['pageId']}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(payload.getvalue()).decode("ascii")
                        },
                    },
                )
            )
        prompt = str(config.get("prompt_content", "")).strip()
        messages: list[dict[str, Any]] = []
        if prompt:
            messages.append({"role": "system", "content": prompt})
        messages.append({"role": "user", "content": content})
        provider = str(config["provider"])
        options = _openai_options(config.get("openai_options"))
        enable_debug_logs = bool(config["enable_debug_logs"])
        if enable_debug_logs:
            LOGGER.info(
                "[详细日志][%s] 完整消息结构，共 %d 条消息",
                "AI校对" if mode == "proofread" else "高质量翻译",
                len(messages),
            )
            for message_index, message in enumerate(messages, start=1):
                LOGGER.info(
                    "[详细日志] Message %d role=%s",
                    message_index,
                    message.get("role"),
                )
                message_content = message.get("content")
                if isinstance(message_content, str):
                    LOGGER.info("[详细日志] %s", message_content)
                elif isinstance(message_content, list):
                    for item_index, item in enumerate(message_content, start=1):
                        if item.get("type") == "text":
                            LOGGER.info(
                                "[详细日志] 文本块 %d：%s",
                                item_index,
                                item.get("text", ""),
                            )
                        elif item.get("type") == "image_url":
                            image_url = str(
                                dict(item.get("image_url", {})).get("url", "")
                            )
                            LOGGER.info(
                                "[详细日志] 图片块 %d：%s...（长度 %d）",
                                item_index,
                                image_url[:100],
                                len(image_url),
                            )
        executor = OpenAICompatibleSyncExecutor()

        def parse(raw: str) -> dict[str, dict[str, str]]:
            try:
                payload = parse_json_block_from_text(raw)
                return _validate_stable_batch_result(
                    payload,
                    expected_pages=request_pages,
                )
            except (TypeError, ValueError, KeyError) as exc:
                raise OpenAICompatibleBusinessRetryableError(str(exc)) from exc

        result = executor.execute(
            UnifiedChatRequest(
                provider=provider,
                api_key=str(config.get("api_key", "")),
                model=str(config.get("model_name", "")),
                credential_version_id=(
                    str(config["credential_version_id"])
                    if config.get("credential_version_id")
                    else None
                ),
                base_url=str(config.get("custom_base_url", "")) or None,
                capability=HQ_TRANSLATION_CAPABILITY,
                openai_options=options,
                runtime_options=build_openai_compatible_runtime_options(
                    timeout=300.0 if options.execution.use_stream else 120.0,
                    print_stream_output=options.execution.use_stream,
                    stream_output_label=(
                        "AI校对" if mode == "proofread" else "高质量翻译"
                    ),
                ),
                messages=messages,
            ),
            capability=HQ_TRANSLATION_CAPABILITY,
            parser=parse,
        )
        if enable_debug_logs:
            LOGGER.info(
                "[详细日志][%s] 模型原始结果（前 1000 字符）：%s",
                "AI校对" if mode == "proofread" else "高质量翻译",
                result.raw_content[:1000],
            )
        return {
            "rawContent": result.raw_content,
            "pages": result.parsed,
            "mode": mode,
        }

    def repair(
        self,
        image: Image.Image,
        bubble_payloads: list[dict[str, Any]],
        config: Mapping[str, Any],
        *,
        precise_mask: Image.Image | None = None,
    ) -> Image.Image:
        import numpy as np

        from src.core.inpainting import inpaint_bubbles

        coords = [payload.get("coords", [0, 0, 0, 0]) for payload in bubble_payloads]
        polygons = [payload.get("polygon", []) for payload in bubble_payloads]
        repaired, _ = inpaint_bubbles(
            image,
            coords,
            method=str(config.get("method", "solid")),
            fill_color=str(config.get("fill_color", "#FFFFFF")),
            bubble_polygons=polygons,
            precise_mask=(
                np.array(precise_mask, dtype=np.uint8)
                if precise_mask is not None
                else None
            ),
            mask_dilate_size=int(config.get("mask_dilate_size", 0)),
            mask_box_expand_ratio=float(config.get("mask_box_expand_ratio", 0)),
            lama_model=str(config.get("lama_model", "lama_mpe")),
            disable_resize=bool(config["disable_resize"]),
        )
        return repaired

    def render(
        self,
        clean_image: Image.Image,
        bubble_payloads: list[dict[str, Any]],
        config: Mapping[str, Any],
    ) -> Image.Image:
        from src.core.config_models import BubbleState
        from src.core.rendering import render_bubbles_unified

        states = [BubbleState.from_dict(payload) for payload in bubble_payloads]
        rendered = clean_image.copy()
        render_bubbles_unified(rendered, states)
        return rendered


class TranslationPipelineService:
    def __init__(
        self,
        *,
        data_root: Path,
        engine: Engine,
        jobs: JobQueueRepository,
        algorithms: TranslationAlgorithms | None = None,
        plugin_runtime: Any | None = None,
    ) -> None:
        self.engine = engine
        self.jobs = jobs
        self.storage = AssetStorageService(data_root, engine)
        self.credentials = SettingsRepository(engine)
        self.algorithms = algorithms or CoreTranslationAlgorithms()
        self.plugin_runtime = plugin_runtime

    def handler(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Execute one atomic step and publish under the latest heartbeat fence."""

        page_id = step.get("pageId")
        if not isinstance(page_id, str):
            raise ValueError("translation step has no page")
        kind = str(step["stepKind"])
        if kind == "detect":
            result = self._detect(fence, step, page_id)
        elif kind == "ocr":
            result = self._ocr(fence, step, page_id)
        elif kind == "color":
            result = self._color(fence, step, page_id)
        elif kind == "auto_terms":
            result = self._auto_terms(fence, step, page_id)
        elif kind == "translate":
            result = self._translate(fence, step, page_id, kind)
        elif kind in {"hq_translate", "proofread"}:
            raise JobConflict(f"{kind} must run through the durable batch executor")
        elif kind == "repair":
            result = self._repair(fence, step, page_id)
        elif kind == "render":
            result = self._render(fence, step, page_id)
        elif kind == "save":
            result = self._save(fence, step, page_id)
        elif kind == "publish_clean":
            result = self._checkpoint_only(
                fence, step, {"published": "clean"}
            )
        else:
            raise ValueError(f"unsupported translation step: {kind}")
        return {**result, "__already_published__": True}

    def batch_handler(
        self,
        fence: AttemptFence,
        steps: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    ) -> Mapping[str, Any]:
        """Execute one HQ/proofreading model batch using stable page/bubble IDs."""

        if not steps:
            raise ValueError("translation batch cannot be empty")
        kind = str(steps[0]["stepKind"])
        step_ordinal = int(steps[0]["stepOrdinal"])
        if kind not in {"hq_translate", "proofread"}:
            raise ValueError(f"unsupported translation batch step: {kind}")
        if any(
            str(step["stepKind"]) != kind
            or int(step["stepOrdinal"]) != step_ordinal
            for step in steps
        ):
            raise ValueError("translation batch mixed step kinds or rounds")

        config = self._config(steps[0])
        if kind == "proofread":
            rounds = config.get("proofreadingRounds")
            round_index = step_ordinal - 1
            if (
                not isinstance(rounds, list)
                or round_index < 0
                or round_index >= len(rounds)
                or not isinstance(rounds[round_index], Mapping)
            ):
                raise JobConflict("frozen proofreading round is missing")
            section = self._with_credential(rounds[round_index])
            mode = "proofread"
        else:
            round_index = None
            section = self._with_credential(config.get("translation", {}))
            mode = "hq_translate"
        section.setdefault(
            "target_language",
            config.get("targetLanguage", "zh"),
        )

        prepared: list[
            tuple[Mapping[str, Any], PageSnapshot, list[dict[str, Any]]]
        ] = []
        for step in steps:
            page_id = step.get("pageId")
            if not isinstance(page_id, str):
                raise ValueError("translation batch step has no page")
            snapshot = self._snapshot(page_id)
            bubble_payloads = []
            for bubble_id, payload in zip(snapshot.bubble_ids, snapshot.bubbles):
                translated_text = str(payload.get("translatedText", ""))
                if kind == "proofread" and not translated_text.strip():
                    continue
                bubble_payloads.append(
                    {
                        "bubbleId": bubble_id,
                        "originalText": str(payload.get("originalText", "")),
                        "translatedText": translated_text,
                        "textDirection": str(
                            payload.get("textDirection", "vertical")
                        ),
                    }
                )
            if kind == "proofread" and not bubble_payloads:
                self.jobs.skip_remaining_item(
                    fence,
                    step_id=str(step["stepId"]),
                    reason="page_has_no_translated_bubbles",
                )
                continue
            prepared.append((step, snapshot, bubble_payloads))

        if not prepared:
            return {"__already_published__": True, "skipped": len(steps)}

        constraint_contexts = [
            {
                "pageId": snapshot.page_id,
                "constraints": self._effective_constraints(
                    step,
                    include_current_page=True,
                ),
            }
            for step, snapshot, _bubble_payloads in prepared
        ]
        constraint_context_by_page = {
            str(context["pageId"]): context
            for context in constraint_contexts
        }
        restore_by_page_bubble: dict[str, dict[str, dict[str, str]]] = {}
        section = self._with_constraint_prompt(
            section,
            constraint_contexts=constraint_contexts,
        )
        images: list[Image.Image] = []
        request_pages: list[dict[str, Any]] = []
        try:
            for step, snapshot, bubble_payloads in prepared:
                before = self._atomic_hook(
                    fence,
                    phase="before",
                    scope="ai_translate",
                    page_id=snapshot.page_id,
                    data={
                        "pageId": snapshot.page_id,
                        "originalTexts": [
                            str(bubble.get("originalText", ""))
                            for bubble in bubble_payloads
                        ],
                        "translations": [
                            str(bubble.get("translatedText", ""))
                            for bubble in bubble_payloads
                        ],
                    },
                )
                original_texts = list(before["originalTexts"])
                current_translations = list(
                    before.get("translations", [])
                )
                if len(original_texts) != len(bubble_payloads) or (
                    current_translations
                    and len(current_translations) != len(bubble_payloads)
                ):
                    raise JobConflict(
                        "AI translation plugin result count does not match bubbles"
                    )
                for index, bubble in enumerate(bubble_payloads):
                    bubble["originalText"] = original_texts[index]
                    if current_translations:
                        bubble["translatedText"] = current_translations[index]
                constraints = constraint_context_by_page[snapshot.page_id][
                    "constraints"
                ]
                non_translate = constraints["nonTranslate"]
                protected_bubbles: list[dict[str, Any]] = []
                restore_by_bubble: dict[str, dict[str, str]] = {}
                for bubble_index, bubble in enumerate(bubble_payloads):
                    protected = dict(bubble)
                    token_by_fragment: dict[str, str] = {}
                    original, original_restore = _protect_non_translate_text(
                        str(bubble.get("originalText", "")),
                        (
                            non_translate["entries"]
                            if bool(non_translate["enabled"])
                            else []
                        ),
                        token_prefix=f"{snapshot.page_id[:8]}_{bubble_index}",
                        token_by_fragment=token_by_fragment,
                    )
                    translated_text, translated_restore = (
                        _protect_non_translate_text(
                            str(bubble.get("translatedText", "")),
                            (
                                non_translate["entries"]
                                if bool(non_translate["enabled"])
                                else []
                            ),
                            token_prefix=f"{snapshot.page_id[:8]}_{bubble_index}",
                            token_by_fragment=token_by_fragment,
                        )
                    )
                    protected["originalText"] = original
                    protected["translatedText"] = translated_text
                    protected_bubbles.append(protected)
                    restore_by_bubble[str(bubble["bubbleId"])] = (
                        translated_restore
                        if translated_restore
                        else original_restore
                    )
                restore_by_page_bubble[snapshot.page_id] = restore_by_bubble
                request_pages.append(
                    {
                        "pageId": snapshot.page_id,
                        "bubbles": protected_bubbles,
                    }
                )
                images.append(
                    self._open_bound_image(
                        fence,
                        step,
                        snapshot.page_id,
                        "source",
                    )
                )
            result = self.algorithms.translate_batch(
                request_pages,
                images,
                section,
                mode=mode,
            )
        finally:
            for image in images:
                image.close()

        parsed = result.get("pages")
        if not isinstance(parsed, Mapping):
            raise JobConflict("HQ batch returned no validated page mapping")
        expected = _validate_stable_batch_result(
            {
                "pages": [
                    {
                        "pageId": page_id,
                        "bubbles": [
                            {
                                "bubbleId": bubble_id,
                                "translatedText": text,
                            }
                            for bubble_id, text in bubble_results.items()
                        ],
                    }
                    for page_id, bubble_results in parsed.items()
                    if isinstance(bubble_results, Mapping)
                ]
            },
            expected_pages=request_pages,
        )
        raw_content = str(result.get("rawContent", ""))
        batch_id = str(uuid.uuid4())
        raw_payload = (
            raw_content
            if raw_content
            else _json({"pages": parsed})
        )
        raw_asset = self.storage.publish_bytes(
            raw_payload.encode("utf-8"),
            extension="json",
            mime_type="application/json",
            bind=lambda connection, asset_id: connection.execute(
                insert(job_step_asset_outputs)
                .values(
                    job_step_id=str(prepared[0][0]["stepId"]),
                    role="model_raw",
                    asset_id=asset_id,
                )
                .prefix_with("OR REPLACE")
            ),
        )

        completed = 0
        for step, snapshot, requested_bubbles in prepared:
            translated_by_id = expected[snapshot.page_id]
            for bubble_id, translated_text in translated_by_id.items():
                translated_by_id[bubble_id] = _restore_non_translate_text(
                    translated_text,
                    restore_by_page_bubble[snapshot.page_id].get(
                        bubble_id,
                        {},
                    ),
                )
            ordered_ids = [
                str(bubble["bubbleId"])
                for bubble in requested_bubbles
            ]
            after = self._atomic_hook(
                fence,
                phase="after",
                scope="ai_translate",
                page_id=snapshot.page_id,
                data={
                    "pageId": snapshot.page_id,
                    "originalTexts": [
                        str(bubble.get("originalText", ""))
                        for bubble in requested_bubbles
                    ],
                    "translations": [
                        str(translated_by_id[bubble_id])
                        for bubble_id in ordered_ids
                    ],
                },
            )
            plugin_translations = list(after["translations"])
            if len(plugin_translations) != len(ordered_ids):
                raise JobConflict(
                    "AI translation plugin result count does not match bubbles"
                )
            translated_by_id = {
                bubble_id: plugin_translations[index]
                for index, bubble_id in enumerate(ordered_ids)
            }
            updated = [dict(payload) for payload in snapshot.bubbles]
            index_by_id = {
                bubble_id: index
                for index, bubble_id in enumerate(snapshot.bubble_ids)
            }
            for requested in requested_bubbles:
                bubble_id = str(requested["bubbleId"])
                updated[index_by_id[bubble_id]]["translatedText"] = str(
                    translated_by_id[bubble_id]
                )
            warnings = _translation_constraint_warnings(
                [
                    str(bubble.get("originalText", ""))
                    for bubble in requested_bubbles
                ],
                [
                    str(translated_by_id[str(bubble["bubbleId"])])
                    for bubble in requested_bubbles
                ],
                constraint_context_by_page[snapshot.page_id]["constraints"],
            )
            warnings_by_bubble: dict[int, list[dict[str, Any]]] = {}
            for warning in warnings:
                warnings_by_bubble.setdefault(
                    int(warning["bubbleIndex"]),
                    [],
                ).append(warning)
            for bubble_index, requested in enumerate(requested_bubbles):
                updated[index_by_id[str(requested["bubbleId"])]][
                    "translationWarnings"
                ] = warnings_by_bubble.get(bubble_index, [])
            fingerprint = _batch_input_fingerprint(
                page_id=snapshot.page_id,
                document_revision=snapshot.document_revision,
                bubbles=requested_bubbles,
                mode=mode,
                round_index=round_index,
                constraint_context=constraint_context_by_page[snapshot.page_id],
            )
            self._publish_bubble_update(
                fence,
                step,
                snapshot,
                updated,
                {
                    "batchId": batch_id,
                    "batchSize": len(prepared),
                    "mode": mode,
                    "roundIndex": round_index,
                    "rawAssetId": raw_asset.id,
                    "parsedBubbleCount": len(translated_by_id),
                    "constraintWarnings": warnings,
                    "inputFingerprint": fingerprint,
                },
                input_fingerprint=fingerprint,
            )
            completed += 1
        return {
            "__already_published__": True,
            "batchId": batch_id,
            "completed": completed,
            "rawAssetId": raw_asset.id,
        }

    def _detect(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
        page_id: str,
    ) -> Mapping[str, Any]:
        snapshot = self._snapshot(page_id)
        source = self._bound_asset(fence, step, page_id, "source")
        before = self._atomic_hook(
            fence,
            phase="before",
            scope="detect",
            page_id=page_id,
            data={
                "pageId": page_id,
                "sourceAssetId": str(source["id"]),
                "detectorConfig": dict(
                    self._config(step).get("detector", {})
                ),
            },
        )
        image = self._open_asset(str(before["sourceAssetId"]), "RGB")
        try:
            result = self.algorithms.detect(
                image,
                dict(before["detectorConfig"]),
            )
        finally:
            image.close()
        coords = list(result.get("coords", []))
        polygons = list(result.get("polygons", []))
        angles = list(result.get("angles", []))
        directions = list(result.get("auto_directions", []))
        textlines = list(result.get("textlines_per_bubble", []))
        payloads = _preserve_detected_text([
            self._new_bubble_payload(
                coords=value,
                polygon=polygons[index] if index < len(polygons) else [],
                angle=angles[index] if index < len(angles) else 0,
                auto_direction=(
                    directions[index] if index < len(directions) else "vertical"
                ),
                textlines=textlines[index] if index < len(textlines) else [],
                style=snapshot.style_defaults,
            )
            for index, value in enumerate(coords)
        ], snapshot.bubbles)
        mask_record: AssetRecord | None = None
        mask = result.get("raw_mask")
        if isinstance(mask, Image.Image):
            mask_record = publish_png_asset(self.storage, mask, mode="L")
            mask.close()
        elif mask is not None:
            mask_image = Image.fromarray(mask)
            try:
                mask_record = publish_png_asset(self.storage, mask_image, mode="L")
            finally:
                mask_image.close()
        after = self._atomic_hook(
            fence,
            phase="after",
            scope="detect",
            page_id=page_id,
            data={
                "pageId": page_id,
                "bubbles": payloads,
                "textMaskAssetId": (
                    mask_record.id if mask_record is not None else None
                ),
            },
        )
        payloads = [dict(value) for value in after["bubbles"]]
        mask_asset_id = after.get("textMaskAssetId")
        mask_record = (
            self._asset_record(str(mask_asset_id))
            if mask_asset_id is not None
            else None
        )
        new_revision = snapshot.document_revision + 1

        def publish(connection: Connection) -> None:
            self._assert_revision(
                connection, page_id, snapshot.document_revision
            )
            standalone_detect = str(step.get("jobKind", "")) == "detect"
            has_translated_asset = (
                connection.execute(
                    select(page_assets.c.asset_id).where(
                        page_assets.c.page_id == page_id,
                        page_assets.c.role == "translated",
                    )
                ).scalar_one_or_none()
                is not None
            )
            has_drawable_text = any(
                str(payload.get("translatedText", "")).strip()
                for payload in payloads
            )
            needs_render = bool(
                standalone_detect
                and (has_translated_asset or has_drawable_text)
            )
            connection.execute(delete(bubbles).where(bubbles.c.page_id == page_id))
            if payloads:
                connection.execute(
                    insert(bubbles),
                    [
                        {
                            "id": str(uuid.uuid4()),
                            "page_id": page_id,
                            "ordinal": index,
                            "payload_json": _json(payload),
                            "payload_schema_version": 1,
                            "updated_revision": new_revision,
                        }
                        for index, payload in enumerate(payloads, start=1)
                    ],
                )
            page_values: dict[str, object] = {
                "document_revision": new_revision,
                "detection_state": "processed",
                "render_status": (
                    "stale"
                    if not standalone_detect or needs_render
                    else "not_rendered"
                ),
            }
            if standalone_detect and not needs_render:
                page_values["rendered_revision"] = None
            connection.execute(
                update(pages)
                .where(
                    pages.c.id == page_id,
                    pages.c.document_revision == snapshot.document_revision,
                )
                .values(**page_values)
            )
            connection.execute(
                delete(page_assets).where(
                    page_assets.c.page_id == page_id,
                    page_assets.c.role == "text_mask",
                )
            )
            if mask_record is not None:
                self._publish_pointer(
                    connection,
                    page_id=page_id,
                    role="text_mask",
                    asset=mask_record,
                    source_revision=snapshot.source_revision,
                    document_revision=new_revision,
                    step_id=str(step["stepId"]),
                )
            if standalone_detect and not needs_render:
                connection.execute(
                    update(job_steps)
                    .where(
                        job_steps.c.job_item_id == step["itemId"],
                        job_steps.c.kind.in_(("render", "save")),
                        job_steps.c.status == "pending",
                    )
                    .values(status="skipped")
                )

        checkpoint = {
            "bubbleCount": len(payloads),
            "documentRevision": new_revision,
        }
        self.jobs.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint=checkpoint,
            publisher=publish,
        )
        return checkpoint

    def _ocr(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
        page_id: str,
    ) -> Mapping[str, Any]:
        snapshot = self._snapshot(page_id)
        source = self._bound_asset(fence, step, page_id, "source")
        config = self._config(step)
        raw_section = config.get("ocr", {})
        before = self._atomic_hook(
            fence,
            phase="before",
            scope="ocr",
            page_id=page_id,
            data={
                "pageId": page_id,
                "sourceAssetId": str(source["id"]),
                "bubbles": [dict(value) for value in snapshot.bubbles],
                "ocrConfig": (
                    dict(raw_section)
                    if isinstance(raw_section, Mapping)
                    else {}
                ),
            },
        )
        image = self._open_asset(str(before["sourceAssetId"]), "RGB")
        try:
            section = self._with_credential(before["ocrConfig"])
            section.setdefault(
                "source_language",
                config.get("sourceLanguage", "japanese"),
            )
            result = self.algorithms.ocr(
                image,
                [dict(value) for value in before["bubbles"]],
                section,
            )
        finally:
            image.close()
        texts = list(result.get("texts", []))
        details = list(result.get("results", []))
        after = self._atomic_hook(
            fence,
            phase="after",
            scope="ocr",
            page_id=page_id,
            data={
                "pageId": page_id,
                "originalTexts": [str(value) for value in texts],
                "ocrResults": details,
            },
        )
        texts = list(after["originalTexts"])
        details = list(after["ocrResults"])
        updated = [dict(payload) for payload in snapshot.bubbles]
        if len(texts) != len(updated):
            raise JobConflict("OCR result count does not match persisted bubbles")
        for index, payload in enumerate(updated):
            payload["originalText"] = str(texts[index])
            payload["ocrResult"] = details[index] if index < len(details) else None
        return self._publish_bubble_update(
            fence,
            step,
            snapshot,
            updated,
            {"recognized": len(texts)},
        )

    def _color(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
        page_id: str,
    ) -> Mapping[str, Any]:
        snapshot = self._snapshot(page_id)
        source = self._bound_asset(fence, step, page_id, "source")
        before = self._atomic_hook(
            fence,
            phase="before",
            scope="color",
            page_id=page_id,
            data={
                "pageId": page_id,
                "sourceAssetId": str(source["id"]),
                "bubbles": [dict(value) for value in snapshot.bubbles],
            },
        )
        image = self._open_asset(str(before["sourceAssetId"]), "RGB")
        try:
            colors = self.algorithms.colors(
                image,
                [dict(value) for value in before["bubbles"]],
            )
        finally:
            image.close()
        after = self._atomic_hook(
            fence,
            phase="after",
            scope="color",
            page_id=page_id,
            data={
                "pageId": page_id,
                "colors": [
                    {
                        "fgColor": (
                            list(color["fg_color"])
                            if color.get("fg_color") is not None
                            else None
                        ),
                        "bgColor": (
                            list(color["bg_color"])
                            if color.get("bg_color") is not None
                            else None
                        ),
                        "confidence": float(
                            color.get("confidence", 0)
                        ),
                    }
                    for color in colors
                ],
            },
        )
        colors = [
            {
                "fg_color": color.get("fgColor"),
                "bg_color": color.get("bgColor"),
                "confidence": color.get("confidence", 0),
            }
            for color in after["colors"]
        ]
        if len(colors) != len(snapshot.bubbles):
            raise JobConflict("color result count does not match persisted bubbles")
        updated = [dict(payload) for payload in snapshot.bubbles]
        uses_auto_color = bool(snapshot.style_defaults["useAutoTextColor"])
        for payload, color in zip(updated, colors):
            foreground = color.get("fg_color")
            background = color.get("bg_color")
            payload["autoFgColor"] = foreground
            payload["autoBgColor"] = background
            payload["colorConfidence"] = float(color.get("confidence", 0))
            if uses_auto_color and foreground is not None:
                payload["textColor"] = rgb_to_hex(foreground)
            if uses_auto_color and background is not None:
                payload["fillColor"] = rgb_to_hex(background)
        return self._publish_bubble_update(
            fence,
            step,
            snapshot,
            updated,
            {"colored": len(colors)},
        )

    def _auto_terms(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
        page_id: str,
    ) -> Mapping[str, Any]:
        snapshot = self._snapshot(page_id)
        config = self._config(step)
        texts = [
            str(payload.get("originalText", "")).strip()
            for payload in snapshot.bubbles
            if str(payload.get("originalText", "")).strip()
        ]
        effective_before = self._effective_constraints(
            step,
            include_current_page=False,
        )
        glossary = effective_before["glossary"]
        baseline_revision = int(
            config.get("translationConstraintRevision", 0)
        )
        fingerprint = hashlib.sha256(
            _json(
                {
                    "pageId": page_id,
                    "texts": texts,
                    "baselineRevision": baseline_revision,
                    "effectiveGlossary": glossary["entries"],
                },
            ).encode("utf-8")
        ).hexdigest()

        if not bool(glossary["autoExtractEnabled"]) or not texts:
            checkpoint = {
                "baselineRevision": baseline_revision,
                "candidateCount": 0,
                "duplicateCount": 0,
                "addedCount": 0,
                "delta": [],
                "skipped": (
                    "disabled"
                    if not bool(glossary["autoExtractEnabled"])
                    else "no_ocr_text"
                ),
            }
            self.jobs.complete_step(
                fence,
                step_id=str(step["stepId"]),
                checkpoint=checkpoint,
                input_fingerprint=fingerprint,
            )
            return checkpoint

        section = self._with_credential(config.get("translation", {}))
        result = self.algorithms.extract_terms(
            texts,
            section,
            prompt=str(glossary["autoExtractPrompt"]),
        )
        raw_candidates = result.get("candidates")
        if not isinstance(raw_candidates, list):
            raise JobConflict("automatic term extraction returned no candidate array")
        candidates: list[dict[str, str]] = []
        for index, raw in enumerate(raw_candidates):
            if not isinstance(raw, Mapping):
                raise JobConflict(
                    f"automatic term candidate {index} must be an object"
                )
            source = str(raw.get("source", "")).strip()
            target = str(raw.get("target", "")).strip()
            if not source or not target:
                raise JobConflict(
                    f"automatic term candidate {index} requires source and target"
                )
            candidates.append(
                {
                    "source": source,
                    "target": target,
                    "note": str(raw.get("note", "")).strip(),
                    "matchMode": "text",
                }
            )

        effective_after, added_count = with_glossary_delta(
            effective_before,
            candidates,
        )
        before_keys = {
            (str(entry["matchMode"]), str(entry["source"]))
            for entry in glossary["entries"]
        }
        delta = [
            dict(entry)
            for entry in effective_after["glossary"]["entries"]
            if (str(entry["matchMode"]), str(entry["source"])) not in before_keys
        ]
        duplicate_count = len(candidates) - added_count
        raw_content_value = result.get("rawContent")
        if raw_content_value is None:
            raw_content_value = _json(candidates)
        raw_content = str(raw_content_value)
        raw_asset = self.storage.publish_bytes(
            raw_content.encode("utf-8"),
            extension="json",
            mime_type="application/json",
            bind=lambda connection, asset_id: connection.execute(
                insert(job_step_asset_outputs)
                .values(
                    job_step_id=str(step["stepId"]),
                    role="model_raw",
                    asset_id=asset_id,
                )
                .prefix_with("OR REPLACE")
            ),
        )
        checkpoint = {
            "baselineRevision": baseline_revision,
            "candidateCount": len(candidates),
            "duplicateCount": duplicate_count,
            "addedCount": added_count,
            "delta": delta,
            "rawAssetId": raw_asset.id,
        }

        def publish(connection: Connection) -> None:
            if not delta:
                return
            book_id = connection.execute(
                select(jobs.c.book_id).where(jobs.c.id == fence.job_id)
            ).scalar_one_or_none()
            if book_id is None:
                raise JobConflict("translation job book no longer exists")
            current = connection.execute(
                select(
                    translation_constraints.c.payload_json,
                    translation_constraints.c.revision,
                ).where(translation_constraints.c.book_id == book_id)
            ).mappings().one_or_none()
            if current is None:
                raise JobConflict("translation constraints no longer exist")
            merged, global_added = with_glossary_delta(
                validate_translation_constraints(
                    json.loads(current["payload_json"])
                ),
                delta,
            )
            if global_added == 0:
                return
            changed = connection.execute(
                update(translation_constraints)
                .where(
                    translation_constraints.c.book_id == book_id,
                    translation_constraints.c.revision == current["revision"],
                )
                .values(
                    payload_json=_json(merged),
                    revision=int(current["revision"]) + 1,
                    updated_at=utcnow(),
                )
            )
            if changed.rowcount != 1:
                raise JobConflict(
                    "translation constraints changed during automatic term append"
                )

        self.jobs.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint=checkpoint,
            input_fingerprint=fingerprint,
            publisher=publish,
        )
        return checkpoint

    def _translate(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
        page_id: str,
        mode: str,
    ) -> Mapping[str, Any]:
        snapshot = self._snapshot(page_id)
        config = self._config(step)
        persisted_texts = [
            str(payload.get("originalText", ""))
            for payload in snapshot.bubbles
        ]
        constraints = self._effective_constraints(
            step,
            include_current_page=True,
        )
        raw_section = config.get("translation", {})
        before = self._atomic_hook(
            fence,
            phase="before",
            scope="translate",
            page_id=page_id,
            data={
                "pageId": page_id,
                "originalTexts": persisted_texts,
                "translationConfig": (
                    dict(raw_section)
                    if isinstance(raw_section, Mapping)
                    else {}
                ),
            },
        )
        texts = list(before["originalTexts"])
        if len(texts) != len(snapshot.bubbles):
            raise JobConflict(
                "before_translate original text count does not match bubbles"
            )
        section = self._with_constraint_prompt(
            self._with_credential(
                before["translationConfig"]
            ),
            constraint_contexts=[
                {
                    "pageId": page_id,
                    "constraints": constraints,
                }
            ],
        )
        section.setdefault(
            "target_language",
            config.get("targetLanguage", "zh"),
        )
        non_translate = constraints["nonTranslate"]
        protected_texts: list[str] = []
        restore_by_index: list[dict[str, str]] = []
        for index, text in enumerate(texts):
            protected, restore = _protect_non_translate_text(
                text,
                (
                    non_translate["entries"]
                    if bool(non_translate["enabled"])
                    else []
                ),
                token_prefix=f"{page_id[:8]}_{index}",
            )
            protected_texts.append(protected)
            restore_by_index.append(restore)
        result = self.algorithms.translate(protected_texts, section, mode=mode)
        raw_translated = list(result.get("translated", []))
        if len(raw_translated) != len(restore_by_index):
            raise JobConflict("translation result count does not match bubbles")
        translated = [
            _restore_non_translate_text(str(value), restore_by_index[index])
            for index, value in enumerate(raw_translated)
        ]
        raw_textbox = list(result.get("textbox", []))
        if raw_textbox and len(raw_textbox) != len(restore_by_index):
            raise JobConflict("textbox translation result count does not match bubbles")
        textbox = [
            (
                _restore_non_translate_text(str(value), restore_by_index[index])
                if str(value)
                else ""
            )
            for index, value in enumerate(raw_textbox)
        ]
        after = self._atomic_hook(
            fence,
            phase="after",
            scope="translate",
            page_id=page_id,
            data={
                "pageId": page_id,
                "originalTexts": texts,
                "translations": translated,
                "textboxTexts": [str(value) for value in textbox],
            },
        )
        translated = list(after["translations"])
        textbox = list(after.get("textboxTexts", []))
        if len(translated) != len(snapshot.bubbles):
            raise JobConflict("translation result count does not match bubbles")
        if textbox and len(textbox) != len(snapshot.bubbles):
            raise JobConflict("textbox translation result count does not match bubbles")
        updated = [dict(payload) for payload in snapshot.bubbles]
        warnings = _translation_constraint_warnings(
            persisted_texts,
            translated,
            constraints,
        )
        warnings_by_bubble: dict[int, list[dict[str, Any]]] = {}
        for warning in warnings:
            warnings_by_bubble.setdefault(
                int(warning["bubbleIndex"]),
                [],
            ).append(warning)
        for index, payload in enumerate(updated):
            payload["translatedText"] = str(translated[index])
            payload["textboxText"] = (
                str(textbox[index]) if index < len(textbox) else ""
            )
            payload["translationWarnings"] = warnings_by_bubble.get(index, [])
        return self._publish_bubble_update(
            fence,
            step,
            snapshot,
            updated,
            {
                "translated": len(translated),
                "mode": mode,
                "constraintWarnings": warnings,
            },
            input_fingerprint=hashlib.sha256(
                _json(
                    {
                        "pageId": page_id,
                        "documentRevision": snapshot.document_revision,
                        "texts": texts,
                        "mode": mode,
                        "translationConstraints": constraints,
                    },
                ).encode("utf-8")
            ).hexdigest(),
        )

    def _repair(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
        page_id: str,
    ) -> Mapping[str, Any]:
        snapshot = self._snapshot(page_id)
        source = self._bound_asset(fence, step, page_id, "source")
        precise_mask_asset_id = self._completed_step_asset_id(
            fence,
            step,
            step_kind="detect",
            role="text_mask",
        )
        before = self._atomic_hook(
            fence,
            phase="before",
            scope="inpaint",
            page_id=page_id,
            data={
                "pageId": page_id,
                "sourceAssetId": str(source["id"]),
                "inputAssetId": str(source["id"]),
                "textMaskAssetId": precise_mask_asset_id,
                "bubbles": [dict(value) for value in snapshot.bubbles],
                "method": str(snapshot.style_defaults["inpaintMethod"]),
                "fillColor": snapshot.style_defaults["fillColor"],
            },
        )
        image = self._open_asset(str(before["inputAssetId"]), "RGB")
        precise_mask = (
            self._open_asset(str(before["textMaskAssetId"]), "L")
            if before.get("textMaskAssetId") is not None
            else None
        )
        # Precise-mask expansion is frozen as task configuration. Repair method
        # and fill color are page facts bound when this item starts.
        inpainting = dict(self._config(step).get("inpainting", {}))
        method = str(before["method"])
        inpainting["method"] = "solid" if method == "solid" else "lama"
        inpainting["lama_model"] = (
            "litelama" if method == "litelama" else "lama_mpe"
        )
        inpainting["fill_color"] = before.get("fillColor") or "#FFFFFF"
        try:
            repaired = self.algorithms.repair(
                image,
                [dict(value) for value in before["bubbles"]],
                inpainting,
                precise_mask=precise_mask,
            )
        finally:
            image.close()
            if precise_mask is not None:
                precise_mask.close()
        record = publish_png_asset(self.storage, repaired, mode="RGB")
        repaired.close()
        after = self._atomic_hook(
            fence,
            phase="after",
            scope="inpaint",
            page_id=page_id,
            data={
                "pageId": page_id,
                "cleanAssetId": record.id,
                "documentRevision": snapshot.document_revision,
            },
        )
        record = self._asset_record(str(after["cleanAssetId"]))

        def publish(connection: Connection) -> None:
            self._assert_revision(
                connection, page_id, snapshot.document_revision
            )
            self._publish_pointer(
                connection,
                page_id=page_id,
                role="clean",
                asset=record,
                source_revision=snapshot.source_revision,
                document_revision=snapshot.document_revision,
                step_id=str(step["stepId"]),
            )

        checkpoint = {
            "cleanAssetId": record.id,
            "documentRevision": snapshot.document_revision,
        }
        self.jobs.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint=checkpoint,
            publisher=publish,
        )
        return checkpoint

    def _render(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
        page_id: str,
    ) -> Mapping[str, Any]:
        snapshot = self._snapshot(page_id)
        from src.backend_v2.rendering.fonts import (
            materialize_render_payloads,
        )

        job_kind = str(step.get("jobKind", ""))
        initialize_auto_fields: frozenset[str]
        if job_kind == "translation":
            initialize_auto_fields = frozenset(
                {"fontSize", "layoutDirection", "textColor", "fillColor"}
            )
        elif job_kind == "style_apply":
            config = self._config(step)
            selected = config.get("selectedFields", [])
            frozen = config.get("frozenStyle", {})
            initialize_auto_fields = (
                frozenset({"fontSize"})
                if (
                    isinstance(selected, list)
                    and "fontSize" in selected
                    and isinstance(frozen, Mapping)
                    and bool(frozen.get("autoFontSize", False))
                )
                else frozenset()
            )
        else:
            # Text import and any future ordinary render jobs must preserve the
            # concrete bubble values already stored in the page document.
            initialize_auto_fields = frozenset()
        with self.engine.connect() as connection:
            projected = materialize_render_payloads(
                connection,
                self.storage,
                page_id,
                initialize_auto_fields=initialize_auto_fields,
            )
        render_payloads = [
            render_payload
            for _bubble_id, _payload, render_payload in projected
        ]
        persisted_payloads = [
            (bubble_id, payload)
            for bubble_id, payload, _render_payload in projected
        ]
        try:
            input_asset = self._bound_asset(
                fence,
                step,
                page_id,
                "clean",
            )
        except JobConflict:
            input_asset = self._bound_asset(
                fence,
                step,
                page_id,
                "source",
            )
        render_section = self._config(step).get("render", {})
        before = self._atomic_hook(
            fence,
            phase="before",
            scope="render",
            page_id=page_id,
            data={
                "pageId": page_id,
                "inputAssetId": str(input_asset["id"]),
                "bubbles": render_payloads,
                "renderConfig": (
                    dict(render_section)
                    if isinstance(render_section, Mapping)
                    else {}
                ),
            },
        )
        clean = self._open_asset(str(before["inputAssetId"]), "RGB")
        try:
            rendered = self.algorithms.render(
                clean,
                [dict(value) for value in before["bubbles"]],
                dict(before["renderConfig"]),
            )
        finally:
            clean.close()
        translated = publish_png_asset(self.storage, rendered, mode="RGB")
        thumbnail = publish_thumbnail_asset(self.storage, rendered)
        rendered.close()
        after = self._atomic_hook(
            fence,
            phase="after",
            scope="render",
            page_id=page_id,
            data={
                "pageId": page_id,
                "translatedAssetId": translated.id,
                "thumbnailAssetId": thumbnail.id,
                "documentRevision": snapshot.document_revision,
            },
        )
        translated = self._asset_record(str(after["translatedAssetId"]))
        thumbnail = self._asset_record(str(after["thumbnailAssetId"]))

        def publish(connection: Connection) -> None:
            self._assert_revision(
                connection, page_id, snapshot.document_revision
            )
            for bubble_id, payload in persisted_payloads:
                connection.execute(
                    update(bubbles)
                    .where(
                        bubbles.c.id == bubble_id,
                        bubbles.c.page_id == page_id,
                        bubbles.c.updated_revision
                        <= snapshot.document_revision,
                    )
                    .values(
                        payload_json=_json(payload),
                        updated_revision=snapshot.document_revision,
                    )
                )
            connection.execute(
                insert(job_step_asset_outputs),
                [
                    {
                        "job_step_id": str(step["stepId"]),
                        "role": "translated",
                        "asset_id": translated.id,
                    },
                    {
                        "job_step_id": str(step["stepId"]),
                        "role": "thumbnail_translated",
                        "asset_id": thumbnail.id,
                    },
                ],
            )

        checkpoint = {
            "translatedAssetId": translated.id,
            "thumbnailAssetId": thumbnail.id,
            "documentRevision": snapshot.document_revision,
        }
        self.jobs.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint=checkpoint,
            publisher=publish,
        )
        return checkpoint

    def _save(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
        page_id: str,
    ) -> Mapping[str, Any]:
        """Publish a prior render checkpoint as the current page projection."""

        snapshot = self._snapshot(page_id)
        with self.engine.connect() as connection:
            rows = list(
                connection.execute(
                    select(
                        job_step_asset_outputs.c.role,
                        assets.c.id,
                        assets.c.relative_path,
                        assets.c.mime_type,
                        assets.c.checksum,
                        assets.c.byte_size,
                        assets.c.width,
                        assets.c.height,
                    )
                    .join(
                        job_steps,
                        job_steps.c.id
                        == job_step_asset_outputs.c.job_step_id,
                    )
                    .join(
                        job_items,
                        job_items.c.id == job_steps.c.job_item_id,
                    )
                    .join(
                        assets,
                        assets.c.id == job_step_asset_outputs.c.asset_id,
                    )
                    .where(
                        job_items.c.id == str(step["itemId"]),
                        job_items.c.job_id == fence.job_id,
                        job_steps.c.kind == "render",
                        job_steps.c.status == "completed",
                        job_step_asset_outputs.c.role.in_(
                            ("translated", "thumbnail_translated")
                        ),
                    )
                ).mappings()
            )
        records = {
            str(row["role"]): AssetRecord(
                id=str(row["id"]),
                relative_path=str(row["relative_path"]),
                mime_type=str(row["mime_type"]),
                checksum=str(row["checksum"]),
                byte_size=int(row["byte_size"]),
                width=int(row["width"]) if row["width"] is not None else None,
                height=int(row["height"]) if row["height"] is not None else None,
            )
            for row in rows
        }
        translated = records.get("translated")
        thumbnail = records.get("thumbnail_translated")
        if translated is None or thumbnail is None:
            raise JobConflict("save step has no complete render asset checkpoint")
        def publish(connection: Connection) -> None:
            self._assert_revision(
                connection, page_id, snapshot.document_revision
            )
            self._publish_pointer(
                connection,
                page_id=page_id,
                role="translated",
                asset=translated,
                source_revision=snapshot.source_revision,
                document_revision=snapshot.document_revision,
                step_id=str(step["stepId"]),
            )
            self._publish_pointer(
                connection,
                page_id=page_id,
                role="thumbnail_translated",
                asset=thumbnail,
                source_revision=snapshot.source_revision,
                document_revision=snapshot.document_revision,
                step_id=str(step["stepId"]),
                parent_asset_id=translated.id,
            )
            connection.execute(
                update(pages)
                .where(
                    pages.c.id == page_id,
                    pages.c.document_revision == snapshot.document_revision,
                )
                .values(
                    rendered_revision=snapshot.document_revision,
                    render_status="ready",
                )
            )

        checkpoint = {
            "translatedAssetId": translated.id,
            "thumbnailAssetId": thumbnail.id,
            "documentRevision": snapshot.document_revision,
        }
        self.jobs.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint=checkpoint,
            publisher=publish,
        )
        return checkpoint

    def _publish_bubble_update(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
        snapshot: PageSnapshot,
        payloads: list[dict[str, Any]],
        checkpoint: dict[str, Any],
        *,
        input_fingerprint: str | None = None,
    ) -> Mapping[str, Any]:
        new_revision = snapshot.document_revision + 1

        def publish(connection: Connection) -> None:
            self._assert_revision(
                connection, snapshot.page_id, snapshot.document_revision
            )
            rows = list(
                connection.execute(
                    select(bubbles.c.id)
                    .where(bubbles.c.page_id == snapshot.page_id)
                    .order_by(bubbles.c.ordinal)
                ).scalars()
            )
            if len(rows) != len(payloads):
                raise JobConflict("bubble set changed before step publication")
            for bubble_id, payload in zip(rows, payloads):
                connection.execute(
                    update(bubbles)
                    .where(
                        bubbles.c.id == bubble_id,
                        bubbles.c.updated_revision
                        == snapshot.document_revision,
                    )
                    .values(
                        payload_json=_json(payload),
                        updated_revision=new_revision,
                    )
                )
            changed = connection.execute(
                update(pages)
                .where(
                    pages.c.id == snapshot.page_id,
                    pages.c.document_revision == snapshot.document_revision,
                )
                .values(
                    document_revision=new_revision,
                    render_status="stale",
                )
            )
            if changed.rowcount != 1:
                raise JobConflict("page revision changed during step publication")

        checkpoint = {**checkpoint, "documentRevision": new_revision}
        self.jobs.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint=checkpoint,
            input_fingerprint=input_fingerprint,
            publisher=publish,
        )
        return checkpoint

    def _checkpoint_only(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
        checkpoint: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.jobs.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint=checkpoint,
        )
        return checkpoint

    def _snapshot(self, page_id: str) -> PageSnapshot:
        with self.engine.connect() as connection:
            page = connection.execute(
                select(pages).where(pages.c.id == page_id)
            ).mappings().one_or_none()
            if page is None:
                raise JobConflict("job target page no longer exists")
            rows = list(connection.execute(
                select(bubbles.c.id, bubbles.c.payload_json)
                .where(bubbles.c.page_id == page_id)
                .order_by(bubbles.c.ordinal)
            ))
            return PageSnapshot(
                page_id=page_id,
                source_revision=int(page["source_revision"]),
                document_revision=int(page["document_revision"]),
                render_status=str(page["render_status"]),
                style_defaults=validate_page_style(
                    json.loads(page["page_style_defaults_json"]),
                    partial=False,
                ),
                bubble_ids=tuple(str(row.id) for row in rows),
                bubbles=tuple(json.loads(row.payload_json) for row in rows),
            )

    def _open_bound_image(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
        page_id: str,
        role: str,
    ) -> Image.Image:
        bound = self._bound_asset(
            fence,
            step,
            page_id,
            role,
        )
        return self._open_asset(str(bound["id"]), "RGB")

    def _bound_asset(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
        page_id: str,
        role: str,
    ) -> dict[str, object]:
        return self.jobs.bind_item_inputs(
            fence,
            item_id=str(step["itemId"]),
            page_id=page_id,
            roles=(role,),
        )[role]

    def _open_asset(self, asset_id: str, mode: str) -> Image.Image:
        with self.engine.connect() as connection:
            relative_path = connection.execute(
                select(assets.c.relative_path).where(
                    assets.c.id == asset_id
                )
            ).scalar_one_or_none()
        if relative_path is None:
            raise JobConflict("plugin referenced an unknown asset")
        image = Image.open(
            self.storage.resolve_relative_path(str(relative_path))
        )
        if image.mode != mode:
            converted = image.convert(mode)
            image.close()
            image = converted
        else:
            image.load()
        return image

    def _asset_record(self, asset_id: str) -> AssetRecord:
        record = self.storage.get_record(asset_id)
        if record is None:
            raise JobConflict("plugin referenced an unknown asset")
        return record

    def _completed_step_asset_id(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
        *,
        step_kind: str,
        role: str,
    ) -> str | None:
        with self.engine.connect() as connection:
            value = connection.execute(
                select(assets.c.id)
                .join(
                    job_step_asset_outputs,
                    job_step_asset_outputs.c.asset_id == assets.c.id,
                )
                .join(
                    job_steps,
                    job_steps.c.id == job_step_asset_outputs.c.job_step_id,
                )
                .join(
                    job_items,
                    job_items.c.id == job_steps.c.job_item_id,
                )
                .where(
                    job_items.c.id == str(step["itemId"]),
                    job_items.c.job_id == fence.job_id,
                    job_steps.c.kind == step_kind,
                    job_steps.c.status == "completed",
                    job_step_asset_outputs.c.role == role,
                )
            ).scalar_one_or_none()
        return str(value) if value is not None else None

    def _atomic_hook(
        self,
        fence: AttemptFence,
        *,
        phase: str,
        scope: str,
        page_id: str,
        data: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.plugin_runtime is None:
            return dict(data)
        return self.plugin_runtime.run_atomic(
            fence,
            phase=phase,
            step=scope,
            page_id=page_id,
            data=data,
        )

    def _config(self, step: Mapping[str, Any]) -> dict[str, Any]:
        value = step.get("config", {})
        return dict(value) if isinstance(value, Mapping) else {}

    def _effective_constraints(
        self,
        step: Mapping[str, Any],
        *,
        include_current_page: bool,
    ) -> dict[str, Any]:
        config = self._config(step)
        raw = config.get("translationConstraints")
        if not isinstance(raw, Mapping):
            raise JobConflict("translation constraints snapshot is missing")
        constraints = validate_translation_constraints(raw)
        job_id = step.get("jobId")
        item_ordinal = step.get("itemOrdinal")
        if not isinstance(job_id, str) or not isinstance(item_ordinal, int):
            return constraints
        comparison = (
            job_items.c.ordinal <= item_ordinal
            if include_current_page
            else job_items.c.ordinal < item_ordinal
        )
        with self.engine.connect() as connection:
            checkpoints = list(
                connection.execute(
                    select(job_steps.c.checkpoint_json)
                    .join(
                        job_items,
                        job_items.c.id == job_steps.c.job_item_id,
                    )
                    .where(
                        job_items.c.job_id == job_id,
                        comparison,
                        job_steps.c.kind == "auto_terms",
                        job_steps.c.status == "completed",
                    )
                    .order_by(job_items.c.ordinal)
                ).scalars()
            )
        for checkpoint_json in checkpoints:
            checkpoint = (
                json.loads(checkpoint_json)
                if isinstance(checkpoint_json, str)
                else {}
            )
            delta = checkpoint.get("delta")
            if isinstance(delta, list):
                constraints, _added = with_glossary_delta(
                    constraints,
                    [
                        entry
                        for entry in delta
                        if isinstance(entry, Mapping)
                    ],
                )
        return constraints

    @staticmethod
    def _with_constraint_prompt(
        section: Mapping[str, Any],
        *,
        constraint_contexts: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        result = dict(section)
        active_contexts: list[dict[str, Any]] = []
        for context in constraint_contexts:
            raw_constraints = context.get("constraints")
            if not isinstance(raw_constraints, Mapping):
                raise JobConflict("page translation constraints are missing")
            constraints = validate_translation_constraints(raw_constraints)
            glossary = constraints["glossary"]
            non_translate = constraints["nonTranslate"]
            if not bool(glossary["enabled"]) and not bool(non_translate["enabled"]):
                continue
            active_contexts.append(
                {
                    "pageId": str(context.get("pageId", "")),
                    "glossary": (
                        glossary["entries"] if bool(glossary["enabled"]) else []
                    ),
                    "nonTranslate": (
                        non_translate["entries"]
                        if bool(non_translate["enabled"])
                        else []
                    ),
                }
            )
        if not active_contexts:
            return result
        instruction = (
            "必须遵守以下按 pageId 冻结的翻译约束。glossary 中 text/regex "
            "规则规定原文到译文的固定映射；nonTranslate 中 text/regex 匹配到的"
            "内容必须原样保留。不得把某一页稍后产生的术语反向用于更早页。\n"
            + _json({"pageConstraints": active_contexts})
        )
        base_prompt = str(result.get("prompt_content", "")).rstrip()
        result["prompt_content"] = (
            f"{base_prompt}\n\n{instruction}" if base_prompt else instruction
        )
        if bool(result.get("use_textbox_prompt")):
            textbox_prompt = str(
                result.get("textbox_prompt_content", "")
            ).rstrip()
            if textbox_prompt:
                result["textbox_prompt_content"] = (
                    f"{textbox_prompt}\n\n{instruction}"
                )
        return result

    def _with_credential(self, section: object) -> dict[str, Any]:
        result = dict(section) if isinstance(section, Mapping) else {}
        version_id = result.pop("credentialVersionId", None)
        if version_id:
            try:
                secret = self.credentials.resolve_secret(str(version_id))
            except LookupError as exc:
                raise JobConflict(
                    "frozen credential version no longer exists"
                ) from exc
            result.update(secret)
            result["credential_version_id"] = str(version_id)
        return result

    @staticmethod
    def _assert_revision(
        connection: Connection,
        page_id: str,
        expected: int,
    ) -> None:
        current = connection.execute(
            select(pages.c.document_revision).where(pages.c.id == page_id)
        ).scalar_one_or_none()
        if current != expected:
            raise JobConflict("page document revision changed")

    @staticmethod
    def _publish_pointer(
        connection: Connection,
        *,
        page_id: str,
        role: str,
        asset: AssetRecord,
        source_revision: int,
        document_revision: int,
        step_id: str,
        parent_asset_id: str | None = None,
    ) -> None:
        existing = connection.execute(
            select(page_assets.c.asset_id).where(
                page_assets.c.page_id == page_id,
                page_assets.c.role == role,
            )
        ).scalar_one_or_none()
        values = {
            "asset_id": asset.id,
            "input_source_revision": source_revision,
            "input_document_revision": document_revision,
            "parent_asset_id": parent_asset_id,
            "producer_job_step_id": step_id,
            "producer_operation_id": None,
            "producer_render_request_id": None,
        }
        if existing is None:
            connection.execute(
                insert(page_assets).values(page_id=page_id, role=role, **values)
            )
        else:
            connection.execute(
                update(page_assets)
                .where(
                    page_assets.c.page_id == page_id,
                    page_assets.c.role == role,
                )
                .values(**values)
            )
        connection.execute(
            insert(job_step_asset_outputs)
            .values(job_step_id=step_id, role=role, asset_id=asset.id)
            .prefix_with("OR REPLACE")
        )

    @staticmethod
    def _new_bubble_payload(
        *,
        coords: object,
        polygon: object,
        angle: object,
        auto_direction: object,
        textlines: object,
        style: Mapping[str, Any],
    ) -> dict[str, Any]:
        direction = str(auto_direction)
        if direction == "v":
            direction = "vertical"
        elif direction == "h":
            direction = "horizontal"
        if direction not in {"vertical", "horizontal"}:
            direction = "vertical"
        defaults = {
            "originalText": "",
            "translatedText": "",
            "textboxText": "",
            "coords": list(coords) if isinstance(coords, (list, tuple)) else [0, 0, 0, 0],
            "polygon": polygon if isinstance(polygon, list) else [],
            "fontSize": 25,
            "textDirection": direction,
            "autoTextDirection": direction,
            "textColor": "#000000",
            "fillColor": "#FFFFFF",
            "rotationAngle": float(angle or 0),
            "position": {"x": 0, "y": 0},
            "strokeEnabled": False,
            "strokeColor": "#FFFFFF",
            "strokeWidth": 2,
            "lineSpacing": 1.0,
            "textAlign": "center",
            "inpaintMethod": "solid",
            "autoFgColor": None,
            "autoBgColor": None,
            "colorConfidence": 0,
            "textlines": textlines if isinstance(textlines, list) else [],
            "ocrResult": None,
        }
        for key in defaults.keys() & style.keys():
            defaults[key] = style[key]
        if style.get("layoutDirection") in {"vertical", "horizontal"}:
            defaults["textDirection"] = style["layoutDirection"]
        return defaults
