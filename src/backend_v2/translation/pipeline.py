"""Durable per-page translation steps executed exclusively by the Worker."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import base64
import hashlib
from io import BytesIO
import json
import logging
import math
from pathlib import Path
import re
from typing import Any, Protocol
import uuid

from PIL import Image
from sqlalchemy import Engine, delete, insert, select, update
from sqlalchemy.engine import Connection

from src.backend_v2.serialization import canonical_json as _json
from src.backend_v2.timestamps import utcnow
from src.backend_v2.content.page_style import (
    PAGE_STYLE_SCHEMA_VERSION,
    rgb_to_hex,
    validate_page_style,
)
from src.backend_v2.content.translation_constraints import (
    validate_translation_constraints,
    with_glossary_delta,
)
from src.backend_v2.jobs.repository import (
    AttemptFence,
    JobConflict,
    JobQueueRepository,
)
from src.backend_v2.rendering.service import publish_png_asset
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
from src.core.config_models import validate_bubble_payload
from src.core.ocr_types import OcrResult


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

    if not isinstance(value, Mapping) or set(value) != {"request", "execution"}:
        raise ValueError("frozen OpenAI-compatible options are invalid")
    request = value["request"]
    execution = value["execution"]
    if not isinstance(request, Mapping) or set(request) != {
        "force_json_output",
        "temperature",
        "extra_body",
    }:
        raise ValueError("frozen OpenAI-compatible request options are invalid")
    if not isinstance(request["force_json_output"], bool):
        raise ValueError("force_json_output must be boolean")
    temperature = request["temperature"]
    if temperature is not None and (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or not 0 <= float(temperature) <= 2
    ):
        raise ValueError("OpenAI-compatible temperature is invalid")
    if not isinstance(request["extra_body"], Mapping):
        raise ValueError("OpenAI-compatible extra_body must be an object")
    if not isinstance(execution, Mapping) or set(execution) != {
        "use_stream",
        "rpm_limit",
        "transport_retries",
        "business_retries",
    }:
        raise ValueError("frozen OpenAI-compatible execution options are invalid")
    if not isinstance(execution["use_stream"], bool):
        raise ValueError("OpenAI-compatible use_stream must be boolean")
    for field in ("rpm_limit", "transport_retries", "business_retries"):
        option = execution[field]
        if isinstance(option, bool) or not isinstance(option, int) or option < 0:
            raise ValueError(f"OpenAI-compatible {field} is invalid")
    return OpenAICompatibleOptions.from_dict(
        {
            "request": {
                **dict(request),
                "extra_body": dict(request["extra_body"]),
            },
            "execution": dict(execution),
        }
    )


def _config_string(
    config: Mapping[str, Any],
    field: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = config.get(field)
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"translation configuration {field} must be {qualifier}")
    return value


def _optional_config_string(
    config: Mapping[str, Any],
    field: str,
) -> str | None:
    if field not in config:
        return None
    value = config[field]
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"translation configuration {field} must be a non-empty string"
        )
    return value


def _config_boolean(config: Mapping[str, Any], field: str) -> bool:
    value = config.get(field)
    if not isinstance(value, bool):
        raise ValueError(f"translation configuration {field} must be boolean")
    return value


def _validate_hq_request_pages(
    pages: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    request_pages: list[dict[str, Any]] = []
    page_ids: set[str] = set()
    for page_index, page in enumerate(pages):
        if not isinstance(page, Mapping) or set(page) != {"pageId", "bubbles"}:
            raise ValueError(
                f"HQ request page {page_index} does not match the current schema"
            )
        page_id = page["pageId"]
        if not isinstance(page_id, str) or not page_id or page_id in page_ids:
            raise ValueError("HQ request pageId values must be non-empty and unique")
        page_ids.add(page_id)
        raw_bubbles = page["bubbles"]
        if not isinstance(raw_bubbles, list):
            raise ValueError(f"HQ request page {page_id} bubbles must be an array")
        bubbles: list[dict[str, str]] = []
        bubble_ids: set[str] = set()
        for bubble_index, bubble in enumerate(raw_bubbles):
            if not isinstance(bubble, Mapping) or set(bubble) != {
                "bubbleId",
                "originalText",
                "translatedText",
                "textDirection",
            }:
                raise ValueError(
                    f"HQ request bubble {page_id}/{bubble_index} does not match "
                    "the current schema"
                )
            bubble_id = bubble["bubbleId"]
            original_text = bubble["originalText"]
            translated_text = bubble["translatedText"]
            text_direction = bubble["textDirection"]
            if (
                not isinstance(bubble_id, str)
                or not bubble_id
                or bubble_id in bubble_ids
            ):
                raise ValueError(
                    f"HQ request page {page_id} bubbleId values must be "
                    "non-empty and unique"
                )
            if not isinstance(original_text, str) or not isinstance(
                translated_text,
                str,
            ):
                raise ValueError("HQ request bubble texts must be strings")
            if text_direction not in {"vertical", "horizontal"}:
                raise ValueError("HQ request bubble textDirection is invalid")
            bubble_ids.add(bubble_id)
            bubbles.append(
                {
                    "bubbleId": bubble_id,
                    "originalText": original_text,
                    "translatedText": translated_text,
                    "textDirection": text_direction,
                }
            )
        request_pages.append({"pageId": page_id, "bubbles": bubbles})
    return request_pages


def _validate_stable_batch_result(
    payload: object,
    *,
    expected_pages: list[Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("pages"), list):
        raise ValueError("HQ response must be an object containing a pages array")
    expected_by_page = {
        page["pageId"]: {
            bubble["bubbleId"]: bubble
            for bubble in page["bubbles"]
        }
        for page in expected_pages
    }
    if len(expected_by_page) != len(expected_pages):
        raise ValueError("HQ request contains duplicate pageId values")

    parsed: dict[str, dict[str, str]] = {}
    for page in payload["pages"]:
        if not isinstance(page, Mapping):
            raise ValueError("HQ response page entries must be objects")
        page_id = page.get("pageId")
        if not isinstance(page_id, str):
            raise ValueError("HQ response pageId values must be strings")
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
            bubble_id = bubble.get("bubbleId")
            if not isinstance(bubble_id, str):
                raise ValueError("HQ response bubbleId values must be strings")
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
                expected_bubble["originalText"].strip()
                or expected_bubble["translatedText"].strip()
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
        pattern = entry["pattern"]
        match_mode = entry["matchMode"]
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
    glossary = constraints["glossary"]
    if not glossary["enabled"]:
        return []
    entries = glossary["entries"]
    warnings: list[dict[str, Any]] = []
    for bubble_index, (source_text, translated_text) in enumerate(
        zip(originals, translated)
    ):
        for entry in entries:
            source = entry["source"]
            target = entry["target"]
            if (
                _matching_fragments(
                    source_text,
                    pattern=source,
                    match_mode=entry["matchMode"],
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
        for index in sorted(unmatched):
            overlap = _box_iou(
                payload.get("coords"),
                existing[index].get("coords"),
            )
            if overlap >= minimum_iou and (
                best_index is None or overlap > best_iou
            ):
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


def _require_result_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise JobConflict(f"{label} must be an object")
    return value


def _require_text_list(value: object, *, label: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise JobConflict(f"{label} must be a string array")
    return list(value)


def _require_non_empty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise JobConflict(f"{label} must be a non-empty string")
    return value


def _require_mapping_list(
    value: object,
    *,
    label: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        raise JobConflict(f"{label} must be an object array")
    return [dict(item) for item in value]


def _validate_box(value: object, *, label: str) -> list[float | int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise JobConflict(f"{label} must contain four coordinates")
    coordinates: list[float | int] = []
    for coordinate in value:
        if (
            isinstance(coordinate, bool)
            or not isinstance(coordinate, (int, float))
            or not math.isfinite(float(coordinate))
        ):
            raise JobConflict(f"{label} coordinates must be finite numbers")
        coordinates.append(coordinate)
    if coordinates[0] >= coordinates[2] or coordinates[1] >= coordinates[3]:
        raise JobConflict(f"{label} must have positive width and height")
    return coordinates


def _validate_detection_result(
    value: object,
) -> tuple[
    list[list[float | int]],
    list[list[Any]],
    list[float | int],
    list[str],
    list[list[Any]],
    object,
]:
    result = _require_result_mapping(value, label="detection result")
    required = (
        "coords",
        "polygons",
        "angles",
        "auto_directions",
        "textlines_per_bubble",
    )
    arrays: dict[str, list[Any]] = {}
    for field in required:
        field_value = result.get(field)
        if not isinstance(field_value, list):
            raise JobConflict(f"detection result {field} must be an array")
        arrays[field] = field_value
    count = len(arrays["coords"])
    if any(len(arrays[field]) != count for field in required[1:]):
        raise JobConflict("detection result arrays are not aligned")
    coords = [
        _validate_box(item, label=f"detection result coords[{index}]")
        for index, item in enumerate(arrays["coords"])
    ]
    polygons: list[list[Any]] = []
    for index, polygon in enumerate(arrays["polygons"]):
        if not isinstance(polygon, list):
            raise JobConflict(
                f"detection result polygons[{index}] must be an array"
            )
        polygons.append(polygon)
    angles: list[float | int] = []
    for index, angle in enumerate(arrays["angles"]):
        if (
            isinstance(angle, bool)
            or not isinstance(angle, (int, float))
            or not math.isfinite(float(angle))
        ):
            raise JobConflict(
                f"detection result angles[{index}] must be a finite number"
            )
        angles.append(angle)
    directions: list[str] = []
    for index, direction in enumerate(arrays["auto_directions"]):
        if direction not in {"v", "h", "vertical", "horizontal"}:
            raise JobConflict(
                f"detection result auto_directions[{index}] is invalid"
            )
        directions.append(direction)
    textlines: list[list[Any]] = []
    for index, lines in enumerate(arrays["textlines_per_bubble"]):
        if not isinstance(lines, list):
            raise JobConflict(
                f"detection result textlines_per_bubble[{index}] "
                "must be an array"
            )
        textlines.append(lines)
    return (
        coords,
        polygons,
        angles,
        directions,
        textlines,
        result.get("raw_mask"),
    )


def _validate_detected_payloads(value: object) -> list[dict[str, Any]]:
    payloads = _require_mapping_list(value, label="detected bubbles")
    try:
        return [
            validate_bubble_payload(payload, render=False)
            for payload in payloads
        ]
    except (TypeError, ValueError) as exc:
        raise JobConflict("detected bubbles do not match the current schema") from exc


def _validate_bubble_inputs(
    value: object,
    *,
    expected_count: int,
    label: str,
    render: bool = False,
) -> list[dict[str, Any]]:
    payloads = _require_mapping_list(value, label=label)
    if len(payloads) != expected_count:
        raise JobConflict(f"{label} count does not match persisted bubbles")
    try:
        return [
            validate_bubble_payload(payload, render=render)
            for payload in payloads
        ]
    except (TypeError, ValueError) as exc:
        raise JobConflict(f"{label} does not match the current schema") from exc


def _payload_text(
    payload: Mapping[str, Any],
    field: str,
    *,
    label: str,
) -> str:
    if field not in payload:
        raise JobConflict(f"{label} is missing")
    value = payload[field]
    if not isinstance(value, str):
        raise JobConflict(f"{label} must be a string")
    return value


def _validate_rgb(value: object, *, label: str) -> list[int] | None:
    if value is None:
        return None
    try:
        rgb_to_hex(value)
    except ValueError as exc:
        raise JobConflict(f"{label} is invalid") from exc
    return list(value)  # type: ignore[arg-type]


def _validate_confidence(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise JobConflict(f"{label} must be a finite number")
    confidence = float(value)
    if not 0 <= confidence <= 1:
        raise JobConflict(f"{label} must be between zero and one")
    return confidence


def _validate_ocr_results(
    value: object,
    *,
    label: str,
) -> list[dict[str, Any]]:
    payloads = _require_mapping_list(value, label=label)
    try:
        return [OcrResult.from_dict(payload).to_dict() for payload in payloads]
    except (TypeError, ValueError) as exc:
        raise JobConflict(f"{label} does not match the current schema") from exc


def _validate_color_results(
    value: object,
    *,
    label: str,
    plugin_fields: bool,
) -> list[dict[str, Any]]:
    payloads = _require_mapping_list(value, label=label)
    foreground_field = "fgColor" if plugin_fields else "fg_color"
    background_field = "bgColor" if plugin_fields else "bg_color"
    expected_fields = {foreground_field, background_field, "confidence"}
    normalized: list[dict[str, Any]] = []
    for index, payload in enumerate(payloads):
        if set(payload) != expected_fields:
            raise JobConflict(
                f"{label}[{index}] does not match the current schema"
            )
        normalized.append(
            {
                "fg_color": _validate_rgb(
                    payload[foreground_field],
                    label=f"{label}[{index}] foreground",
                ),
                "bg_color": _validate_rgb(
                    payload[background_field],
                    label=f"{label}[{index}] background",
                ),
                "confidence": _validate_confidence(
                    payload["confidence"],
                    label=f"{label}[{index}] confidence",
                ),
            }
        )
    return normalized


class CoreTranslationAlgorithms:
    """Worker-side adapters around the current core algorithms."""

    def detect(self, image: Image.Image, config: Mapping[str, Any]) -> Mapping[str, Any]:
        from src.core.detection import (
            get_bubble_detection_result_with_auto_directions,
        )

        required_fields = {
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
        if set(config) != required_fields:
            raise ValueError("detector configuration fields are invalid")
        if config["detector_type"] not in {"default", "ctd", "yolo"}:
            raise ValueError("detector type is invalid")
        for field in (
            "enable_aux_yolo_detection",
            "enable_saber_yolo_refine",
        ):
            if not isinstance(config[field], bool):
                raise ValueError(f"detector configuration {field} must be boolean")
        for field in (
            "expand_ratio",
            "expand_top",
            "expand_bottom",
            "expand_left",
            "expand_right",
            "aux_yolo_conf_threshold",
            "aux_yolo_overlap_threshold",
            "saber_yolo_refine_overlap_threshold",
            "min_text_block_area_percent",
        ):
            value = config[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"detector configuration {field} must be finite")
        for field in (
            "aux_yolo_conf_threshold",
            "aux_yolo_overlap_threshold",
            "saber_yolo_refine_overlap_threshold",
        ):
            if not 0 <= config[field] <= 1:
                raise ValueError(f"detector configuration {field} must be from 0 to 1")
        if config["min_text_block_area_percent"] < 0:
            raise ValueError("minimum text block area cannot be negative")
        return get_bubble_detection_result_with_auto_directions(image, **dict(config))

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

        coords = [payload["coords"] for payload in bubble_payloads]
        textlines = [payload["textlines"] for payload in bubble_payloads]
        base_fields = {
            "source_language",
            "ocr_engine",
            "enable_hybrid_ocr",
            "secondary_ocr_engine",
            "hybrid_ocr_threshold",
        }
        engine = config.get("ocr_engine")
        if engine not in {
            "manga_ocr",
            "paddle_ocr",
            "paddleocr_vl",
            "baidu_ocr",
            "ai_vision",
            "48px_ocr",
        }:
            raise ValueError("OCR engine is invalid")
        expected_fields = set(base_fields)
        if engine == "baidu_ocr":
            expected_fields.update(
                {
                    "baidu_api_key",
                    "baidu_secret_key",
                    "baidu_version",
                    "baidu_ocr_language",
                    "credential_version_id",
                }
            )
        elif engine == "ai_vision":
            expected_fields.update(
                {
                    "ai_vision_provider",
                    "ai_vision_model_name",
                    "ai_vision_ocr_prompt",
                    "ai_vision_prompt_mode",
                    "custom_ai_vision_base_url",
                    "ai_vision_min_image_size",
                    "ai_vision_openai_options",
                }
            )
            credential_fields = {
                "ai_vision_api_key",
                "credential_version_id",
            }
            present_credential_fields = set(config).intersection(credential_fields)
            if present_credential_fields and present_credential_fields != credential_fields:
                raise ValueError("AI vision OCR credential fields are incomplete")
            expected_fields.update(present_credential_fields)
        if set(config) != expected_fields:
            raise ValueError("OCR configuration fields are invalid")
        if not isinstance(config["source_language"], str) or not config["source_language"]:
            raise ValueError("OCR source language is invalid")
        if not isinstance(config["enable_hybrid_ocr"], bool):
            raise ValueError("hybrid OCR flag must be boolean")
        if (
            not isinstance(config["secondary_ocr_engine"], str)
            or not config["secondary_ocr_engine"]
        ):
            raise ValueError("secondary OCR engine is invalid")
        threshold = config["hybrid_ocr_threshold"]
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or not 0 <= threshold <= 1
        ):
            raise ValueError("hybrid OCR threshold is invalid")

        kwargs = dict(config)
        if engine == "ai_vision":
            kwargs["ai_vision_openai_options"] = _openai_options(
                config["ai_vision_openai_options"]
            )
        kwargs["textlines_per_bubble"] = textlines
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

        coords = [payload["coords"] for payload in bubble_payloads]
        textlines = [payload["textlines"] for payload in bubble_payloads]
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
        provider = _config_string(config, "provider")
        api_key = _optional_config_string(config, "api_key")
        model_name = _config_string(config, "model_name")
        custom_base_url = _config_string(
            config,
            "custom_base_url",
            allow_empty=True,
        )
        credential_version_id = _optional_config_string(
            config,
            "credential_version_id",
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
            provider=provider,
            api_key=api_key,
            model=model_name,
            credential_version_id=credential_version_id,
            messages=[{"role": "user", "content": rendered_prompt}],
            base_url=custom_base_url or None,
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

        provider = _config_string(config, "provider")
        target_language = _config_string(config, "target_language")
        translation_mode = _config_string(config, "translation_mode")
        if translation_mode not in {"batch", "single"}:
            raise ValueError("unsupported translation mode")
        openai_options = _openai_options(config.get("openai_options"))
        enable_debug_logs = _config_boolean(config, "enable_debug_logs")
        api_key = _optional_config_string(config, "api_key")
        model_name = _config_string(
            config,
            "model_name",
            allow_empty=True,
        )
        custom_base_url = _config_string(
            config,
            "custom_base_url",
            allow_empty=True,
        )
        credential_version_id = _optional_config_string(
            config,
            "credential_version_id",
        )
        prompt_content = _config_string(
            config,
            "prompt_content",
            allow_empty=True,
        )
        textbox_prompt = _config_string(
            config,
            "textbox_prompt_content",
            allow_empty=True,
        )
        use_textbox_prompt = _config_boolean(config, "use_textbox_prompt")

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
                "api_key": api_key,
                "model_name": model_name,
                "prompt_content": prompt,
                "custom_base_url": custom_base_url,
                "openai_options": options,
                "credential_version_id": credential_version_id,
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
            prompt_content,
            openai_options,
            label="标准翻译",
        )
        textbox: list[str] = []
        if use_textbox_prompt and textbox_prompt:
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
        request_pages = _validate_hq_request_pages(pages)
        target_language = _config_string(config, "target_language")
        prompt = _config_string(
            config,
            "prompt_content",
            allow_empty=True,
        ).strip()
        provider = _config_string(config, "provider")
        api_key = _optional_config_string(config, "api_key")
        model_name = _config_string(config, "model_name")
        custom_base_url = _config_string(
            config,
            "custom_base_url",
            allow_empty=True,
        )
        credential_version_id = _optional_config_string(
            config,
            "credential_version_id",
        )
        enable_debug_logs = _config_boolean(config, "enable_debug_logs")
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
                            "targetLanguage": target_language,
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
        messages: list[dict[str, Any]] = []
        if prompt:
            messages.append({"role": "system", "content": prompt})
        messages.append({"role": "user", "content": content})
        options = _openai_options(config.get("openai_options"))
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
                api_key=api_key,
                model=model_name,
                credential_version_id=credential_version_id,
                base_url=custom_base_url or None,
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

        common_fields = {
            "disable_resize",
            "lama_model",
            "mask_box_expand_ratio",
            "mask_dilate_size",
            "method",
        }
        method = config.get("method")
        required_fields = (
            common_fields | {"fill_color"}
            if method == "solid"
            else common_fields
        )
        if method not in {"solid", "lama"} or set(config) != required_fields:
            raise ValueError("inpainting configuration fields are invalid")
        coords = [payload["coords"] for payload in bubble_payloads]
        polygons = [payload["polygon"] for payload in bubble_payloads]
        repaired = inpaint_bubbles(
            image,
            coords,
            method=method,
            fill_color=config.get("fill_color"),
            bubble_polygons=polygons,
            precise_mask=(
                np.array(precise_mask, dtype=np.uint8)
                if precise_mask is not None
                else None
            ),
            mask_dilate_size=config["mask_dilate_size"],
            mask_box_expand_ratio=config["mask_box_expand_ratio"],
            lama_model=config["lama_model"],
            disable_resize=config["disable_resize"],
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

        if config:
            raise ValueError("render configuration fields are invalid")
        states = [BubbleState.from_dict(payload) for payload in bubble_payloads]
        rendered = clean_image.copy()
        try:
            render_bubbles_unified(rendered, states)
        except Exception:
            rendered.close()
            raise
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
        if not isinstance(page_id, str) or not page_id:
            raise ValueError("translation step has no page")
        kind = step.get("stepKind")
        if not isinstance(kind, str):
            raise ValueError("translation step kind is invalid")
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
        kind = steps[0].get("stepKind")
        step_ordinal = steps[0].get("stepOrdinal")
        if not isinstance(kind, str):
            raise ValueError("translation batch step kind is invalid")
        if (
            isinstance(step_ordinal, bool)
            or not isinstance(step_ordinal, int)
            or step_ordinal < 1
        ):
            raise ValueError("translation batch step ordinal is invalid")
        if kind not in {"hq_translate", "proofread"}:
            raise ValueError(f"unsupported translation batch step: {kind}")
        if any(
            step.get("stepKind") != kind
            or step.get("stepOrdinal") != step_ordinal
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
            section = self._with_credential(config.get("translation"))
            mode = "hq_translate"
        target_language = config.get("targetLanguage")
        if not isinstance(target_language, str) or not target_language:
            raise JobConflict("frozen target language is invalid")
        section["target_language"] = target_language

        prepared: list[
            tuple[Mapping[str, Any], PageSnapshot, list[dict[str, Any]]]
        ] = []
        for step in steps:
            page_id = step.get("pageId")
            if not isinstance(page_id, str) or not page_id:
                raise ValueError("translation batch step has no page")
            snapshot = self._snapshot(page_id)
            bubble_payloads = []
            for bubble_id, payload in zip(snapshot.bubble_ids, snapshot.bubbles):
                translated_text = _payload_text(
                    payload,
                    "translatedText",
                    label="persisted translated text",
                )
                if kind == "proofread" and not translated_text.strip():
                    continue
                original_text = _payload_text(
                    payload,
                    "originalText",
                    label="persisted original text",
                )
                text_direction = payload.get("textDirection")
                if text_direction not in {"vertical", "horizontal"}:
                    raise JobConflict("persisted text direction is invalid")
                bubble_payloads.append(
                    {
                        "bubbleId": bubble_id,
                        "originalText": original_text,
                        "translatedText": translated_text,
                        "textDirection": text_direction,
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
            context["pageId"]: context for context in constraint_contexts
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
                            bubble["originalText"]
                            for bubble in bubble_payloads
                        ],
                        "translations": [
                            bubble["translatedText"]
                            for bubble in bubble_payloads
                        ],
                    },
                )
                original_texts = _require_text_list(
                    before.get("originalTexts"),
                    label="AI translation plugin original texts",
                )
                current_translations = _require_text_list(
                    before.get("translations"),
                    label="AI translation plugin current translations",
                )
                if (
                    len(original_texts) != len(bubble_payloads)
                    or len(current_translations) != len(bubble_payloads)
                ):
                    raise JobConflict(
                        "AI translation plugin result count does not match bubbles"
                    )
                for index, bubble in enumerate(bubble_payloads):
                    bubble["originalText"] = original_texts[index]
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
                        bubble["originalText"],
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
                            bubble["translatedText"],
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
                    restore_by_bubble[bubble["bubbleId"]] = (
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

        result = _require_result_mapping(result, label="HQ batch result")
        parsed = result.get("pages")
        if not isinstance(parsed, Mapping):
            raise JobConflict("HQ batch returned no validated page mapping")
        if not all(
            isinstance(page_id, str)
            and isinstance(bubble_results, Mapping)
            for page_id, bubble_results in parsed.items()
        ):
            raise JobConflict("HQ batch page mapping is invalid")
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
                ]
            },
            expected_pages=request_pages,
        )
        raw_content_value = result.get("rawContent")
        if not isinstance(raw_content_value, str) or not raw_content_value:
            raise JobConflict("HQ batch rawContent must be a non-empty string")
        raw_content = raw_content_value
        batch_id = str(uuid.uuid4())
        raw_asset = self.storage.publish_bytes(
            raw_content.encode("utf-8"),
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
            ordered_ids = [bubble["bubbleId"] for bubble in requested_bubbles]
            after = self._atomic_hook(
                fence,
                phase="after",
                scope="ai_translate",
                page_id=snapshot.page_id,
                data={
                    "pageId": snapshot.page_id,
                    "originalTexts": [
                        bubble["originalText"]
                        for bubble in requested_bubbles
                    ],
                    "translations": [
                        translated_by_id[bubble_id]
                        for bubble_id in ordered_ids
                    ],
                },
            )
            plugin_translations = _require_text_list(
                after["translations"],
                label="AI translation plugin translations",
            )
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
                bubble_id = requested["bubbleId"]
                updated[index_by_id[bubble_id]]["translatedText"] = (
                    translated_by_id[bubble_id]
                )
            warnings = _translation_constraint_warnings(
                [
                    bubble["originalText"]
                    for bubble in requested_bubbles
                ],
                [
                    translated_by_id[bubble["bubbleId"]]
                    for bubble in requested_bubbles
                ],
                constraint_context_by_page[snapshot.page_id]["constraints"],
            )
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
        style_defaults, task_font_id = self._task_text_style(step, snapshot)
        source = self._bound_asset(fence, step, page_id, "source")
        detector_config = self._config(step).get("detector")
        if not isinstance(detector_config, Mapping):
            raise JobConflict("detector configuration is invalid")
        before = self._atomic_hook(
            fence,
            phase="before",
            scope="detect",
            page_id=page_id,
            data={
                "pageId": page_id,
                "sourceAssetId": str(source["id"]),
                "detectorConfig": dict(detector_config),
            },
        )
        source_asset_id = _require_non_empty_string(
            before.get("sourceAssetId"),
            label="detection source asset",
        )
        detector_config = before.get("detectorConfig")
        if not isinstance(detector_config, Mapping):
            raise JobConflict("detector plugin configuration is invalid")
        image = self._open_asset(source_asset_id, "RGB")
        source_size = image.size
        try:
            result = self.algorithms.detect(
                image,
                dict(detector_config),
            )
        finally:
            image.close()
        (
            coords,
            polygons,
            angles,
            directions,
            textlines,
            mask,
        ) = _validate_detection_result(result)
        payloads = _preserve_detected_text([
            self._new_bubble_payload(
                coords=value,
                polygon=polygons[index],
                angle=angles[index],
                auto_direction=directions[index],
                textlines=textlines[index],
                style=style_defaults,
            )
            for index, value in enumerate(coords)
        ], snapshot.bubbles)
        mask_record: AssetRecord | None = None
        if isinstance(mask, Image.Image):
            try:
                if mask.size != source_size:
                    raise JobConflict("detection mask size does not match source image")
                mask_record = publish_png_asset(self.storage, mask, mode="L")
            finally:
                mask.close()
        elif mask is not None:
            mask_image = Image.fromarray(mask)
            try:
                if mask_image.size != source_size:
                    raise JobConflict("detection mask size does not match source image")
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
        payloads = _validate_detected_payloads(after.get("bubbles"))
        mask_asset_id = after.get("textMaskAssetId")
        if mask_asset_id is not None:
            mask_asset_id = _require_non_empty_string(
                mask_asset_id,
                label="detection mask asset",
            )
        mask_record = (
            self._asset_record(mask_asset_id)
            if mask_asset_id is not None
            else None
        )
        if mask_record is not None:
            stored_mask = self._open_asset(mask_record.id, "L")
            try:
                if stored_mask.size != source_size:
                    raise JobConflict(
                        "detection mask asset size does not match source image"
                    )
            finally:
                stored_mask.close()
        new_revision = snapshot.document_revision + 1

        def publish(connection: Connection) -> None:
            self._assert_revision(
                connection, page_id, snapshot.document_revision
            )
            standalone_detect = step.get("jobKind") == "detect"
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
                _payload_text(
                    payload,
                    "translatedText",
                    label="detected translated text",
                ).strip()
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
                            "font_id": task_font_id,
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
            changed = connection.execute(
                update(pages)
                .where(
                    pages.c.id == page_id,
                    pages.c.document_revision == snapshot.document_revision,
                )
                .values(**page_values)
            )
            if changed.rowcount != 1:
                raise JobConflict(
                    "page revision changed during detection publication"
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
        raw_section = config.get("ocr")
        if not isinstance(raw_section, Mapping):
            raise JobConflict("OCR configuration is invalid")
        before = self._atomic_hook(
            fence,
            phase="before",
            scope="ocr",
            page_id=page_id,
            data={
                "pageId": page_id,
                "sourceAssetId": str(source["id"]),
                "bubbles": [dict(value) for value in snapshot.bubbles],
                "ocrConfig": dict(raw_section),
            },
        )
        input_bubbles = _validate_bubble_inputs(
            before["bubbles"],
            expected_count=len(snapshot.bubbles),
            label="OCR input bubbles",
        )
        source_asset_id = _require_non_empty_string(
            before.get("sourceAssetId"),
            label="OCR source asset",
        )
        image = self._open_asset(source_asset_id, "RGB")
        try:
            section = self._with_credential(before.get("ocrConfig"))
            result = self.algorithms.ocr(
                image,
                input_bubbles,
                section,
            )
        finally:
            image.close()
        result = _require_result_mapping(result, label="OCR result")
        texts = _require_text_list(result.get("texts"), label="OCR texts")
        details = _validate_ocr_results(
            result.get("results"),
            label="OCR details",
        )
        if len(texts) != len(snapshot.bubbles) or len(details) != len(texts):
            raise JobConflict("OCR result count does not match persisted bubbles")
        after = self._atomic_hook(
            fence,
            phase="after",
            scope="ocr",
            page_id=page_id,
            data={
                "pageId": page_id,
                "originalTexts": texts,
                "ocrResults": details,
            },
        )
        texts = _require_text_list(
            after.get("originalTexts"),
            label="OCR plugin texts",
        )
        details = _validate_ocr_results(
            after.get("ocrResults"),
            label="OCR plugin details",
        )
        updated = [dict(payload) for payload in snapshot.bubbles]
        if len(texts) != len(updated) or len(details) != len(updated):
            raise JobConflict("OCR result count does not match persisted bubbles")
        for index, payload in enumerate(updated):
            payload["originalText"] = texts[index]
            payload["ocrResult"] = details[index]
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
        style_defaults, _task_font_id = self._task_text_style(step, snapshot)
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
        input_bubbles = _validate_bubble_inputs(
            before["bubbles"],
            expected_count=len(snapshot.bubbles),
            label="color input bubbles",
        )
        source_asset_id = _require_non_empty_string(
            before.get("sourceAssetId"),
            label="color source asset",
        )
        image = self._open_asset(source_asset_id, "RGB")
        try:
            colors = self.algorithms.colors(
                image,
                input_bubbles,
            )
        finally:
            image.close()
        colors = _validate_color_results(
            colors,
            label="color result",
            plugin_fields=False,
        )
        if len(colors) != len(snapshot.bubbles):
            raise JobConflict("color result count does not match persisted bubbles")
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
                            if color["fg_color"] is not None
                            else None
                        ),
                        "bgColor": (
                            list(color["bg_color"])
                            if color["bg_color"] is not None
                            else None
                        ),
                        "confidence": color["confidence"],
                    }
                    for color in colors
                ],
            },
        )
        colors = _validate_color_results(
            after.get("colors"),
            label="color plugin result",
            plugin_fields=True,
        )
        if len(colors) != len(snapshot.bubbles):
            raise JobConflict("color result count does not match persisted bubbles")
        updated = [dict(payload) for payload in snapshot.bubbles]
        uses_auto_color = bool(style_defaults["useAutoTextColor"])
        for payload, color in zip(updated, colors):
            foreground = color["fg_color"]
            background = color["bg_color"]
            payload["autoFgColor"] = foreground
            payload["autoBgColor"] = background
            payload["colorConfidence"] = color["confidence"]
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
        all_texts = [
            _payload_text(
                payload,
                "originalText",
                label="persisted original text",
            ).strip()
            for payload in snapshot.bubbles
        ]
        texts = [text for text in all_texts if text]
        effective_before = self._effective_constraints(
            step,
            include_current_page=False,
        )
        glossary = effective_before["glossary"]
        baseline_revision = config.get("translationConstraintRevision")
        if (
            isinstance(baseline_revision, bool)
            or not isinstance(baseline_revision, int)
            or baseline_revision < 0
        ):
            raise JobConflict("translation constraint revision is invalid")
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

        section = self._with_credential(config.get("translation"))
        result = self.algorithms.extract_terms(
            texts,
            section,
            prompt=str(glossary["autoExtractPrompt"]),
        )
        result = _require_result_mapping(
            result,
            label="automatic term extraction result",
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
            source_value = raw.get("source")
            target_value = raw.get("target")
            note_value = raw.get("note", "")
            if not isinstance(source_value, str) or not isinstance(
                target_value,
                str,
            ):
                raise JobConflict(
                    f"automatic term candidate {index} source and target "
                    "must be strings"
                )
            if not isinstance(note_value, str):
                raise JobConflict(
                    f"automatic term candidate {index} note must be a string"
                )
            source = source_value.strip()
            target = target_value.strip()
            if not source or not target:
                raise JobConflict(
                    f"automatic term candidate {index} requires source and target"
                )
            candidates.append(
                {
                    "source": source,
                    "target": target,
                    "note": note_value.strip(),
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
        if not isinstance(raw_content_value, str):
            raise JobConflict(
                "automatic term extraction rawContent must be a string"
            )
        raw_content = raw_content_value
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
            _payload_text(
                payload,
                "originalText",
                label="persisted original text",
            )
            for payload in snapshot.bubbles
        ]
        constraints = self._effective_constraints(
            step,
            include_current_page=True,
        )
        raw_section = config.get("translation")
        if not isinstance(raw_section, Mapping):
            raise JobConflict("translation provider configuration is invalid")
        before = self._atomic_hook(
            fence,
            phase="before",
            scope="translate",
            page_id=page_id,
            data={
                "pageId": page_id,
                "originalTexts": persisted_texts,
                "translationConfig": dict(raw_section),
            },
        )
        texts = _require_text_list(
            before.get("originalTexts"),
            label="translation plugin original texts",
        )
        if len(texts) != len(snapshot.bubbles):
            raise JobConflict(
                "before_translate original text count does not match bubbles"
            )
        section = self._with_constraint_prompt(
            self._with_credential(
                before.get("translationConfig")
            ),
            constraint_contexts=[
                {
                    "pageId": page_id,
                    "constraints": constraints,
                }
            ],
        )
        target_language = config.get("targetLanguage")
        if not isinstance(target_language, str) or not target_language:
            raise JobConflict("frozen target language is invalid")
        section["target_language"] = target_language
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
        result = _require_result_mapping(result, label="translation result")
        raw_translated = _require_text_list(
            result.get("translated"),
            label="translation result translated",
        )
        if len(raw_translated) != len(restore_by_index):
            raise JobConflict("translation result count does not match bubbles")
        translated = [
            _restore_non_translate_text(value, restore_by_index[index])
            for index, value in enumerate(raw_translated)
        ]
        raw_textbox_value = result.get("textbox")
        raw_textbox = _require_text_list(
            raw_textbox_value,
            label="translation result textbox",
        )
        if raw_textbox and len(raw_textbox) != len(restore_by_index):
            raise JobConflict("textbox translation result count does not match bubbles")
        textbox = [
            (
                _restore_non_translate_text(value, restore_by_index[index])
                if value
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
                "textboxTexts": textbox,
            },
        )
        translated = _require_text_list(
            after.get("translations"),
            label="translation plugin translated texts",
        )
        textbox = _require_text_list(
            after.get("textboxTexts"),
            label="translation plugin textbox texts",
        )
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
        for index, payload in enumerate(updated):
            payload["translatedText"] = translated[index]
            payload["textboxText"] = (
                textbox[index] if index < len(textbox) else ""
            )
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
        style_defaults, _task_font_id = self._task_text_style(step, snapshot)
        source = self._bound_asset(fence, step, page_id, "source")
        precise_mask_asset_id = self._completed_step_asset_id(
            fence,
            step,
            step_kind="detect",
            role="text_mask",
        )
        inpaint_method = str(style_defaults["inpaintMethod"])
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
                "method": inpaint_method,
                "fillColor": (
                    style_defaults["fillColor"]
                    if inpaint_method == "solid"
                    else None
                ),
            },
        )
        input_bubbles = _validate_bubble_inputs(
            before["bubbles"],
            expected_count=len(snapshot.bubbles),
            label="inpaint input bubbles",
        )
        input_asset_id = _require_non_empty_string(
            before.get("inputAssetId"),
            label="inpaint input asset",
        )
        image = self._open_asset(input_asset_id, "RGB")
        source_size = image.size
        precise_mask: Image.Image | None = None
        try:
            mask_asset_id = before.get("textMaskAssetId")
            if mask_asset_id is not None:
                mask_asset_id = _require_non_empty_string(
                    mask_asset_id,
                    label="inpaint mask asset",
                )
                precise_mask = self._open_asset(
                    mask_asset_id,
                    "L",
                )
                if precise_mask.size != source_size:
                    raise JobConflict(
                        "inpaint mask size does not match source image"
                    )
            # Precise-mask expansion is frozen as task configuration. The
            # repair method is a page fact bound when this item starts; only
            # solid repair consumes the page fill color.
            raw_inpainting = self._config(step).get("inpainting")
            if not isinstance(raw_inpainting, Mapping):
                raise JobConflict("inpainting configuration is invalid")
            inpainting = dict(raw_inpainting)
            method = before["method"]
            if method not in {"solid", "lama_mpe", "litelama"}:
                raise JobConflict("inpainting method is invalid")
            inpainting["method"] = "solid" if method == "solid" else "lama"
            inpainting["lama_model"] = (
                "litelama" if method == "litelama" else "lama_mpe"
            )
            if method == "solid":
                inpainting["fill_color"] = validate_page_style(
                    {"fillColor": before.get("fillColor")},
                    partial=True,
                )["fillColor"]
            repaired = self.algorithms.repair(
                image,
                input_bubbles,
                inpainting,
                precise_mask=precise_mask,
            )
        finally:
            image.close()
            if precise_mask is not None:
                precise_mask.close()
        if not isinstance(repaired, Image.Image):
            raise JobConflict("inpainting did not return an image")
        try:
            if repaired.size != source_size:
                raise JobConflict(
                    "inpainting result size does not match source image"
                )
            record = publish_png_asset(self.storage, repaired, mode="RGB")
        finally:
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
        clean_asset_id = _require_non_empty_string(
            after.get("cleanAssetId"),
            label="inpaint clean asset",
        )
        record = self._asset_record(clean_asset_id)

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
        style_defaults, task_font_id = self._task_text_style(step, snapshot)
        has_task_text_style = isinstance(
            self._config(step).get("textStyleSnapshot"),
            Mapping,
        )
        from src.backend_v2.rendering.fonts import (
            materialize_render_payloads,
        )

        job_kind = step.get("jobKind")
        if not isinstance(job_kind, str):
            raise JobConflict("translation job kind is invalid")
        initialize_auto_fields: frozenset[str]
        if job_kind == "translation":
            initialize_auto_fields = frozenset(
                {"fontSize", "layoutDirection", "textColor", "fillColor"}
            )
        elif job_kind == "style_apply":
            config = self._config(step)
            selected = config.get("selectedFields")
            frozen = config.get("frozenStyle")
            if (
                not isinstance(selected, list)
                or not all(isinstance(field, str) for field in selected)
                or not isinstance(frozen, Mapping)
            ):
                raise JobConflict("style apply configuration is invalid")
            auto_font_size = frozen.get("autoFontSize")
            if "fontSize" in selected and not isinstance(auto_font_size, bool):
                raise JobConflict("style apply auto font size is invalid")
            initialize_auto_fields = (
                frozenset({"fontSize"})
                if (
                    "fontSize" in selected
                    and auto_font_size
                )
                else frozenset()
            )
        elif job_kind in {"remove_text", "detect", "text_import"}:
            initialize_auto_fields = frozenset()
        else:
            raise JobConflict(f"unsupported render job kind: {job_kind}")
        with self.engine.connect() as connection:
            projected = materialize_render_payloads(
                connection,
                self.storage,
                page_id,
                initialize_auto_fields=initialize_auto_fields,
                style_defaults_override=style_defaults,
                override_font_id=has_task_text_style,
                font_id_override=task_font_id,
            )
        render_payloads = [
            render_payload
            for _bubble_id, _payload, render_payload in projected
        ]
        persisted_payloads = [
            (bubble_id, payload)
            for bubble_id, payload, _render_payload in projected
        ]
        if job_kind in {"translation", "remove_text"}:
            input_asset = self._bound_asset(
                fence,
                step,
                page_id,
                "clean",
            )
        else:
            with self.engine.connect() as connection:
                has_clean_asset = connection.execute(
                    select(page_assets.c.asset_id).where(
                        page_assets.c.page_id == page_id,
                        page_assets.c.role == "clean",
                    )
                ).scalar_one_or_none()
            input_asset = self._bound_asset(
                fence,
                step,
                page_id,
                "clean" if has_clean_asset is not None else "source",
            )
        if job_kind in {"translation", "remove_text"}:
            render_section = self._config(step).get("render")
            if not isinstance(render_section, Mapping):
                raise JobConflict("render configuration is invalid")
        else:
            render_section = {}
        before = self._atomic_hook(
            fence,
            phase="before",
            scope="render",
            page_id=page_id,
            data={
                "pageId": page_id,
                "inputAssetId": str(input_asset["id"]),
                "bubbles": render_payloads,
                "renderConfig": dict(render_section),
            },
        )
        render_bubbles = _validate_bubble_inputs(
            before["bubbles"],
            expected_count=len(snapshot.bubbles),
            label="render input bubbles",
            render=True,
        )
        input_asset_id = _require_non_empty_string(
            before.get("inputAssetId"),
            label="render input asset",
        )
        render_config = before.get("renderConfig")
        if not isinstance(render_config, Mapping):
            raise JobConflict("render plugin configuration is invalid")
        clean = self._open_asset(input_asset_id, "RGB")
        clean_size = clean.size
        try:
            rendered = self.algorithms.render(
                clean,
                render_bubbles,
                dict(render_config),
            )
        finally:
            clean.close()
        if not isinstance(rendered, Image.Image):
            raise JobConflict("renderer did not return an image")
        try:
            if rendered.size != clean_size:
                raise JobConflict("render result size does not match input image")
            translated = publish_png_asset(self.storage, rendered, mode="RGB")
        finally:
            rendered.close()
        after = self._atomic_hook(
            fence,
            phase="after",
            scope="render",
            page_id=page_id,
            data={
                "pageId": page_id,
                "translatedAssetId": translated.id,
                "documentRevision": snapshot.document_revision,
            },
        )
        translated_asset_id = _require_non_empty_string(
            after.get("translatedAssetId"),
            label="rendered asset",
        )
        translated = self._asset_record(translated_asset_id)

        def publish(connection: Connection) -> None:
            self._assert_revision(
                connection, page_id, snapshot.document_revision
            )
            for bubble_id, payload in persisted_payloads:
                values: dict[str, object] = {
                    "payload_json": _json(payload),
                    "updated_revision": snapshot.document_revision,
                }
                if has_task_text_style:
                    values["font_id"] = task_font_id
                changed = connection.execute(
                    update(bubbles)
                    .where(
                        bubbles.c.id == bubble_id,
                        bubbles.c.page_id == page_id,
                        bubbles.c.updated_revision
                        == snapshot.document_revision,
                    )
                    .values(**values)
                )
                if changed.rowcount != 1:
                    raise JobConflict(
                        "bubble changed during render publication"
                    )
            connection.execute(
                insert(job_step_asset_outputs).values(
                    job_step_id=str(step["stepId"]),
                    role="translated",
                    asset_id=translated.id,
                )
            )

        checkpoint = {
            "translatedAssetId": translated.id,
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
                        job_step_asset_outputs.c.role == "translated",
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
        if translated is None:
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
            changed = connection.execute(
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
            if changed.rowcount != 1:
                raise JobConflict("page revision changed during save publication")

        checkpoint = {
            "translatedAssetId": translated.id,
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
            current_ids = tuple(str(row) for row in rows)
            if current_ids != snapshot.bubble_ids:
                raise JobConflict("bubble set changed before step publication")
            if len(payloads) != len(snapshot.bubble_ids):
                raise JobConflict("bubble result count changed before publication")
            for bubble_id, payload in zip(current_ids, payloads):
                changed = connection.execute(
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
                if changed.rowcount != 1:
                    raise JobConflict(
                        "bubble changed during step publication"
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
            if page["page_style_schema_version"] != PAGE_STYLE_SCHEMA_VERSION:
                raise JobConflict("page style schema version is not current")
            rows = list(
                connection.execute(
                    select(
                        bubbles.c.id,
                        bubbles.c.payload_json,
                        bubbles.c.payload_schema_version,
                        bubbles.c.updated_revision,
                    )
                    .where(bubbles.c.page_id == page_id)
                    .order_by(bubbles.c.ordinal)
                )
            )
            parsed_bubbles: list[dict[str, Any]] = []
            for row in rows:
                if row.payload_schema_version != 1:
                    raise JobConflict("bubble payload schema version is not current")
                if row.updated_revision != page["document_revision"]:
                    raise JobConflict("bubble revision does not match page document")
                try:
                    payload = validate_bubble_payload(
                        json.loads(row.payload_json),
                        render=False,
                    )
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise JobConflict(
                        "bubble payload does not match the current schema"
                    ) from exc
                parsed_bubbles.append(payload)
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
                bubbles=tuple(parsed_bubbles),
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
        item_id = step.get("itemId")
        if not isinstance(item_id, str) or not item_id:
            raise JobConflict("translation step item ID is invalid")
        return self.jobs.bind_item_inputs(
            fence,
            item_id=item_id,
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
        try:
            image.load()
            if image.mode != mode:
                converted = image.convert(mode)
                image.close()
                image = converted
        except Exception:
            image.close()
            raise
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
        value = step.get("config")
        if not isinstance(value, Mapping):
            raise JobConflict("translation job configuration is invalid")
        return dict(value)

    def _task_text_style(
        self,
        step: Mapping[str, Any],
        snapshot: PageSnapshot,
    ) -> tuple[dict[str, Any], str | None]:
        value = self._config(step).get("textStyleSnapshot")
        if not isinstance(value, Mapping):
            return dict(snapshot.style_defaults), None
        if set(value) != {
            "sourcePageId",
            "sourceDocumentRevision",
            "defaultFontId",
            "pageStyleDefaults",
        }:
            raise JobConflict("frozen text style snapshot fields are invalid")
        source_page_id = value["sourcePageId"]
        source_revision = value["sourceDocumentRevision"]
        if not isinstance(source_page_id, str) or not source_page_id:
            raise JobConflict("frozen text style source page is invalid")
        if (
            isinstance(source_revision, bool)
            or not isinstance(source_revision, int)
            or source_revision < 1
        ):
            raise JobConflict("frozen text style source revision is invalid")
        defaults = validate_page_style(
            value["pageStyleDefaults"],
            partial=False,
        )
        default_font_id = value["defaultFontId"]
        if default_font_id is not None and (
            not isinstance(default_font_id, str) or not default_font_id
        ):
            raise JobConflict("frozen text style font is invalid")
        return defaults, default_font_id

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
        if (
            not isinstance(job_id, str)
            or not job_id
            or isinstance(item_ordinal, bool)
            or not isinstance(item_ordinal, int)
            or item_ordinal < 1
        ):
            raise JobConflict("translation step ordering metadata is invalid")
        comparison = (
            job_items.c.ordinal <= item_ordinal
            if include_current_page
            else job_items.c.ordinal < item_ordinal
        )
        with self.engine.connect() as connection:
            checkpoints = list(
                connection.execute(
                    select(
                        job_steps.c.checkpoint_json,
                        job_steps.c.checkpoint_schema_version,
                    )
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
                ).mappings()
            )
        for row in checkpoints:
            if row["checkpoint_schema_version"] != 1:
                raise JobConflict(
                    "automatic term checkpoint schema version is invalid"
                )
            checkpoint_json = row["checkpoint_json"]
            if not isinstance(checkpoint_json, str):
                raise JobConflict("automatic term checkpoint is invalid")
            try:
                checkpoint = json.loads(checkpoint_json)
            except json.JSONDecodeError as exc:
                raise JobConflict(
                    "automatic term checkpoint is invalid"
                ) from exc
            if not isinstance(checkpoint, Mapping):
                raise JobConflict("automatic term checkpoint is invalid")
            delta = checkpoint.get("delta")
            if not isinstance(delta, list) or not all(
                isinstance(entry, Mapping) for entry in delta
            ):
                raise JobConflict("automatic term checkpoint delta is invalid")
            constraints, _added = with_glossary_delta(
                constraints,
                [dict(entry) for entry in delta],
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
            page_id = context.get("pageId")
            if not isinstance(page_id, str) or not page_id:
                raise JobConflict("page translation constraint ID is invalid")
            raw_constraints = context.get("constraints")
            if not isinstance(raw_constraints, Mapping):
                raise JobConflict("page translation constraints are missing")
            constraints = validate_translation_constraints(raw_constraints)
            glossary = constraints["glossary"]
            non_translate = constraints["nonTranslate"]
            if not glossary["enabled"] and not non_translate["enabled"]:
                continue
            active_contexts.append(
                {
                    "pageId": page_id,
                    "glossary": (
                        glossary["entries"] if glossary["enabled"] else []
                    ),
                    "nonTranslate": (
                        non_translate["entries"]
                        if non_translate["enabled"]
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
        base_prompt_value = result.get("prompt_content")
        if not isinstance(base_prompt_value, str):
            raise JobConflict("frozen translation prompt is invalid")
        base_prompt = base_prompt_value.rstrip()
        result["prompt_content"] = (
            f"{base_prompt}\n\n{instruction}" if base_prompt else instruction
        )
        if "use_textbox_prompt" in result:
            use_textbox_prompt = result["use_textbox_prompt"]
            textbox_prompt_value = result.get("textbox_prompt_content")
            if not isinstance(use_textbox_prompt, bool) or not isinstance(
                textbox_prompt_value,
                str,
            ):
                raise JobConflict("frozen textbox translation prompt is invalid")
            textbox_prompt = textbox_prompt_value.rstrip()
            if use_textbox_prompt and textbox_prompt:
                result["textbox_prompt_content"] = (
                    f"{textbox_prompt}\n\n{instruction}"
                )
        return result

    def _with_credential(self, section: object) -> dict[str, Any]:
        if not isinstance(section, Mapping):
            raise JobConflict("frozen provider configuration is invalid")
        result = dict(section)
        version_id = result.pop("credentialVersionId", None)
        if version_id is not None:
            if not isinstance(version_id, str) or not version_id:
                raise JobConflict("frozen credential version is invalid")
            try:
                secret = self.credentials.resolve_secret(version_id)
            except LookupError as exc:
                raise JobConflict(
                    "frozen credential version no longer exists"
                ) from exc
            result.update(secret)
            result["credential_version_id"] = version_id
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
            "parent_asset_id": None,
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
        direction = auto_direction
        if direction == "v":
            direction = "vertical"
        elif direction == "h":
            direction = "horizontal"
        if direction not in {"vertical", "horizontal"}:
            raise JobConflict("detected bubble direction is invalid")
        validated_style = validate_page_style(style, partial=False)
        text_direction = (
            direction
            if validated_style["layoutDirection"] == "auto"
            else validated_style["layoutDirection"]
        )
        payload = {
            "originalText": "",
            "translatedText": "",
            "textboxText": "",
            "coords": coords,
            "polygon": polygon,
            "fontSize": validated_style["fontSize"],
            "textDirection": text_direction,
            "autoTextDirection": direction,
            "textColor": validated_style["textColor"],
            "fillColor": validated_style["fillColor"],
            "rotationAngle": angle,
            "position": {"x": 0, "y": 0},
            "strokeEnabled": validated_style["strokeEnabled"],
            "strokeColor": validated_style["strokeColor"],
            "strokeWidth": validated_style["strokeWidth"],
            "lineSpacing": validated_style["lineSpacing"],
            "textAlign": validated_style["textAlign"],
            "inpaintMethod": validated_style["inpaintMethod"],
            "autoFgColor": None,
            "autoBgColor": None,
            "colorConfidence": 0,
            "textlines": textlines,
            "ocrResult": None,
        }
        try:
            return validate_bubble_payload(payload, render=False)
        except (TypeError, ValueError) as exc:
            raise JobConflict("detected bubble does not match the current schema") from exc
